from sklearn.cluster import KMeans
import numpy as np
import torch

class HierarchicalKMeans:
    """
    Repeatedly apply k-means until each leaf cluster has < max_leaf_size points.
    """

    def __init__(self, n_clusters=2, max_leaf_size=50, random_state=None, n_init="auto", max_iter=300):
        """
        Parameters
        ----------
        n_clusters : int
            Branching factor (k) at each split (upper bound; actual k is min(n_clusters, size)).
        max_leaf_size : int
            Target maximum number of points per leaf (leaf must have < max_leaf_size).
        random_state : int or None
            Random state passed to sklearn KMeans.
        n_init : int or "auto"
            Passed to sklearn KMeans.
        max_iter : int
            Passed to sklearn KMeans.
        """
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")
        if max_leaf_size < 2:
            raise ValueError("max_leaf_size must be >= 2")

        self.n_clusters = int(n_clusters)
        self.max_leaf_size = int(max_leaf_size)
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

        # Tree representation
        # node_id -> dict with keys:
        #   is_leaf: bool
        #   indices: np.ndarray of data indices (for leaves)
        #   centroids: np.ndarray [k, d] (for internal nodes)
        #   children: list of child node_ids (for internal nodes)
        self.tree_ = {}
        self.root_ = "root"
        self.n_features_in_ = None

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Initialize with a single root node containing all indices
        self.tree_ = {
            self.root_: {
                "is_leaf": False,
                "indices": np.arange(n_samples, dtype=int),
                "centroids": None,
                "children": []
            }
        }

        # BFS queue: nodes to attempt splitting
        queue = [self.root_]

        while queue:
            node_id = queue.pop(0)
            node = self.tree_[node_id]
            idx = node["indices"]
            size = idx.size

            # If node already small enough, make it a leaf and continue
            if size < self.max_leaf_size:
                node["is_leaf"] = True
                node["children"] = []
                node["centroids"] = None
                # keep indices for leaf
                continue

            # Determine k for this node (cannot exceed number of points)
            k = min(self.n_clusters, size)
            # If k drops below 2, we cannot split meaningfully
            if k < 2:
                node["is_leaf"] = True
                node["children"] = []
                node["centroids"] = None
                continue

            # Run k-means on the subset
            Xi = X[idx]
            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter
            )
            labels = km.fit_predict(Xi)

            # Check if the split is meaningful:
            #  - All labels identical -> no split
            #  - Any empty cluster (shouldn't happen in sklearn, but safeguard)
            unique_labels, counts = np.unique(labels, return_counts=True)
            if unique_labels.size < 2 or np.any(counts == 0):
                # Degenerate: mark as leaf to avoid infinite loop
                node["is_leaf"] = True
                node["children"] = []
                node["centroids"] = None
                continue

            # Create children
            node["is_leaf"] = False
            node["centroids"] = km.cluster_centers_
            node["children"] = []

            for lab in range(k):
                child_indices = idx[labels == lab]
                child_id = f"{node_id}/{lab}"
                self.tree_[child_id] = {
                    "is_leaf": False,        # may change after checking size
                    "indices": child_indices,
                    "centroids": None,
                    "children": []
                }
                node["children"].append(child_id)

            # Push children that still exceed the leaf limit back onto the queue
            for child_id in node["children"]:
                child = self.tree_[child_id]
                if child["indices"].size >= self.max_leaf_size:
                    queue.append(child_id)
                else:
                    # finalize as leaf right away
                    child["is_leaf"] = True

            # Internal nodes don't need to keep full indices; free memory
            node["indices"] = None

        return self

    def _traverse_one(self, x):
        """Traverse one vector x from root to a leaf using nearest-centroid routing."""
        node_id = self.root_
        while not self.tree_[node_id]["is_leaf"]:
            centroids = self.tree_[node_id]["centroids"]
            # compute distances to centroids
            # reshape x to (1, d) for broadcasting
            diffs = centroids - x.reshape(1, -1)
            dists = np.einsum("ij,ij->i", diffs, diffs)  # squared L2
            child_idx = int(np.argmin(dists))
            node_id = self.tree_[node_id]["children"][child_idx]
        return node_id

    def predict(self, X):
        """
        Assign each sample to a leaf node ID.
        Returns a list of node_id strings, one per sample.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features")

        assignments = []
        for i in range(X.shape[0]):
            assignments.append(self._traverse_one(X[i]))
        return assignments

    def leaf_members(self):
        """
        Returns a dict: leaf_node_id -> np.ndarray of member indices.
        """
        out = {}
        for nid, node in self.tree_.items():
            if node["is_leaf"]:
                out[nid] = node["indices"]
        return out

    def centroids(self):
        """
        Returns a dict: internal_node_id -> centroids (np.ndarray [k, d]).
        """
        out = {}
        for nid, node in self.tree_.items():
            if not node["is_leaf"] and node["centroids"] is not None:
                out[nid] = node["centroids"]
        return out


# ------------------- Example ------------------- #
if __name__ == "__main__":
    # Make some fake data
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(loc=[0, 0], scale=0.6, size=(200, 2)),
        rng.normal(loc=[5, 5], scale=0.6, size=(200, 2)),
        rng.normal(loc=[0, 6], scale=0.6, size=(200, 2)),
    ])

    X = torch.from_numpy(X)
    print(X)

    # Build a k-means tree with branching factor 3, and leaf size < 60
    hkm = HierarchicalKMeans(n_clusters=10, max_leaf_size=60, random_state=0)
    hkm.fit(X)

    # Show leaf sizes
    members = hkm.leaf_members()
    print("Number of leaves:", len(members))
    sizes = {nid: idxs for nid, idxs in members.items()}
    print("Leaf sizes:", sizes)

    # Predict which leaf a new point belongs to
    test_pts = np.array([[0.2, 0.1], [5.2, 5.0], [0.1, 5.9]])
    print("Assignments:", hkm.predict(test_pts))
