"""Hierarchical k-means clustering utilities used for tag tree construction."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

__all__ = ["HierarchicalKMeans"]


class HierarchicalKMeans:
    """Repeatedly apply k-means until each leaf cluster is below ``max_leaf_size``."""

    def __init__(
        self,
        n_clusters: int = 2,
        max_leaf_size: int = 50,
        *,
        random_state: int | None = None,
        n_init: int | str = "auto",
        max_iter: int = 300,
    ) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")
        if max_leaf_size < 2:
            raise ValueError("max_leaf_size must be >= 2")

        self.n_clusters = int(n_clusters)
        self.max_leaf_size = int(max_leaf_size)
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

        # node_id -> dict with:
        #   is_leaf: bool
        #   indices: np.ndarray of data indices (for leaves)
        #   centroids: np.ndarray [k, d] (for internal nodes)
        #   children: list of child node_ids (for internal nodes)
        self.tree_: Dict[str, Dict[str, Any]] = {}
        self.root_ = "root"
        self.n_features_in_: int | None = None

    def fit(self, X: torch.Tensor) -> "HierarchicalKMeans":
        X_np = np.asarray(X.detach().cpu())
        if X_np.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        n_samples, n_features = X_np.shape
        self.n_features_in_ = n_features

        self.tree_ = {
            self.root_: {
                "is_leaf": False,
                "indices": np.arange(n_samples, dtype=int),
                "centroids": None,
                "children": [],
            }
        }

        queue = [self.root_]
        while queue:
            node_id = queue.pop(0)
            node = self.tree_[node_id]
            indices = node["indices"]
            size = indices.size

            if size < self.max_leaf_size:
                node["is_leaf"] = True
                node["children"] = []
                node["centroids"] = None
                continue

            k = min(self.n_clusters, size)
            if k < 2:
                node["is_leaf"] = True
                node["children"] = []
                node["centroids"] = None
                continue

            Xi = X_np[indices]
            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            labels = km.fit_predict(Xi)

            unique_labels, counts = np.unique(labels, return_counts=True)
            if unique_labels.size < 2 or np.any(counts == 0):
                node["is_leaf"] = True
                node["children"] = []
                node["centroids"] = None
                continue

            node["is_leaf"] = False
            node["centroids"] = km.cluster_centers_
            node["children"] = []

            for lab in range(k):
                child_indices = indices[labels == lab]
                child_id = f"{node_id}/{lab}"
                self.tree_[child_id] = {
                    "is_leaf": child_indices.size < self.max_leaf_size,
                    "indices": child_indices,
                    "centroids": None,
                    "children": [],
                }
                node["children"].append(child_id)
                if child_indices.size >= self.max_leaf_size:
                    queue.append(child_id)

            node["indices"] = None

        return self

    def _traverse_one(self, x: np.ndarray) -> str:
        node_id = self.root_
        while not self.tree_[node_id]["is_leaf"]:
            centroids = self.tree_[node_id]["centroids"]
            diffs = centroids - x.reshape(1, -1)
            dists = np.einsum("ij,ij->i", diffs, diffs)
            child_idx = int(np.argmin(dists))
            node_id = self.tree_[node_id]["children"][child_idx]
        return node_id

    def predict(self, X: np.ndarray | torch.Tensor) -> List[str]:
        X_np = np.asarray(X)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        if X_np.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features")
        return [self._traverse_one(X_np[i]) for i in range(X_np.shape[0])]

    def leaf_members(self) -> Dict[str, np.ndarray]:
        return {nid: node["indices"] for nid, node in self.tree_.items() if node["is_leaf"]}

    def centroids(self) -> Dict[str, np.ndarray]:
        return {nid: node["centroids"] for nid, node in self.tree_.items() if not node["is_leaf"] and node["centroids"] is not None}

    def leaf_members_json(self) -> List[Dict[str, Any]]:
        """Return the hierarchical leaf assignments as a JSON-serialisable list."""
        # tag_tree_recs structure -> [{"tag_ids": [int, ...], "children": [...]}]
        return self.convert_tree_dict_to_json(self.leaf_members())

    @classmethod
    def convert_tree_dict_to_json(cls, tree: Dict[str, Iterable[int]]) -> List[Dict[str, Any]]:
        """Convert ``leaf_members()`` output to a nested dict used downstream.

        ``tag_tree_recs`` structure -> ``[{"tag_ids": [int, ...], "children": [...]}]``.
        """
        nodes: Dict[Tuple[str, ...], Dict[str, Any]] = {}

        def get_node(path: Tuple[str, ...]) -> Dict[str, Any]:
            if path not in nodes:
                nodes[path] = {"tag_ids": [], "children": {}}
            return nodes[path]

        get_node(())

        for key, ids in tree.items():
            parts = key.split('/')
            if not parts or parts[0] != 'root':
                raise ValueError(f"All keys must start with 'root': got {key}")
            segments = tuple(parts[1:])

            for idx in range(len(segments)):
                parent_path = segments[:idx]
                child_key = segments[idx]
                parent = get_node(parent_path)
                child_path = segments[: idx + 1]
                child = get_node(child_path)
                parent["children"].setdefault(child_key, child)

            leaf = get_node(segments)
            leaf["tag_ids"] = cls._as_int_list(ids)

        def serialise(node: Dict[str, Any]) -> Dict[str, Any]:
            out = {"tag_ids": node["tag_ids"]}
            if node["children"]:
                ordered = []
                for key in cls._sorted_keys_numeric(node["children"].keys()):
                    ordered.append(serialise(node["children"][key]))
                out["children"] = ordered
            return out

        root = nodes[()]
        result = []
        for key in cls._sorted_keys_numeric(root["children"].keys()):
            result.append(serialise(root["children"][key]))
        return result

    @staticmethod
    def _as_int_list(xs: Iterable[Any]) -> List[int]:
        return [int(x) for x in xs]

    @staticmethod
    def _sorted_keys_numeric(keys: Iterable[str]) -> List[str]:
        return sorted(keys, key=lambda k: (0, int(k)) if k.isdigit() else (1, k))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X_data = np.vstack([
        rng.normal(loc=[0, 0], scale=0.6, size=(200, 2)),
        rng.normal(loc=[5, 5], scale=0.6, size=(200, 2)),
        rng.normal(loc=[0, 6], scale=0.6, size=(200, 2)),
    ])
    tensor_data = torch.from_numpy(X_data)

    model = HierarchicalKMeans(n_clusters=3, max_leaf_size=60, random_state=0)
    model.fit(tensor_data)
    members = model.leaf_members()
    print("Number of leaves:", len(members))
    tree_json = model.leaf_members_json()
    print("Sample tree node:", tree_json[0])
