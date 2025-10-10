Awesome project. You can absolutely boost a hierarchical k-means tree by (1) letting points “spill” across boundaries, (2) adding graph edges that capture real nearest-neighbor structure, and (3) regularly *rewiring* the graph as embeddings evolve. Below is a compact plan with methods, why they work, and runnable Python snippets.

# 1) Turn your tree into a better graph

### A. Overlap the partitions (“spill trees” / overlapping splits)

Classic KD/ball trees route each point to exactly one child; *spill* variants allow a point to belong to both children near the split. This reduces hard-boundary errors and improves nearest-neighbor recall, which directly helps assignment accuracy in your hierarchy. See spill trees and overlapping randomized trees for background. ([CMU School of Computer Science][1])

**How to use:** even if you keep your current hierarchical k-means, mark border points (e.g., within margin τ of a split hyperplane or near the Voronoi boundary between centroids) and duplicate them into both children. Later, prune duplicates by using the graph edges (below).

---

### B. Add *k*-NN edges (mutual or shared-NN) around your tree

Build a *k*-nearest-neighbor (kNN) graph over all leaf centroids or documents. Prefer **mutual kNN** (edge i↔j only if both list each other) or **Shared-Nearest-Neighbor (SNN)** weights (edge weight = size of intersection of NN lists). These reduce hub noise and capture manifold structure better than raw cosine alone. ([Wiley Online Library][2])

**Python (fast kNN graph):**

```python
import numpy as np
from pynndescent import NNDescent  # pip install pynndescent
# X: (n, d) embedding matrix, metric: "cosine" or "euclidean"
index = NNDescent(X, n_neighbors=32, metric="cosine", random_state=42)
knn_ind, knn_dist = index.neighbor_graph  # arrays (n, k)
# Mutual kNN mask
n = X.shape[0]
mutual = [set(neighs) for neighs in knn_ind]
edges = []
for i in range(n):
    for j in knn_ind[i]:
        if i in mutual[j]:
            edges.append((i, j))
# SNN weights
from collections import defaultdict
snn_w = defaultdict(int)
for i in range(n):
    Ni = set(knn_ind[i])
    for j in knn_ind[i]:
        Nj = set(knn_ind[j])
        w = len(Ni & Nj)
        if w > 0:
            snn_w[(i, j)] = w
```

PyNNDescent is a standard kNN-graph builder used in UMAP; it’s fast and flexible. ([PyNNDescent][3])

---

### C. Small-world long-range edges (HNSW)

Augment your local kNN graph with a few *long-range* “express” edges to improve navigability (like adding highways). **HNSW** creates a multi-layer small-world graph that supports fast search *and* incremental updates; you can either use it directly for assignment or mine its edges to enrich your graph. ([arXiv][4])

**Python (HNSW in memory):**

```python
import hnswlib  # pip install hnswlib
dim = X.shape[1]
p = hnswlib.Index(space='cosine', dim=dim)
p.init_index(max_elements=len(X), ef_construction=200, M=32)
p.add_items(X, np.arange(len(X)))
p.set_ef(128)
# Query neighbors (to add as graph edges)
labels, dists = p.knn_query(X, k=16)
```

`hnswlib` supports incremental insertions/deletions; FAISS also has an HNSW index if you prefer that stack. ([GitHub][5])

---

### D. Re-cluster the enriched graph with community detection

Once you have a better graph (local kNN + SNN weights + a few small-world edges), detect communities to refine/override the brittle tree splits. **Leiden** improves on **Louvain** by guaranteeing well-connected communities and typically higher modularity, and scales to millions of nodes. Use it on the document-graph or on a centroid-graph per level. ([Nature][6])

**Python (Leiden via igraph):**

```python
import igraph as ig
import leidenalg as la  # pip install python-igraph leidenalg

# Build weighted graph from SNN or similarity weights
g = ig.Graph(n=X.shape[0])
g.add_edges(list(snn_w.keys()))
g.es["weight"] = list(snn_w.values())

part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=1.0)
labels = np.array(part.membership)  # community id per node
```

Leiden’s guarantees and speed are well documented in both the original paper and library docs. ([Nature][6])

---

### E. Spectral / diffusion rewiring (optional but powerful)

If you can afford an eigenproblem, run **spectral clustering** on your similarity graph (normalized Laplacian), then k-means in the spectral space. It’s excellent for non-convex structure; even using just the top few eigenvectors to *rewire edges by cosine in eigen-space* helps. ([ai.stanford.edu][7])

**Python (sketch):**

```python
from sklearn.cluster import SpectralClustering
spec = SpectralClustering(n_clusters=K, affinity='precomputed', assign_labels='kmeans')
S = build_affinity_matrix_from_graph(snn_w, n_nodes=X.shape[0])  # sparse or dense
y = spec.fit_predict(S)
```

# 2) Keep the topology fresh as embeddings change

When you update your embedding model (or stream new docs), *update the graph incrementally*:

* **Incremental kNN maintenance:** PyNNDescent and HNSW both support warm starts/insertions. For embeddings that drift, re-insert changed points and *age out* stale edges (time decay or low similarity) before re-running Leiden on affected subgraphs. ([GitHub][8])
* **Spill boundary refresh:** recompute border points per split and keep duplicates only if their cross-edge similarity remains above a threshold (reduces bloat).
* **Coreset/medoid refresh:** replace level-centroids with **k-medoids** on each community (robust to outliers) and push changes down the hierarchy before rewiring edges locally.

# 3) A practical pipeline (drop-in beside your current tree)

1. **Embed** docs → matrix `X`.
2. **Tree pass:** your hierarchical k-means builds initial structure; mark split-border spill points.
3. **Graph build:** kNN via PyNNDescent → prune to **mutual**; weight with **SNN**. Add a few HNSW long-range edges (degree budget, e.g., +4 per node). ([PyNNDescent][3])
4. **Community refine:** run Leiden per level or on the whole graph.
5. **Assign & route:** use community ids + graph distances for routing/lookup; keep the original tree for speed but fall back to graph hops for tie-breaks.
6. **Online updates:** on new/changed embeddings, update HNSW + local mutual kNN; periodically re-run Leiden on touched components.

# 4) End-to-end, minimal working example

```python
# pip install sentence-transformers pynndescent hnswlib python-igraph leidenalg networkx
import numpy as np, networkx as nx
from sentence_transformers import SentenceTransformer
from pynndescent import NNDescent
import igraph as ig, leidenalg as la
import hnswlib

docs = ["... your texts ..."]
model = SentenceTransformer("all-mpnet-base-v2")
X = model.encode(docs, normalize_embeddings=True)

# (A) build kNN + mutual + SNN
index = NNDescent(X, n_neighbors=32, metric="cosine", random_state=0)
knn_ind, _ = index.neighbor_graph
n = len(docs); neigh_sets = [set(neigh) for neigh in knn_ind]
edges, snn_w = [], {}
for i in range(n):
    for j in knn_ind[i]:
        if i<j and i in neigh_sets[j]:  # mutual
            w = len(neigh_sets[i] & neigh_sets[j])  # SNN weight
            if w>0:
                edges.append((i,j)); snn_w[(i,j)] = w

# (B) add small-world edges via HNSW (few per node)
dim = X.shape[1]
hidx = hnswlib.Index(space='cosine', dim=dim)
hidx.init_index(max_elements=n, ef_construction=200, M=32)
hidx.add_items(X, np.arange(n)); hidx.set_ef(64)
lbls, dists = hidx.knn_query(X, k=6)  # small k for extra long-ish links
for i in range(n):
    for j in lbls[i]:
        if i<j:
            edges.append((i,j))
            snn_w[(i,j)] = snn_w.get((i,j), 1)  # light weight for HNSW edges

# (C) Leiden community detection
g = ig.Graph(n=n)
g.add_edges(edges)
g.es["weight"] = [snn_w[e.tuple] if e.tuple in snn_w else snn_w[(e.tuple[1], e.tuple[0])] for e in g.es]
part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=1.0)
communities = np.array(part.membership)

# communities[i] is your refined cluster id; keep your existing tree, but
# let these communities add cross-links & correct misassignments.
```

# 5) When you need global structure or non-convex shapes

* **Spectral clustering** (normalize Laplacian → eigenvectors → k-means) is robust for non-convex clusters; use it per level when you detect low modularity communities. ([ai.stanford.edu][7])
* A **diffusion map** or Laplacian eigenmap can be used just to *smooth the graph*, then rebuild mutual kNN on the smoothed coordinates. ([arXiv][9])

# 6) Libraries you’ll want (all have good docs)

* **PyNNDescent** (fast kNN graph, ANN; used by UMAP). ([PyNNDescent][3])
* **hnswlib** (small-world ANN index with dynamic insert/delete). FAISS also offers HNSW. ([GitHub][5])
* **igraph + leidenalg** (Leiden community detection). ([leidenalg.readthedocs.io][10])
* **scikit-learn** (spectral clustering, k-medoids, etc.).
* Optional: **UMAP** (dimension reduction + kNN graph tooling) and **NetworkX** (quick graph manipulation). ([GitHub][11])

# 7) Why this improves accuracy (intuition)

* Trees impose **hard partitions**; manifolds of text embeddings are messy. Overlaps (spill) + graph edges **reconnect** neighbors cut by splits. ([CMU School of Computer Science][1])
* Mutual/SNN edges prefer **reciprocal** relationships, suppressing hubs and noise. ([www-users.cse.umn.edu][12])
* Small-world links make the graph **navigable** and give you shortcuts across the hierarchy. ([arXiv][4])
* Leiden ensures communities are **well-connected** (a known Louvain weakness) and typically gives higher-quality partitions. ([Nature][6])

---

If you share a small sample of your current code (tree build + assignment), I can drop these pieces in *surgically*—keeping your interfaces—and show how to (a) generate spill duplicates, (b) build/maintain the kNN+HNSW hybrid graph, and (c) re-label leaves with Leiden while preserving the original hierarchy for speed.

[1]: https://www.cs.cmu.edu/~tingliu/my_papers/nips04.pdf?utm_source=chatgpt.com "An Investigation of Practical Approximate Nearest Neighbor ..."
[2]: https://onlinelibrary.wiley.com/doi/10.1002/sam.10149?utm_source=chatgpt.com "Clustering algorithm based on mutual K‐nearest neighbor ..."
[3]: https://pynndescent.readthedocs.io/en/latest/how_to_use_pynndescent.html?utm_source=chatgpt.com "How to use PyNNDescent"
[4]: https://arxiv.org/pdf/1603.09320?utm_source=chatgpt.com "Efficient and robust approximate nearest neighbor search ..."
[5]: https://github.com/nmslib/hnswlib?utm_source=chatgpt.com "nmslib/hnswlib: Header-only C++/python library for fast ..."
[6]: https://www.nature.com/articles/s41598-019-41695-z?utm_source=chatgpt.com "From Louvain to Leiden: guaranteeing well-connected ..."
[7]: https://ai.stanford.edu/~ang/papers/nips01-spectral.pdf?utm_source=chatgpt.com "On Spectral Clustering: Analysis and an algorithm"
[8]: https://github.com/lmcinnes/pynndescent?utm_source=chatgpt.com "lmcinnes/pynndescent: A Python nearest neighbor descent ..."
[9]: https://arxiv.org/pdf/0711.0189?utm_source=chatgpt.com "A Tutorial on Spectral Clustering"
[10]: https://leidenalg.readthedocs.io/en/stable/intro.html?utm_source=chatgpt.com "Introduction - leidenalg documentation"
[11]: https://github.com/lmcinnes/umap?utm_source=chatgpt.com "lmcinnes/umap: Uniform Manifold Approximation and ..."
[12]: https://www-users.cse.umn.edu/~kumar/papers/snn14.pdf?utm_source=chatgpt.com "A Shared Nearest Neighbor Approach* Abstract 1. Introduction"
