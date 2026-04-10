# SPORE

**SPORE (Skeleton Propagation Over Recalibrating Expansions)** is a graph-based clustering algorithm that discovers clusters by expanding subgraphs of a global k-nearest-neighbor (k-NN) graph using **breadth-first search (BFS)**. Cluster growth is constrained by an **evolving variance rule** that adapts dynamically as clusters expand, allowing SPORE to capture core structure structure of arbitrary shape and scale. Once cluster skeletons are identified, they are **propagated** outward over surrounding fragments to create a final distribution of large, sharply-defined clusters.

---

## How it works

SPORE proceeds in three conceptual stages:

### 1. Global k-NN graph construction

A nearest-neighbor graph is constructed using either exact or approximate search. Neighbor indices and distances may optionally be reused through a `DataIndex` object to avoid rebuilding the structure.

Neighbor counts used internally for connectivity and expansion scale approximately as **$O(\log N)$** with dataset size by default, balancing statistical robustness and computational efficiency.

---

### 2. Variance-aware BFS expansion from density-ordered seeds

Clusters are seeded (typically from densest points first) and expanded via BFS traversal of the k-NN graph.

Edges are accepted only if their distance is statistically consistent with the evolving distribution of accepted intra-cluster k-NN distances:

* Mean and variance are **not assumed beforehand**.
* They are updated **incrementally** as edges are accepted.
* As expansion progresses, these statistics stabilize and reflect the intrinsic geometry of the cluster.

This adaptive constraint helps prevent expansion across low-density gaps while preserving irregular cluster shapes.

---

### 3. Small-cluster handling and reassignment

After initial assignment:

* Clusters smaller than `min_cluster_size` are either reassigned into nearby valid clusters or treated as noise.
* When reassignment is enabled, candidate target clusters are ranked using a composite score incorporating:

  * proximity,
  * relative size,
  * local density,
  * angular isotropy (how well the candidate cluster geometrically surrounds the smaller cluster’s points).

In practice:

* Nonconvex structure is primarily recovered during expansion.
* Weakly separated clusters can emerge as stable cores first and are completed during reassignment.

---

## Parameters

### Expansion threshold estimation

#### **neighborhood_percentile : float, optional**

A bounded reparameterization of `expansion` expressed as a percentile ($q \in [0, 100]$).

Internally, SPORE converts this percentile into a z-score expansion cutoff:

$$
\text{expansion} \approx
\frac{Q_q({\delta_{ik} : \delta_{ik} \ge \mu_{\delta_{ik}}})

- \mu_{\delta_{ik}}}
  {\sigma_{\delta_{ik}}}
  $$

where:

* $(\delta_{ik})$ are k-NN distances,
* $(\mu_{\delta_{ik}})$ and $(\sigma_{\delta_{ik}})$ are their mean and standard deviation,
* $(Q_q(\cdot))$ denotes the q-th percentile restricted to distances at or above the mean.

This produces a stable expansion threshold estimate while reducing sensitivity to outliers.

Typical values:

```
{25, 50, 75, 93.75}
```

If `expansion` is provided explicitly, this parameter is ignored.

---

#### **expansion : float, optional**

Upper z-score threshold used during BFS expansion.

An edge is accepted only if its distance lies within this threshold relative to the evolving intra-cluster distance distribution.

* Smaller values → stricter expansion, reduced bridging.
* Larger values → more permissive cluster growth across density variation.

---

### Neighbor counts and connectivity

Neighbor counts used internally typically scale logarithmically with dataset size.

#### **expansion_neighbors : int, optional**

Number of nearest neighbors considered during BFS expansion.

#### **min_retained : int, optional**

Minimum number of candidate neighbors that must remain after variance filtering for expansion to continue from a node. Prevents propagation through narrow bridges or sparse corridors.

#### **retention_rate : float, optional**

Fractional version of `min_retained` relative to `expansion_neighbors`.
Overrides `min_retained` when provided and is often easier to tune.

#### **min_connectivity : int, optional**

Minimum number of neighbors retrieved per point when building the global k-NN graph.

#### **max_connectivity : int, optional**

Maximum number of neighbors retrieved per point when building the global k-NN graph. Higher values improve robustness to duplicates and disconnected graph artifacts at modest computational cost.

---

### Density and seeding

#### **seeding_order : {"none", "random", "density"}, optional**

Determines the order in which cluster seeds are initialized:

* `"density"`: densest points first (ascending k-NN distance).
* `"random"`: randomized seeding.
* `"none"`: input order.

Density is estimated using the neighbor count specified by `density_neighbors`.

#### **density_neighbors : int, optional**

Neighbor count used to compute k-NN distance statistics serving as a proxy for local density.

---

### Reassignment and merging

#### **small_cluster_policy : {"reassign", "noise"}, optional**

Handling of clusters smaller than `min_cluster_size`:

* `"reassign"`: attempt reassignment into nearby valid clusters using ranking criteria.
* `"noise"`: merge all small clusters into a single noise label.

#### **post_reassignment_policy : {"noise", "none"}, optional**

Applied after reassignment:

* `"noise"`: remaining small clusters become noise.
* `"none"`: leave remaining small clusters unchanged.

#### **reassignment_neighbors : int, optional**

Neighbors consulted when reassigning points.

#### **far_percentile : float, optional**

Percentile of k-NN distances defining distances considered too far for reassignment.

#### **min_cluster_size : float or int, optional**

Minimum allowable cluster size:

* Integer → absolute minimum point count.
* Float → interpreted as an exponent:

$$
\text{min cluster size} = N^{(\text{value})}.
$$

This scaling keeps minimum cluster size proportional to dataset size.

---

### Precomputed neighbor indices

#### **dindex : SPORE.DataIndex, optional**

Container for reusing precomputed neighbor graph data, including neighbors, distances, connectivity, and dataset scale.

#### **manage_dindex : bool, optional**

Whether SPORE automatically manages lifecycle updates to the stored neighbor data.

---

### Heuristics and neighbor search backend

* **use_heuristics**: enables additional stability and performance heuristics.
* **exact_knn**: exact vs approximate nearest-neighbor search.
* **nn_kwargs**: additional backend parameters.
* **shuffle_for_hnsw**: improves approximate neighbor stability by shuffling input order.
* **shuffle_seed**: random seed for shuffling and randomized seeding.
* **n_jobs**: number of threads used during neighbor index construction (`-1` uses all cores).

---

### Output and diagnostics

* **show_progress**: display progress information during clustering.

---

## Attributes

**labels_ : ndarray of shape $(N,)$**
Cluster label assigned to each sample. Noise (if used) is typically encoded as `-1`.

**n_clusters_ : int**
Number of clusters found (excluding noise).

---

## Complexity

Let:

* $N$ be the number of samples,
* $d$ be the feature dimension,
* $k$ be the effective neighbor count used during traversal (often set to scale like $k \approx \log N$ by default).

### k-NN index construction and querying

The cost depends heavily on the backend and whether search is approximate or exact:

* **Approximate k-NN (default in practice):** commonly behaves like $O(N d \log N)$ for building/querying the neighbor structure (implementation-dependent, but $d$ enters because distance computations scale with dimension).
* **Exact k-NN:** can be significantly more expensive in the worst case (up to $O(d N^2)$ for brute-force style computation), though tree-based methods may help in low $d$ and degrade as $d$ increases.

### Expansion phase (variance-aware BFS)

Expansion visits each point and considers $k$ neighbors per point:

* Work: $O(N \cdot k)$
* Under the default scaling $k \approx \log N$:

$$
O(N \log N).
$$

### Reassignment phase (small-cluster handling)

Reassignment considers candidate target clusters using criteria including **angular isotropy**, which requires $d$-dimensional computations. Isotropy scoring costs $O(dk)$ per reassigned point, so for $R$ reassigned points time complexity is:

$$
O(R \cdot d \cdot k).
$$

Under $k \approx \log N$, this becomes:

$$
O(R d \log N).
$$

Since reassignment only applies to points in undersized clusters, $R$ always satisfies $0 \le R \le N$. $R$ depends on how much fragmentation occurs during expansion and on the chosen minimum cluster size so permissive expansion or small minimum cluster thresholds can reduce it to a value substantially below $N$. The worst-case bound $O(N d \log N)$ therefore reflects omission of a fractional factor $f = R/N \in [0,1]$ rather than the introduction of an additional growth multiplier.

### Overall

With approximate k-NN and $k \approx \log N$, a practical summary is:

* Expansion: $O(N \log N)$
* Reassignment: $O(R d \log N)$ (at most $O(N d \log N)$)
* Approximate k-NN backend: often $O(N d \log N)$

yielding a worst-case complexity of:

$$
O(N d \log N)
$$

with variation largely driven by how many points ultimately require reassignment.

---

## Scikit-learn API compatibility

SPORE follows standard **scikit-learn estimator conventions**:

* `fit(X, y=None)`
* `fit_predict(X, y=None)`
* `get_params(deep=True)`
* `set_params(**params)`

Dataset size and dimensionality are inferred directly from `X`.

---

### Special behavior: `fit_predict(..., dindex_only=True)`

`fit_predict` supports one SPORE-specific fit parameter:

**dindex_only : bool**

If `True`, SPORE builds and populates the internal `DataIndex` (neighbor graph, distances, and related metadata) but skips clustering.

This is useful for:

* reusing neighbor structures across runs,
* inspecting or serializing the index,
* deferring clustering while avoiding manual k-NN construction.

No labels are produced in this mode, but the populated `dindex` can be reused in subsequent runs depending on `manage_dindex`.

---

## Examples

### Basic usage

```python
import numpy as np

model = SPORE()
labels = model.fit_predict(X)
```

---

### Reusing a precomputed neighbor index

```python
# Assuming `neighbors` and `distances` are dense 2D arrays 
dindex = SPORE.DataIndex(
    connectivity=len(neighbors[0]), 
    neighbors=neighbors,
    dists=distances,
    dataset_scale=scale,
    use_heuristics=True,
)

model = SPORE(dindex=dindex, retention_rate=0.25)
labels = model.fit_predict(X)
```
