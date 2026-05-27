# SPORE

**SPORE (Skeleton Propagation Over Recalibrating Expansions)** is a graph-based clustering algorithm for nonlinear clusters under heterogeneous density and weak boundary contrast.

SPORE builds clusters over a reusable k-nearest-neighbor graph. It first isolates stable cluster skeletons with adaptive breadth-first expansion, then uses Small-Cluster Reassignment (SCR) to propagate those skeletons into nearby fragments.

---

## The Algorithm

SPORE builds a reusable k-nearest-neighbor graph, then runs two main phases:

### 1. Expansion

Clusters are seeded from dense regions and expanded with breadth-first search over the k-NN graph.

Candidate neighbors are accepted only when their distances are consistent with the growing cluster's evolving distance statistics. This lets each cluster adapt to its own local density scale while still following nonconvex shapes.

In practice:

* Lower `z` values produce tighter, more conservative skeletons.
* Higher `z` values allow broader expansion across density variation.
* Higher `retention_rate` values make traversal less likely to pass through thin bridges or sparse corridors.

### 2. Small-Cluster Reassignment (SCR)

Clusters below `min_cluster_size` are treated as fragments.

Fragment points are reassigned to established clusters using local k-NN majority voting, with candidate neighbors filtered by cluster size and density compatibility. Any fragments still unresolved after SCR can be labeled as noise or left unchanged, depending on `post_reassignment_policy`.

SCR lets stable skeleton clusters propagate outward into nearby boundary or shell regions without forcing clusters to be convex or centroid-shaped.

---

## Parameters

### Expansion Thresholds

#### `z : float, optional`

Z-score threshold controlling how aggressively clusters expand.

During Expansion, SPORE compares candidate neighbor distances against the growing cluster's evolving distance statistics. A candidate neighbor is accepted only if its distance is not abnormally large relative to the cluster's current local scale.

Lower values make Expansion stricter. Higher values make Expansion more permissive.

---

#### `z_percentile : float, optional`

Percentile-based alternative to `z`.

SPORE computes kth-nearest-neighbor distances using `k = expansion_neighbors`, then converts the selected percentile into a z-score estimate for `z`.

This gives a bounded way to tune the Expansion threshold across datasets.

If `z` is provided, `z_percentile` is ignored.

---

#### `max_z : float, optional`

Maximum z-score allowed for candidate receiving-cluster neighbors during SCR.

During reassignment, candidate neighbors are filtered by density compatibility. `max_z` controls how far a fragment point may be from the local density regime of a candidate receiving cluster.

Lower values make SCR more conservative. Higher values allow labels to propagate more freely.

---

#### `max_z_percentile : float, optional`

Percentile-based alternative to `max_z`.

This is the SCR-side counterpart to `z_percentile`. It provides a bounded way to tune the density-compatibility gate used during reassignment.

If `max_z` is provided, `max_z_percentile` is ignored.

---

### Neighbor Counts and Connectivity

#### `expansion_neighbors : int, optional`

Number of nearest neighbors considered during the Expansion phase.

This controls the local graph neighborhood used when growing cluster skeletons.

---

#### `min_retained : int, optional`

Minimum number of neighbors that must remain after filtering during Expansion for traversal to continue from a point.

This helps prevent clusters from propagating through thin bridges, sparse corridors, or weakly supported connections.

Ignored when `retention_rate` is provided.

---

#### `retention_rate : float, optional`

Fraction of neighbors that must pass the expansion filter for traversal to continue.

This is the normalized version of `min_retained`, scaled relative to `expansion_neighbors`.

If provided, `retention_rate` takes precedence over `min_retained`.

For example, with `expansion_neighbors=40` and `retention_rate=0.25`, traversal requires roughly 10 retained neighbors to continue.

---

#### `reassignment_neighbors : int, optional`

Number of nearest neighbors consulted during SCR.

This controls the local neighborhood used for majority-vote reassignment.

---

#### `density_neighbors : int, optional`

Number of neighbors used to compute kth-nearest-neighbor distances for density ordering.

This mainly affects `seeding_order="density"`.

---

#### `min_connectivity : int, optional`

Minimum number of neighbors fetched per point when building the global k-NN graph.

This helps ensure enough cached connectivity for the internal stages that reuse neighbor data.

---

#### `max_connectivity : int, optional`

Maximum number of neighbors fetched per point when building the global k-NN graph.

Higher values can improve robustness when using approximate neighbors, duplicates, or difficult local graph structure. They may also increase memory use and neighbor-search cost.

---

### Density and Seeding

#### `seeding_order : {"none", "random", "density"}, optional`

Strategy used to choose the order in which cluster seeds are initialized.

Options:

* `"density"`: seed from densest to sparsest points, based on ascending kth-nearest-neighbor distance. Ties are broken lexicographically by coordinates.
* `"random"`: seed in random order.
* `"none"`: seed in input order.

Density ordering is usually the most stable option because dense regions are allowed to claim nearby structure before sparse regions expand.

---

### SCR and Small-Cluster Handling

#### `min_cluster_size : float or int, optional`

Minimum established-cluster size.

Clusters smaller than this threshold after Expansion are treated as fragments and are eligible for SCR. Clusters at or above this threshold can act as receiving clusters.

Integers are interpreted as absolute point counts.

Floats are interpreted as:

```text
N ** min_cluster_size
```

For example:

```text
0.5 -> sqrt(N)
1.0 -> N
```

---

#### `max_scr_rounds : int, optional`

Maximum number of SCR propagation rounds.

Multiple rounds allow reassignment to spread outward from established skeleton clusters. A fragment point that cannot be reassigned in one round may become reachable after nearby fragments are reassigned.

Setting this to `0` disables SCR propagation after Expansion.

---

#### `post_reassignment_policy : {"noise", "none"}, optional`

Policy applied after SCR finishes.

Options:

* `"noise"`: remaining unresolved small clusters become noise.
* `"none"`: remaining unresolved small clusters are left unchanged.

---

### Precomputed Neighbor Index

#### `dindex : SPORE.DataIndex, optional`

Object storing precomputed nearest-neighbor graph data, distances, connectivity, and related metadata.

Providing a `dindex` can reduce overhead when running SPORE multiple times on the same data, such as during hyperparameter search.

---

#### `manage_dindex : bool, optional`

Whether SPORE should automatically manage the lifecycle of `dindex`.

When enabled, SPORE may update, replace, or clear stored neighbor data as needed.

---

### Neighbor-Search Backend

#### `exact_knn : bool, optional`

Whether to use exact nearest-neighbor search.

If `False`, SPORE may use approximate nearest-neighbor search. The algorithm can tolerate approximate neighbors because its Expansion rule is statistical and local rather than dependent on exact all-pairs distances.

---

#### `nn_kwargs : dict[str, Any], optional`

Additional keyword arguments passed to the nearest-neighbor backend.

---

#### `shuffle_for_hnsw : bool, optional`

Whether to shuffle data indices before building the HNSW index.

Shuffling can improve the stability and quality of approximate k-NN results.

---

#### `shuffle_seed : int, optional`

Random seed used for HNSW shuffling and for random seeding order when `seeding_order="random"`.

---

#### `n_jobs : int, optional`

Number of threads used during neighbor index construction.

A value of `-1` uses all available cores.

---

### Output and Diagnostics

#### `show_progress : bool, optional`

Whether to display progress information during clustering.

---

## Attributes

### `labels_ : ndarray of shape (N,)`

Cluster label assigned to each sample.

Noise, when used, is typically encoded as `-1`.

---

### `n_clusters_ : int`

Number of clusters found, excluding noise.

---

## Time Complexity

Let:

* `N` be the number of samples.
* `d` be the feature dimension.
* `k` be the effective neighbor count used during traversal.

With an efficient k-NN backend and default neighbor scaling, where `k ~ O(log N)`:

| Phase                   | Complexity         |
| ----------------------- | ------------------ |
| k-NN graph construction | *O*(*N d* log *N*) |
| Expansion               | *O*(*N* log *N*)   |
| SCR                     | *O*(*N* log *N*)   |

In the worst case, with a bounded number of SCR rounds, the clustering phases after neighbor construction scale as *O*(*N* log *N*). Including approximate k-NN construction, the practical overall complexity is *O*(*N d* log *N*).

Exact nearest-neighbor search can be more expensive, especially in high dimensions or with brute-force backends.

---

## Scikit-learn Compatibility

SPORE follows standard scikit-learn estimator conventions:

```python
fit(X, y=None)
fit_predict(X, y=None)
get_params(deep=True)
set_params(**params)
```

Dataset size and dimensionality are inferred directly from `X`.

---

### Special Behavior: `fit_predict(..., dindex_only=True)`

`fit_predict` supports one SPORE-specific fit parameter:

#### `dindex_only : bool`

If `True`, SPORE builds and populates the internal `DataIndex`, including neighbor graph data, distances, and related metadata, but skips clustering.

This is useful for:

* reusing neighbor structures across runs;
* inspecting or serializing the index;
* deferring clustering while avoiding manual k-NN construction.

No labels are produced in this mode. The populated `dindex` can be reused in later runs depending on `manage_dindex`.

---

## Examples

### Basic Usage

```python
from spore_clustering import SPORE

model = SPORE()
labels = model.fit_predict(X)
```

---

### Reusing a Precomputed Neighbor Index

```python
dindex = SPORE.DataIndex(
    connectivity=len(neighbors[0]),
    neighbors=neighbors,
    dists=distances,
    dataset_scale=scale,
)

model = SPORE(
    dindex=dindex,
    retention_rate=0.25,
)

labels = model.fit_predict(X)
```

---

### Using Bounded Threshold Parameters

```python
model = SPORE(
    z_percentile=75,
    max_z_percentile=100,
    retention_rate=0.25,
    min_cluster_size=0.5,
)

labels = model.fit_predict(X)
```

---

### Disabling SCR Propagation

```python
model = SPORE(
    max_scr_rounds=0,
    post_reassignment_policy="noise",
)

labels = model.fit_predict(X)
```
