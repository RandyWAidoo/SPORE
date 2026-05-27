# SPORE

**SPORE (Skeleton Propagation Over Recalibrating Expansions)** is a graph-based clustering algorithm for nonlinear clusters under heterogeneous density and weak boundary contrast.

![SPORE](https://raw.githubusercontent.com/RandyWAidoo/SPORE/main/docs/logo.png)

## The Algorithm

SPORE builds a reusable k-nearest-neighbor graph, then runs two main phases:

1. **Expansion**: clusters are seeded from dense regions and expanded with breadth-first search over the k-NN graph. Candidate neighbors are accepted only when their distances are consistent with the growing cluster's evolving distance statistics. This lets each cluster adapt to its own local density scale while still following nonconvex shapes.

2. **Small-Cluster Reassignment (SCR)**: clusters below `min_cluster_size` are treated as fragments. Fragment points are reassigned to established clusters using local k-NN majority voting, with candidate neighbors filtered by cluster size and density compatibility. Any fragments still unresolved after SCR can be labeled as noise or left unchanged, depending on `post_reassignment_policy`.

## Installation

```bash
pip install spore-clustering
```

## Quick Start

```python
from spore_clustering import SPORE

labels = SPORE().fit_predict(X)
```

## Key Parameters

| Parameter                  | Description                                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `z`                        | Z-score threshold controlling how aggressively clusters expand                                                |
| `z_percentile`             | Percentile-based alternative to `z`; ignored if `z` is provided                                               |
| `retention_rate`           | Fraction of neighbors that must pass the expansion filter for traversal to continue                           |
| `min_cluster_size`         | Minimum established-cluster size; ints are absolute counts, floats are interpreted as `N ** min_cluster_size` |
| `max_z`                    | Maximum z-score allowed for candidate receiving-cluster neighbors during SCR                                  |
| `max_z_percentile`         | Percentile-based alternative to `max_z`; ignored if `max_z` is provided                                       |
| `max_scr_rounds`           | Maximum number of SCR propagation rounds                                                                      |
| `post_reassignment_policy` | Whether remaining unresolved small clusters become noise or are left unchanged                                |

See the [full API reference](https://github.com/RandyWAidoo/SPORE/blob/main/docs/01-SPORE.md) for all parameters.

## Reusing a Precomputed Neighbor Index

```python
dindex = SPORE.DataIndex(
    connectivity=k,
    neighbors=neighbors,
    dists=distances,
    dataset_scale=scale,
)

labels = SPORE(dindex=dindex, retention_rate=0.25).fit_predict(X)
```

## Time Complexity

With an efficient k-NN backend and default neighbor scaling, where `k ~ O(log N)`:

| Phase                     | Complexity                                                     |
| ------------------------- | -------------------------------------------------------------- |
| k-NN graph construction   | *O*(*N d* log *N*)                                             |
| Expansion                 | *O*(*N* log *N*)                                               |
| SCR                       | *O*(*N* log *N*)                                               |

In the worst case, with a bounded number of SCR rounds, the clustering phases after neighbor construction scale as *O*(*N* log *N*). Including approximate k-NN construction, the practical overall complexity is *O*(*N d* log *N*).

## Scikit-learn Compatibility

SPORE follows standard scikit-learn estimator conventions: `fit`, `fit_predict`, `get_params`, and `set_params`.
