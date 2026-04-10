# SPORE

**Skeleton Propagation Over Recalibrating Expansions** — a graph-based clustering algorithm for arbitrary-shape, arbitrary-scale clusters.

![SPORE](https://raw.githubusercontent.com/RandyWAidoo/SPORE/main/docs/logo.png)


## How it works

SPORE builds clusters in three stages:

1. **k-NN graph construction**: a global nearest-neighbor graph is built (exact or approximate), with neighbor counts scaling as ~*O*(log *N*) by default.
2. **Variance-aware BFS expansion**: clusters are seeded from the densest points outward. Nearby points are accepted only if their distance is statistically consistent with the cluster's particular distance distribution, the mean and variance of which are updated as neighbors are accepted. This allows for density- and shape-adaptive cluster identification.
3. **Reassignment**: clusters below `min_cluster_size` are merged into nearby larger ones using a composite score weighing proximity, relative size, density, and angular isotropy, or labeled as noise.

## Installation

```bash
pip install spore-clustering
```

## Quick start

```python
from spore_clustering import SPORE

labels = SPORE().fit_predict(X)
```

## Key parameters

| Parameter | Description |
|---|---|
| `expansion` | Z-score threshold controlling how aggressively clusters grow |
| `neighborhood_percentile` | Bounded alternative to `expansion`; typical values: 25, 50, 75, 93.75 |
| `retention_rate` | Fraction of neighbors that must pass variance filter to continue expansion |
| `min_cluster_size` | Minimum cluster size (int) or exponent for *N*-relative scaling (float) |

See the [full API reference](https://github.com/RandyWAidoo/SPORE/blob/main/docs/01-SPORE.md) for all parameters.

## Reusing a precomputed neighbor index

```python
dindex = SPORE.DataIndex(
    connectivity=k,
    neighbors=neighbors,
    dists=distances,
    dataset_scale=scale,
)

labels = SPORE(dindex=dindex, retention_rate=0.25).fit_predict(X)
```

## Complexity

With approximate k-NN and default neighbor scaling (*k* ~ log *N*):

| Phase | Complexity |
|---|---|
| k-NN construction | *O*(*Nd* log *N*) |
| BFS expansion | *O*(*N* log *N*) |
| Reassignment | *O*(*Rd* log *N*), *R* ≤ *N* |

## scikit-learn compatibility

SPORE follows standard scikit-learn estimator conventions: `fit`, `fit_predict`, `get_params`, `set_params`.