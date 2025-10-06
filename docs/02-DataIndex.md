# SPORE.DataIndex

```python
class SPORE.DataIndex:
    def __init__(
        self,
        connectivity=None,
        n_copies=None,
        neighbors=None,
        dists=None,
        dataset_scale=None,
        use_heuristics=None,
    )
```

A lightweight container for holding neighbor graph state and related metadata so SPORE can avoid rebuilding k-NN structures.

### Parameters / Attributes

**connectivity : int, optional**
Neighbor count stored per point (width of `neighbors` / `dists`).

**n_copies : int, optional**
Internal multiplier used by certain backends/heuristics (implementation-specific).

**neighbors : ndarray, optional**
Neighbor indices per point, typically shape `(N, connectivity)`.

**dists : ndarray, optional**
Neighbor distances per point, typically shape `(N, connectivity)`.

**dataset_scale : float, optional**
A characteristic scale estimate of the dataset used for normalization or heuristics.

**use_heuristics : bool, optional**
Whether the stored index corresponds to a heuristic-enabled configuration.

### Methods

**clear()**
Clears all stored fields (sets them to `None`).

