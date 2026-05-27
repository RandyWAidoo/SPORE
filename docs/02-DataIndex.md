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
    )
```

A lightweight container for holding neighbor graph state and related metadata so SPORE can avoid rebuilding k-NN structures across runs.

## Parameters / Attributes

### `connectivity : int, optional`

Neighbor count stored per point.

This usually matches the second dimension of `neighbors` and `dists`.

---

### `n_copies : int, optional`

Number of copied or expanded points represented by the stored index, when applicable.

This is mainly used internally.

---

### `neighbors : ndarray, optional`

Neighbor indices for each point.

Typically has shape:

```text
(N, connectivity)
```

---

### `dists : ndarray, optional`

Neighbor distances for each point.

Typically has shape:

```text
(N, connectivity)
```

---

### `dataset_scale : float, optional`

Characteristic scale estimate of the dataset.

SPORE may use this when reusing stored neighbor graph data.

---

## Methods

### `clear()`

Clears all stored fields by setting them to `None`.

This resets:

```text
connectivity
n_copies
neighbors
dists
dataset_scale
```
