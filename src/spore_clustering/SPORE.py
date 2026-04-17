from .types import *
from typing import Any, Union, Literal, Optional
from .LinkedList import LinkedList
from collections import deque
import numpy as np
from sklearn.neighbors import NearestNeighbors
import hnswlib
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array


_DEFAULTS_AND_CONSTS: dict[str, Any] = dict(
    exact_knn = False,
    nn_kwargs = dict(
        hnsw = dict(
            space = 'l2',
            ef_construction = 200,
            M = 16,
            random_seed = 42
        ),
        nearest_neighbors = dict(
            metric = 'minkowski',
            p = 2.0,
        ),
    ),
    shuffle_for_hnsw = True,
    shuffle_seed = 42,
    n_jobs = -1,

    use_heuristics = True,
    show_progress = False,
    min_connectivity = 1,
    seeding_order = 'density',
    neighborhood_percentile = 50,
    min_retained = 1,
    far_percentile = 95,
    min_cluster_size = 0.45,
    expansion = 2, 
    small_cluster_policy = 'reassign',
    post_reassignment_policy = 'noise',
)
NOISE_LABEL: int = -1


def _bounds_check(x: Union[int, float], lbound: float, ubound: float, err_msg: str):
    if (x < lbound or x > ubound):
        raise ValueError(err_msg)

class _ClusterCount_Ref:
    def __init__(self, data: int):
        self.data: int = data

def _bounded_sample_count(N: int, n_samples: int)->int:
    sample_rbound: int = (int(np.sqrt(N)) if N > 1 else 1)
    return max(1, min(sample_rbound, n_samples, N - 1))

def group_by_cluster_idx(classifications: NDArray[clust_idx_t])->list[list[int]]:
    """
    Group sample indices by cluster label.

    Assumes cluster labels are integers between 1 and the number of points,
    optionally including a noise label (-1). Labels are used directly
    for indexing and are not remapped.

    Parameters
    ----------
    classifications : NDArray[clust_idx_t]
        1D array of cluster labels, one per sample.

    Returns
    -------
    list[list[int]]
        Lists of sample indices per cluster. Empty clusters are omitted.

    Notes
    -----
    Not efficient for unbounded or sparse cluster labels.
    """
    
    clusters: list[list[int]] = [
        [] for _ in range(
            np.max(classifications) + 1 + (-1 in classifications) 
            if len(classifications) > 0 
            else 0
        )
    ]

    for idx, cluster_idx in enumerate(classifications):
        clusters[cluster_idx].append(idx)
    return [c for c in clusters if len(c) > 0]

def _choose_cluster_to_join(
    X: NDArray,
    p: NDArray, core_label: clust_idx_t,
    _nearby_idxs: NDArray[idx_t], 
    _dists: NDArray[float_t], 
    classifications_buff: NDArray[clust_idx_t], 
    far: float,
    orig_cluster_sizes: NDArray[size_t],
    min_cluster_size: int,
)->clust_idx_t:
    """
    Select the label whose neighbors best balance frequency, angular isotropy(enclosure), and closeness to the core point.
    This mitigates joining far-away-but-dense clusters and can create more visually coherent structure
    """

    # Filter out neighbors that are far away or are in small clusters 
    dist_mask: NDArray[np.bool_] = ((_dists <= far) & (_dists > 0))
    _nearby_idxs, _dists = _nearby_idxs[dist_mask], _dists[dist_mask]

    neighbor_labels: NDArray[clust_idx_t] = classifications_buff[_nearby_idxs]
    sizes: NDArray[size_t] = orig_cluster_sizes[neighbor_labels] 
    mask: NDArray[np.bool_] = ((neighbor_labels != core_label) & (sizes >= min_cluster_size))
    if not mask.any():
        if sizes.shape[0]:
            return neighbor_labels[np.argmax(sizes)]
        return core_label
    _nearby_idxs, _dists = _nearby_idxs[mask], _dists[mask]

    # Group neighbors by label
    label_to_subset: dict[clust_idx_t, Union[NDArray[idx_t], list[idx_t]]] = {classifications_buff[i]: [] for i in _nearby_idxs} 
    if len(label_to_subset) == 1:
        return classifications_buff[_nearby_idxs[0]]     
    for i, idx in enumerate(_nearby_idxs):
        label_to_subset[classifications_buff[idx]].append(idx_t(i)) 
    label_to_subset = {k: np.array(v, dtype=idx_t) for k, v in label_to_subset.items()}

    # Return the label whose neighbors have the best geometric support
    best_label: clust_idx_t
    best_score: float = -np.inf
    for curr_label, idxs in label_to_subset.items():
        diff: NDArray[float_t] = (X[_nearby_idxs[idxs]] - p).astype(float_t)
        diff /= np.linalg.norm(diff, axis=1, keepdims=True)
        score: float_t = idxs.shape[0] - np.linalg.norm(np.sum(diff, axis=0, dtype=float_t)) # frequency * angular isotropy
        score /= _dists[idxs[0]]
        if score > best_score:
            best_score = score
            best_label = curr_label
    return best_label


def _initialize_index(
    X: NDArray, 
    exact: bool, 
    nn_kwargs: dict[str, Any], 
    n_jobs: int = -1, 
    ids: NDArray[idx_t] = np.empty(0, dtype=idx_t)
)->Union[NearestNeighbors, hnswlib.Index]:
    """
    Initialize the kNN indenx
    """

    nn_kwargs = nn_kwargs.copy()
    
    if not exact:
        dim: int = X.shape[1]
        space: Literal['l2', 'ip', 'cosine'] = nn_kwargs.get("space", _DEFAULTS_AND_CONSTS['nn_kwargs']['hnsw']['space'])
        max_elements: int = int(nn_kwargs.pop('max_elements', X.shape[0]))
        ef_construction: int = int(nn_kwargs.pop('ef_construction', _DEFAULTS_AND_CONSTS['nn_kwargs']['hnsw']['ef_construction']))
        M: int = int(nn_kwargs.pop('M', _DEFAULTS_AND_CONSTS['nn_kwargs']['hnsw']['M']))

        index = hnswlib.Index(space=space, dim=dim)
        index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M, **nn_kwargs)
        index.add_items(X, ids=(ids if ids.shape[0] else None), num_threads=n_jobs)
    else:
        nn_kwargs['metric'] = nn_kwargs.pop('metric', _DEFAULTS_AND_CONSTS['nn_kwargs']['nearest_neighbors']['metric'])
        nn_kwargs['p'] = nn_kwargs.pop('p', _DEFAULTS_AND_CONSTS['nn_kwargs']['nearest_neighbors']['p'])
        nn_kwargs['n_jobs'] = n_jobs
        index = NearestNeighbors(**nn_kwargs)
        index.fit(X)
    
    return index

def _get_neighborhoods(
    X: NDArray, 
    connectivity: int, 
    nn_obj: Union[hnswlib.Index, NearestNeighbors],
    use_heuristics: bool = True
)->tuple[NDArray[size_t], NDArray[idx_t], NDArray[float_t]]:
    """
    Generate nearest neighbors, their distances, and the number of copies, per point
    """

    n_copies_single: size_t = size_t(1) 
    
    if isinstance(nn_obj, hnswlib.Index):
        idxs, dists = nn_obj.knn_query(X, k=min(n_copies_single + connectivity, X.shape[0]))
    else:
        dists, idxs = nn_obj.kneighbors(
            X, 
            n_neighbors=min(n_copies_single + connectivity, X.shape[0]), 
            return_distance=True
        )
        params: dict = nn_obj.get_params()
        metric: str = params.get('metric')
        p: float = params.get('p')
        if use_heuristics and metric == "minkowski": 
            # Remove the power of 1/p as is done with the HNSW for l2 distance
            np.power(dists, p, out=dists)

    # Calculate the number of copies of each point, accounting for potential HNSW error
    n_copies: NDArray[size_t] = np.array([ 
        max(np.sum(arr <= 0, dtype=size_t), size_t(1)) for arr in dists 
    ])
    
    return n_copies.astype(size_t), idxs.astype(idx_t), dists.astype(float_t)
    

def _get_density_proxies(
    X: NDArray, 
    dists: NDArray[float_t],
    n_copies: NDArray[size_t],
    density_neighbors: int,
    dataset_scale: float,
    show_progress: bool = False, 
)->NDArray[float_t]:
    """
    Return a proxy value for density in the range [1, inf)
    """

    # Edge cases
    if X.shape[0] < 1:
        return np.empty(0, dtype=float_t)
    elif dataset_scale <= 0:
        return np.ones(X.shape[0], dtype=float_t) 
    if density_neighbors is not None and density_neighbors < 1:
        raise ValueError(f"density_neighbors({density_neighbors}) must be greater than 0")

    # Compute a proxy for the local density of each point.
    # Points with duplicates appear sparser so they are seeded and dealt with later
    N: int = X.shape[0]
    densities: NDArray[float_t] = np.ones(N, dtype=float_t)

    for i in range(N):
        _n_copies: int = int(n_copies[i])
        _dists: NDArray[float_t] = dists[i]
        num_neighbors_used: int = min(density_neighbors, len(_dists) - _n_copies)

        denominator: float_t = float_t(
            _dists[_n_copies + num_neighbors_used - 1] if num_neighbors_used > 0
            else dataset_scale
        )
        densities[i] = (num_neighbors_used + 1) / (denominator/dataset_scale) 
        
        if show_progress:
            print(
                f"\r\033[K| Finding densities and neighborhoods...{round((i+1)/N*100, 2)}%",
                end=""
            )
        
    return densities

def _expand_graph_clusters(
    X: NDArray,
    
    seeding_order: Literal['none', 'random', 'density'],
    neighbors: NDArray[idx_t],
    dists: NDArray[float_t],
    n_copies: NDArray[size_t],
    densities: NDArray[float_t], 
    dataset_scale: float,
    expansion: float,
    min_retained: int,
    expansion_neighbors: int,
    
    classifications_buff: NDArray[clust_idx_t], 
    n_clusters_buff: _ClusterCount_Ref, 

    shuffle_seed: int = 42,
    show_progress: bool = False, 
):
    """
    Extract subgraphs from the dataset-wide knn graph via BFS. \\
    Edges are crossed only if their weight (distance) is below a variance threshold.
    Distance variance is learned and updated online as edges are accepted.
    """

    # Edge cases
    if X.shape[0] < 1:
        n_clusters_buff.data = 0
        return
    elif dataset_scale <= 0:
        classifications_buff[:] = 0
        n_clusters_buff.data = 1
        return

    N: int = X.shape[0]

    # Seed ordering
    ordered_idxs: LinkedList[idx_t] = LinkedList()
    if seeding_order == 'density':
        # Sort by density and secondarily by coordinate
        density_sorted_idxs: NDArray[idx_t] = np.argsort(densities)[::-1].astype(idx_t)
        equals_end = 1
        equals_start = 0
        while equals_end < len(density_sorted_idxs):
            if densities[density_sorted_idxs[equals_end]] != densities[density_sorted_idxs[equals_end - 1]]:
                if equals_end - equals_start > 1:
                    lexsorted_idxs: NDArray[np.intp] = np.lexsort(
                        X[density_sorted_idxs[equals_start : equals_end]].T[::-1]
                    ) + equals_start
                    density_sorted_idxs[equals_start : equals_end] = density_sorted_idxs[lexsorted_idxs]
                equals_start = equals_end
            equals_end += 1
        if equals_end - equals_start == len(density_sorted_idxs):
            density_sorted_idxs[:] = np.lexsort(X.T[::-1])
        del equals_end, equals_start
        
        ordered_idxs.extend(density_sorted_idxs)
        del density_sorted_idxs
    elif seeding_order == 'random':
        ordered_idxs.extend(np.random.default_rng(shuffle_seed).permutation(N).astype(idx_t))
    else:
        ordered_idxs.extend([idx_t(i) for i in range(N)])
    
    # Store an array mapping each index to its linked list node in the density sorted indexes
    # to allow for O(1) access and O(1) deletion
    node_ptrs: list[Optional[LinkedList.Node[idx_t]]] = [None for _ in range(N)]
    for node in ordered_idxs:
        node_ptrs[node.data] = node 

    # Clustering
    curr_cluster_idx: int = 0
    n_clusters_buff.data = curr_cluster_idx
    total_clustered: int = 0
    starting_running_avg_edge_len: float_t = np.mean(dists[:, :expansion_neighbors + 1], dtype=float_t)
    starting_running_edge_len_deviation: float_t = np.std(dists[:, :expansion_neighbors + 1], dtype=float_t)
    
    while len(ordered_idxs):
        unvisited: int = curr_cluster_idx + 1 
        
        start_idx: idx_t = ordered_idxs.pop(0) 
        node_ptrs[start_idx] = None
        
        cluster_size: int = 0
        to_visit: deque[idx_t] = deque([start_idx])
        running_avg_edge_len: float_t = starting_running_avg_edge_len
        running_edge_len_deviation: float_t = starting_running_edge_len_deviation
        
        while len(to_visit):
            # Aquire and filter neighbors by visitation status and z-score in knn distance
            curr_idx: idx_t = to_visit.popleft()

            classifications_buff[curr_idx] = curr_cluster_idx
            cluster_size += 1
            total_clustered += 1

            _n_copies: size_t = n_copies[curr_idx]
            nearby: NDArray[idx_t] = neighbors[curr_idx][:_n_copies + expansion_neighbors]
            nearby_dists: NDArray[float_t] = dists[curr_idx][:_n_copies + expansion_neighbors]
            
            # Filter for unvisited points
            mask: NDArray[np.bool_] = (classifications_buff[nearby] == NOISE_LABEL)
            nearby, nearby_dists = nearby[mask], nearby_dists[mask]

            # Filter for positive z-score, not absolute value, so that nearby points are always included
            mask = (
                (nearby_dists - running_avg_edge_len)
                <= expansion*running_edge_len_deviation 
            ) 
            nearby, nearby_dists = nearby[mask], nearby_dists[mask]

            # Update distance statistics and add to the visitation queue if there are enough nearby points left
            if len(nearby) >= min_retained: 
                n_dist_samples: int = cluster_size + len(to_visit)
            
                # Standard deviation in knn distance
                running_edge_len_deviation = (
                    np.square(running_edge_len_deviation)*n_dist_samples
                    + np.sum(np.square(nearby_dists - running_avg_edge_len))
                )
                running_edge_len_deviation /= n_dist_samples + len(nearby_dists)
                running_edge_len_deviation = np.sqrt(running_edge_len_deviation)
                
                # Mean of knn distance
                running_avg_edge_len = running_avg_edge_len*n_dist_samples + np.sum(nearby_dists) 
                running_avg_edge_len /= n_dist_samples + len(nearby_dists)

                # Remove the points from the sorted indexes 
                for idx in nearby:
                    ordered_idxs.pop(node_ptrs[idx]) 
                    node_ptrs[idx] = None

                # Add the points to the visitation queue
                to_visit.extend(nearby)
                classifications_buff[nearby] = unvisited
            
            if show_progress:
                print(
                    "\r\033[K| Clustering: "
                    + f"{round(total_clustered/N*100, 2)}%"
                    + f", {curr_cluster_idx} cluster(s)",
                    end=""
                )
        
        # Increment cluster index
        curr_cluster_idx += 1
        n_clusters_buff.data = curr_cluster_idx

    if show_progress:
        print(
            "\r\033[K| Clustering: 100%"
            + f", {curr_cluster_idx} cluster(s)",
            end=""
        )

def _reassign_clusters(
    X: NDArray, 

    use_heuristics: bool,
    
    neighbors: NDArray[idx_t],
    dists: NDArray[float_t],
    n_copies: NDArray[size_t],
    densities: NDArray[float_t],
    dataset_scale: float,

    expansion_neighbors: int,
    min_cluster_size: int,
    small_cluster_policy: Literal['reassign', 'noise'],
    post_reassignment_policy: Literal['noise', 'none'],
    reassignment_neighbors: int,
    far_percentile: float,
    
    classifications_buff: NDArray[clust_idx_t],
    n_clusters_buff: _ClusterCount_Ref,
    show_progress: bool = False,
):
    # Edge cases
    if X.shape[0] < 1:
        n_clusters_buff.data = 0
        return 
    elif dataset_scale <= 0:
        classifications_buff[:] = 0    
        n_clusters_buff.data = 1
        return 

    N: int = X.shape[0]

    # Aquire points to relabel and, if we are reassigning, the sizes of their clusters
    recluster_list: Union[list[int], NDArray[idx_t], deque[int]] = []
    orig_cluster_sizes: Union[list[int], NDArray[size_t]] = [] 
    for g in group_by_cluster_idx(classifications_buff):
        if len(g) < min_cluster_size:
            recluster_list.extend(g)
        if small_cluster_policy == 'reassign':
            orig_cluster_sizes.append(len(g))
        g.clear()
    recluster_list = np.array(recluster_list, dtype=idx_t)
    orig_cluster_sizes = np.array(orig_cluster_sizes, dtype=size_t)
    
    # Reassigning: Reassign points of clusters that are too small to larger clusters
    if small_cluster_policy == 'reassign':
        # Sort the points for reclustering in the reverse density order since dense regions are typically seeded first,
        # allowing them to grow into large clusters
        recluster_list = recluster_list[np.argsort(densities[recluster_list])[::-1]]

        # Define a notion of 'far' beyond which other-cluster neighbors cannot be considered for joining
        far: float = np.inf
        if far_percentile < 100:
            far = float(np.percentile(dists[:, reassignment_neighbors], far_percentile))
        
        # Perform reassignment
        recluster_list = deque(recluster_list)
        n_to_recluster: int = len(recluster_list)
        n_reclustered: int = 0
        log2N: int = max(1, int(np.log2(N)))
        while len(recluster_list):
            idx: int = recluster_list.popleft()
            k: int = min(
                reassignment_neighbors,
                max( 
                    log2N,
                    expansion_neighbors,
                    int(np.sqrt( min_cluster_size * orig_cluster_sizes[classifications_buff[idx]] )),
                ) if use_heuristics
                else reassignment_neighbors
            )
            _n_copies: size_t = n_copies[idx]
            _nearby_idxs: NDArray[idx_t] = neighbors[idx][:_n_copies + k]
            _dists: NDArray[float_t] = dists[idx][:_n_copies + k]
            if len(_nearby_idxs) <= _n_copies:
                continue
            
            classifications_buff[idx] = _choose_cluster_to_join(
                X, X[idx], classifications_buff[idx], _nearby_idxs, _dists, classifications_buff, 
                far, orig_cluster_sizes, min_cluster_size,
            )
            
            if show_progress:
                n_reclustered += 1
                print(
                    f"\r\033[K| Reassigning points: {round(n_reclustered/n_to_recluster*100, 2)}%",
                    end=""
                )

        # Cleanup remaining small clusters
        if post_reassignment_policy == 'noise':
            groups: list[list[int]] = group_by_cluster_idx(classifications_buff)
            remaining: int = sum(len(g) for g in groups if len(g) < min_cluster_size)
            handled: int = 0
            for g in groups:
                if len(g) < min_cluster_size:
                    classifications_buff[g] = NOISE_LABEL
                    handled += len(g)
                    if show_progress:
                        print(
                            f"\r\033[K| Assigning remainder to noise: {round(handled/remaining*100, 2)}%",
                            end=""
                        )
                g.clear()

    # Noise: Treat the small clusters as noise
    else:
        classifications_buff[recluster_list] = NOISE_LABEL
        if show_progress:
            print(
                f"\r\033[K| Assigned remainder to noise: 100%",
                end=""
            )
    
    del recluster_list
 
    # Redetermine cluster indexes/labels - indexes should have no gaps from the min to the max
    curr_cluster_idx: int
    handled: int = 0
    for curr_cluster_idx, g in enumerate(group_by_cluster_idx(classifications_buff)): 
        if classifications_buff[g[0]] != NOISE_LABEL:
            classifications_buff[g] = curr_cluster_idx
        handled += len(g)
        if show_progress:
            print(f"\r\033[K| Densifying labels: {handled/N*100}%", end="")
        g.clear()

    # Recalculate the number of clusters
    n_clusters_buff.data = int(np.max(classifications_buff) + 1)
        
    if show_progress:
        print(f"\r\033[K| Reassigned 100%, {n_clusters_buff.data} cluster(s) found", end="")
        

class SPORE(BaseEstimator, ClusterMixin):
    """
    **SPORE (Skeleton Propagation Over Recalibrating Expansions)** clusters data by BFS-expanding 
    subgraphs of a global k-NN graph, seeded in descending order of local density. Edges are accepted only 
    if their distance falls within a z-score threshold of the cluster's running mean and standard deviation.
    After initial assignment, small clusters may be merged into larger neighbors using 
    a composite score weighing proximity, relative size, density, and angular isotropy. 
    This yields robustness to both nonconvex geometry (driven by expansion) and weak separation (conservative expansion + merging).        
    """

    class DataIndex:
        def __init__(
            self, connectivity=None, n_copies=None, neighbors=None, dists=None, 
            dataset_scale=None, use_heuristics=None,
        ):
            self.connectivity = connectivity
            self.n_copies = n_copies
            self.neighbors = neighbors
            self.dists = dists
            self.dataset_scale = dataset_scale
            self.use_heuristics = use_heuristics
        
        def clear(self):
            self.connectivity = \
            self.n_copies = self.neighbors = self.dists = \
            self.dataset_scale = self.use_heuristics = None

    def __init__( 
        self,
        neighborhood_percentile: Optional[float] = None, 
        retention_rate: Optional[float] = None, 
        min_cluster_size: Optional[Union[float, int]] = None, 
        seeding_order: Optional[Literal['none', 'random', 'density']] = None, 
        
        expansion: Optional[float] = None, 
        expansion_neighbors: Optional[int] = None, 
        min_retained: Optional[int] = None, 
        far_percentile: Optional[float] = None, 
        reassignment_neighbors: Optional[int] = None, 
        density_neighbors: Optional[int] = None, 
        min_connectivity: Optional[int] = None,
        max_connectivity: Optional[int] = None, 
        small_cluster_policy: Optional[Literal['reassign', 'noise']] = None, 
        post_reassignment_policy: Optional[Literal['noise', 'none']] = None, 
        dindex: Optional[DataIndex] = None, 
        manage_dindex: Optional[bool] = None, 
        
        use_heuristics: Optional[bool] = None, 
        
        exact_knn: Optional[bool] = None, 
        nn_kwargs: Optional[dict[str, Any]] = None, 
        shuffle_for_hnsw: Optional[bool] = None, 
        shuffle_seed: Optional[int] = None, 
        n_jobs: Optional[int] = None, 
        
        show_progress: Optional[bool] = None, 
    ): 
        """
        Initialize SPORE

        Parameters
        ----------
        neighborhood_percentile : float, optional
            Percentile of the k-th nearest neighbor distance (with `k = expansion_neighbors`),
            converted to a z-score and used to estimate the expansion threshold.
            This parameter has lower precedence than `expansion`; if `expansion` is provided,
            this value is ignored. 
            This parameter is intended as a bounded, less-sensitive way to estimate `expansion`.

        retention_rate : float, optional
            Fractional version of `min_retained`, scaled relative to `expansion_neighbors`.
            Takes precedence over `min_retained` if provided. This bounded, normalized form
            is generally more stable and easier to tune across different neighbor counts.

        min_cluster_size : float or int, optional
            The size above which reassignment will not be attempted for a post-expansion cluster. 
            A floating point value is internally computed as `N**min_cluster_size`.

        seeding_order : {'none', 'random', 'density'}, optional
            Strategy used to determine the order in which cluster seeds are initialized:
            - 'density': Seeds clusters in ascending k-NN distance order (i.e., descending
            density), preventing sparse regions from overtaking dense ones. Ties are
            broken lexicographically by coordinates.
            - 'random': Seeds clusters in random order.
            - 'none': Uses the input data order without reordering.

        expansion : float, optional
            Upper z-score threshold on k-NN distance defining the boundary between normal
            density variation and abnormally large distances. Conceptually represents a
            cutoff for low-density transitions during cluster expansion.

        expansion_neighbors : int, optional
            Number of nearest neighbors considered during the expansion phase.

        min_retained : int, optional
            Minimum number of neighbors that must remain after filtering during expansion
            for traversal to continue. This prevents propagation through thin bridges
            between clusters or regions.

        far_percentile : float, optional
            Percentile of the k-th nearest neighbor distance (with `k = reassignment_neighbors`) 
            defining a distance that is considered too far to join a cluster.

        reassignment_neighbors : int, optional
            Number of neighbors used when reassigning points.

        density_neighbors : int, optional
            Number of neighbors used for ranking points by k-NN distance, a proxy for local density.

        min_connectivity : int, optional
            Minimum number of neighbors fetched per point when building the global k-NN
            graph. 

        max_connectivity : int, optional
            Maximum number of neighbors fetched per point when building the global k-NN
            graph. 
            
        small_cluster_policy : {'reassign', 'noise'}, optional
            Policy for handling clusters that are smaller than the minimum size:
            - 'reassign': Reassigns points from small clusters to nearby valid clusters,
            subject to constraints.
            - 'noise': Merges all small clusters into a single noise cluster.

        post_reassignment_policy : {'noise', 'none'}, optional
            Policy applied after reassignment:
            - 'noise': Remaining small clusters are assigned to noise.
            - 'none': Remaining small clusters are left unchanged.

        dindex : DataIndex, optional
            An object storing a preconstructed nearest-neighbor graph, 
            nearest-neighbor distances, and other data. It is used to avoid 
            rebuilding k-NN structures and reduce overhead.

        manage_dindex : bool, optional
            Whether the algorithm should automatically manage the lifecycle of `dindex`
            (e.g., replacing or clearing internal data as needed).

        use_heuristics : bool, optional
            Enables a collection of heuristics designed to improve stability, performance,
            and execution speed.

        exact_knn : bool, optional
            Whether to use exact nearest-neighbor search. The algorithm can tolerate
            approximate neighbors due to its statistical, evolving expansion logic.

        nn_kwargs : dict[str, Any], optional
            Additional keyword arguments passed to the nearest-neighbor backend.

        shuffle_for_hnsw : bool, optional
            Whether to shuffle data indices before building the HNSW index. Shuffling
            significantly improves stability and accuracy of k-NN results.

        shuffle_seed : int, optional
            Random seed used for shuffling and for random seeding order when applicable.

        n_jobs : int, optional
            Number of threads used during k-NN index construction. A value of -1 uses all
            available cores.

        show_progress : bool, optional
            Whether to display progress information in the terminal during clustering.
        """

        # Validate and save parameters
        if neighborhood_percentile is not None:
            _bounds_check(neighborhood_percentile, 0, 100, f"neighborhood percentile({neighborhood_percentile}) must be in the range [0,100]")
        if retention_rate is not None:
            _bounds_check(retention_rate, 0, 1, f"retention_rate({retention_rate}) must be in the range [0,1]")
        if far_percentile is not None:
            _bounds_check(far_percentile, 0, 100, f"far_percentile({far_percentile}) must be in the range [0, 100]")
        if seeding_order is not None and seeding_order not in ("none", "random", "density"):
            raise ValueError(f"seeding_order({seeding_order}) must be one of 'none', 'random', or 'density'")
        if small_cluster_policy is not None and small_cluster_policy not in ("reassign", "noise"):
            raise ValueError(f"small_cluster_policy({small_cluster_policy}) must be one of 'reassign' or 'noise'")
        if post_reassignment_policy is not None and post_reassignment_policy not in ("noise", "none"):
            raise ValueError(f"post_reassignment_policy({post_reassignment_policy}) must be one of 'noise' or 'none'")
        if shuffle_seed is not None:
            _bounds_check(shuffle_seed, 0, np.inf, f"shuffle_seed({shuffle_seed}) must be between 0 and inf")
        if n_jobs is not None:
            _bounds_check(n_jobs, -1, np.inf, f"n_jobs({n_jobs}) must be between -1 and inf")
            if n_jobs == 0:
                raise ValueError("n_jobs cannot be 0")
            
        self.dindex = dindex
        self.manage_dindex = manage_dindex
        self.exact_knn = exact_knn
        self.nn_kwargs = nn_kwargs
        self.n_jobs = n_jobs
        self.shuffle_for_hnsw = shuffle_for_hnsw
        self.shuffle_seed = shuffle_seed
        self.seeding_order = seeding_order
        self.use_heuristics = use_heuristics
        self.density_neighbors = density_neighbors
        self.retention_rate = retention_rate
        self.min_retained = min_retained 
        self.expansion_neighbors = expansion_neighbors
        self.reassignment_neighbors = reassignment_neighbors
        self.min_connectivity = min_connectivity
        self.max_connectivity = max_connectivity
        self.expansion = expansion
        self.neighborhood_percentile = neighborhood_percentile
        self.far_percentile = far_percentile
        self.min_cluster_size = min_cluster_size
        self.small_cluster_policy = small_cluster_policy
        self.post_reassignment_policy = post_reassignment_policy
        self.show_progress = show_progress
        self.labels_: Optional[NDArray[clust_idx_t]] = None
        self.n_clusters_: Optional[int] = None

    def _resolve_params(self, N: int, D: int):
        # Extract parameters from the self
        dindex = self.dindex
        manage_dindex = self.manage_dindex
        exact_knn = self.exact_knn
        nn_kwargs = self.nn_kwargs
        n_jobs = self.n_jobs
        shuffle_for_hnsw = self.shuffle_for_hnsw
        shuffle_seed = self.shuffle_seed
        seeding_order = self.seeding_order
        use_heuristics = self.use_heuristics
        density_neighbors = self.density_neighbors
        retention_rate = self.retention_rate
        min_retained = self.min_retained
        expansion_neighbors = self.expansion_neighbors
        reassignment_neighbors = self.reassignment_neighbors
        min_connectivity = self.min_connectivity
        max_connectivity = self.max_connectivity
        expansion = self.expansion
        neighborhood_percentile = self.neighborhood_percentile
        far_percentile = self.far_percentile
        min_cluster_size = self.min_cluster_size
        small_cluster_policy = self.small_cluster_policy
        post_reassignment_policy = self.post_reassignment_policy
        show_progress = self.show_progress

        # Validate parameters
        _bounds_check(N, 2, np.inf, f"The number of points in the dataset({N}) must be in the range [2,inf)")
        _bounds_check(D, 1, np.inf, f"The number of dimensions in the dataset({D}) must be in the range [1,inf)")
        if expansion_neighbors is not None:
            _bounds_check(
                expansion_neighbors, 1, N - 1, 
                f"expansion_neighbors({expansion_neighbors}) must be between 1 and the number of datapoints - 1({N - 1})"
            )
        if retention_rate is None and min_retained is not None:
            _bounds_check(
                min_retained, 1, (N - 1 if expansion_neighbors is None else expansion_neighbors), 
                f"min_retained({min_retained}) must be between 1 and {f'N - 1({N - 1})' if expansion_neighbors is None else f'expansion_neighbors({expansion_neighbors})'}"
            )
        if min_cluster_size is not None:
            _bounds_check(
                min_cluster_size, 0, N, 
                f"min_cluster_size({min_cluster_size}) must be between 0 and the number of datapoints({N})"
            )
        if reassignment_neighbors is not None:
            _bounds_check(
                reassignment_neighbors, 1, N - 1, 
                f"reassignment_neighbors({reassignment_neighbors}) must be between 1 and the number of datapoints({N}) - 1"
            )
        if density_neighbors is not None:
            _bounds_check(density_neighbors, 1, N - 1, f"density_neighbors({density_neighbors}) must be between 1 and the number of datapoints - 1({N - 1})")
        if min_connectivity is not None:
            _bounds_check(
                min_connectivity, 1, N - 1, 
                f"min_connectivity({min_connectivity}) must be between 1 and the number of datapoints({N}) - 1"
            )
        if max_connectivity is not None:
            _bounds_check(
                max_connectivity, 1, N - 1, 
                f"max_connectivity({max_connectivity}) must be between 1 and the number of datapoints({N}) - 1"
            )

        # Define or resolve final internal parameter values
        exact_knn = (exact_knn if exact_knn is not None else _DEFAULTS_AND_CONSTS['exact_knn'])
        nn_kwargs = (nn_kwargs if nn_kwargs is not None else dict())
        n_jobs = (n_jobs if n_jobs is not None else _DEFAULTS_AND_CONSTS['n_jobs'])
        shuffle_for_hnsw = (shuffle_for_hnsw if shuffle_for_hnsw is not None else _DEFAULTS_AND_CONSTS['shuffle_for_hnsw'])
        shuffle_seed = (shuffle_seed if shuffle_seed is not None else _DEFAULTS_AND_CONSTS['shuffle_seed'])
        seeding_order = (seeding_order if seeding_order is not None else _DEFAULTS_AND_CONSTS['seeding_order'])
        use_heuristics = (use_heuristics if use_heuristics is not None else _DEFAULTS_AND_CONSTS['use_heuristics'])
        min_connectivity = (min_connectivity if min_connectivity is not None else _DEFAULTS_AND_CONSTS['min_connectivity'])
        max_connectivity = (max_connectivity if max_connectivity is not None else N - 1)
        far_percentile = (
            far_percentile if far_percentile is not None 
            else _DEFAULTS_AND_CONSTS['far_percentile'] if use_heuristics
            else 100
        )
        small_cluster_policy = (small_cluster_policy if small_cluster_policy is not None else _DEFAULTS_AND_CONSTS['small_cluster_policy'])
        post_reassignment_policy = (post_reassignment_policy if post_reassignment_policy is not None else _DEFAULTS_AND_CONSTS['post_reassignment_policy'])
        show_progress = (show_progress if show_progress is not None else _DEFAULTS_AND_CONSTS['show_progress'])

        if expansion is not None:
            neighborhood_percentile = None
        elif neighborhood_percentile is None:
            neighborhood_percentile = _DEFAULTS_AND_CONSTS['neighborhood_percentile']
        
        log2N = max(1, int(np.log2(N)))
        if expansion_neighbors is None:
            expansion_neighbors = min(max_connectivity, int(N**0.4), 2*log2N)
        min_retained = (
            max(int(retention_rate * expansion_neighbors), 1) if retention_rate is not None
            else _DEFAULTS_AND_CONSTS['min_retained'] if min_retained is None
            else min_retained
        )
        if density_neighbors is None: 
            density_neighbors = min(max_connectivity, log2N)
        
        min_cluster_size = int(
            (
                min_cluster_size if min_cluster_size >= 1 else int(N**min_cluster_size)
            ) if min_cluster_size is not None
            else int(N**_DEFAULTS_AND_CONSTS['min_cluster_size']) 
        )
        reassignment_neighbors = min(
            max_connectivity,
            reassignment_neighbors if reassignment_neighbors is not None
            else min(
                _bounded_sample_count(N, 4*log2N),
                min_cluster_size,
            ) if use_heuristics 
            else min_cluster_size
        )

        # Define final connectivity (degree of the knn graph)
        connectivity = min(max_connectivity, max(density_neighbors, reassignment_neighbors, expansion_neighbors, min_connectivity))

        # Potentially clear out the DataIndex object to be repopulated later
        if dindex is not None and manage_dindex and \
        (dindex.connectivity != connectivity or dindex.use_heuristics != use_heuristics): 
            dindex.clear()
        
        # Save resolved values within self
        self._dindex = (dindex if dindex is not None else SPORE.DataIndex())
        self.manage_dindex_ = manage_dindex
        self.exact_knn_ = exact_knn
        self.nn_kwargs_ = nn_kwargs.copy()
        self.n_jobs_ = n_jobs
        self.shuffle_for_hnsw_ = shuffle_for_hnsw
        self.shuffle_seed_ = shuffle_seed
        self.seeding_order_ = seeding_order
        self.use_heuristics_ = use_heuristics
        self.connectivity_ = connectivity
        self.density_neighbors_ = density_neighbors
        self.retention_rate_ = retention_rate
        self.min_retained_ = min_retained 
        self.expansion_neighbors_ = expansion_neighbors
        self.reassignment_neighbors_ = reassignment_neighbors
        self.min_connectivity_ = min_connectivity
        self.max_connectivity_ = max_connectivity
        self.expansion_ = expansion
        self.neighborhood_percentile_ = neighborhood_percentile
        self.far_percentile_ = far_percentile
        self.min_cluster_size_ = min_cluster_size
        self.small_cluster_policy_ = small_cluster_policy
        self.post_reassignment_policy_ = post_reassignment_policy
        self.show_progress_ = show_progress

    def fit(self, X: NDArray, y=None, **fit_params):
        # Validate and resolve parameters
        check_array(X, ensure_2d=True) 
        self._resolve_params(*X.shape)
        
        # Fit to the dataset
        dindex_only = fit_params.get("dindex_only", False)
        manage_dindex = self.manage_dindex_
        dindex = self._dindex

        N = X.shape[0]
        use_heuristics = self.use_heuristics_
        connectivity = self.connectivity_
        expansion_neighbors = self.expansion_neighbors_
        exact_knn = self.exact_knn_
        nn_kwargs = self.nn_kwargs_.copy()
        neighborhood_percentile = self.neighborhood_percentile_
        n_jobs = self.n_jobs_
        shuffle_for_hnsw = self.shuffle_for_hnsw_
        shuffle_seed = self.shuffle_seed_

        # Define or retrieve data needed for clustering
        if dindex.connectivity:
            nn_obj = None
            n_copies, neighbors, dists, dataset_scale = ( 
                dindex.n_copies, dindex.neighbors, dindex.dists, dindex.dataset_scale,
            )
        else:
            dataset_scale: float = 1.0
            space, metric = nn_kwargs.get("space"), nn_kwargs.get("metric")
            if (
                (space is None or space == "l2") and
                (metric is None or metric == "minkowski")
            ):
                dataspace_dims: NDArray[float_t] = (np.max(X, axis=0) - np.min(X, axis=0)).astype(float_t)
                dataset_scale = float(np.power(np.linalg.norm(dataspace_dims), 1 + use_heuristics))

            if not exact_knn and "random_seed" not in nn_kwargs:
                nn_kwargs = dict(**nn_kwargs, random_seed=_DEFAULTS_AND_CONSTS['nn_kwargs']['hnsw']['random_seed'])
            
            shuffled_idxs: NDArray[idx_t] = np.empty(0, dtype=idx_t)
            if not exact_knn and shuffle_for_hnsw:
                shuffled_idxs = np.random.default_rng(shuffle_seed).permutation(N).astype(dtype=idx_t)
            
            nn_obj = _initialize_index(
                (X[shuffled_idxs] if shuffled_idxs.shape[0] else X), exact=exact_knn, nn_kwargs=nn_kwargs, 
                n_jobs=n_jobs, ids=shuffled_idxs
            )
            n_copies, neighbors, dists = _get_neighborhoods(X=X, connectivity=connectivity, nn_obj=nn_obj, use_heuristics=use_heuristics)

        # Compute expansion from a percentile of knn distance
        if neighborhood_percentile is not None and not dindex_only:
            neighborhoods: NDArray[float_t] = dists[:, expansion_neighbors]
            mean_neighborhood: float_t = np.mean(neighborhoods, dtype=float_t)
            std_neighborhood: float_t = np.std(neighborhoods, dtype=float_t)
            neighborhoods = neighborhoods[neighborhoods >= mean_neighborhood]
            self.expansion_ = (
                (
                    np.percentile(neighborhoods, neighborhood_percentile) - mean_neighborhood
                ) / std_neighborhood 
                if std_neighborhood > 0 
                else 0.0
            )

        # Save results within the self and potentially update the DataIndex object
        self._neighbors = neighbors
        self._dists = dists
        self._n_copies = n_copies
        self._dataset_scale = dataset_scale

        if manage_dindex or dindex_only:
            dindex.connectivity, \
            dindex.n_copies, dindex.neighbors, dindex.dists, \
            dindex.dataset_scale, dindex.use_heuristics = (
                self.connectivity_, n_copies, neighbors, dists, dataset_scale, use_heuristics,
            )

        # Perform clustering if desired
        if not dindex_only:
            self._cluster(X=X)

        # Remove the internal reference to DataIndex data to clear memory in this object.
        self._dindex = SPORE.DataIndex()
        del self._neighbors, self._dists, self._n_copies, self._dataset_scale

        return self

    def _cluster(self, X: NDArray):
        N = X.shape[0]

        seeding_order = self.seeding_order_
        shuffle_seed = self.shuffle_seed_
        use_heuristics = self.use_heuristics_
        density_neighbors = self.density_neighbors_
        min_retained = self.min_retained_
        expansion_neighbors = self.expansion_neighbors_
        reassignment_neighbors = self.reassignment_neighbors_
        expansion = self.expansion_
        far_percentile = self.far_percentile_
        min_cluster_size = self.min_cluster_size_
        small_cluster_policy = self.small_cluster_policy_
        post_reassignment_policy = self.post_reassignment_policy_
        show_progress = self.show_progress_
        dists = self._dists
        dataset_scale = self._dataset_scale
        n_copies = self._n_copies
        neighbors = self._neighbors
        
        self.labels_ = classifications_buff = np.full(N, NOISE_LABEL, dtype=clust_idx_t)
        n_clusters_buff = _ClusterCount_Ref(0)

        # Aquire duplicate-aware density proxies for seeding order
        densities = _get_density_proxies(
            X=X, dists=dists, n_copies=n_copies,  
            density_neighbors=density_neighbors, dataset_scale=dataset_scale,
            show_progress=show_progress
        )
        
        # Expansion
        _expand_graph_clusters(
            X=X, seeding_order=seeding_order, 
            neighbors=neighbors, dists=dists, n_copies=n_copies,
            densities=densities, dataset_scale=dataset_scale, 
            expansion=expansion, min_retained=min_retained, expansion_neighbors=expansion_neighbors,  
            classifications_buff=classifications_buff, n_clusters_buff=n_clusters_buff, 
            shuffle_seed=shuffle_seed,
            show_progress=show_progress,
        )

        if min_cluster_size > 1:
            # Reassignment
            _reassign_clusters(
                X=X, use_heuristics=use_heuristics, 
                neighbors=neighbors, dists=dists, n_copies=n_copies, 
                densities=densities, dataset_scale=dataset_scale,
                expansion_neighbors=expansion_neighbors, min_cluster_size=min_cluster_size, 
                small_cluster_policy=small_cluster_policy, post_reassignment_policy=post_reassignment_policy, 
                reassignment_neighbors=reassignment_neighbors, far_percentile=far_percentile,
                classifications_buff=classifications_buff, n_clusters_buff=n_clusters_buff, 
                show_progress=show_progress, 
            )

        # Save the number of clusters
        self.n_clusters_ = n_clusters_buff.data

        if show_progress:
            print(
                f"\r\033[K| Clustering complete, {n_clusters_buff.data} cluster(s) found",
                end=""
            )

        if show_progress:
            print()

    def fit_predict(self, X: NDArray, y=None)->NDArray[clust_idx_t]:
        return self.fit(X=X).labels_ 
