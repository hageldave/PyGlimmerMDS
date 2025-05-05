#pip install numpy numba
import numpy as np
import numba as nb


def execute_glimmer(
        data: np.ndarray,
        initialization: np.ndarray = None,
        target_dim = None,
        decimation_factor=2,
        neighbor_set_size=8,
        max_iter=512,
        min_level_size=1000,
        rng=None,
        callback=None,
        verbose=True,
        stress_ratio_tol = 1 - 1e-5
) -> np.ndarray:
    """
    Execute the glimmer algorithm to perform multidimensional scaling on the provided data set.

    Parameters
    ----------
    data : np.ndarray
        the high-dimensional data set for which multidimensional scaling is performed. (2D array)
    initialization: np.ndarray
        [optional] initial low-dimensional embedding (2D array). If None, random initialization will be used.
    target_dim: int
        [optional] dimensionality of embedding. Only used if initialization is None.
    decimation_factor: int
        factor by which the data set is divided into smaller sets for the different
        levels. Larger factor results in less levels.
        E.g., n=10,000 f=2, level sizes result in: 10,000, 10,000/2=5000, 5000/2=2500, 2500/2=1250.
    neighbor_set_size: int
        [optional] the number of neighbors used with every data point. Needs to be divisible by 2,
        in order for the first half relating to the nearest neighbors (near set), and the latter half to the random
        neighbors (far set).
    max_iter: int
        [optional] maximum number of iterations per level.
    min_level_size: int
        [optional] minimum number of points in the smallest level.
    rng: np.random.Generator
        [optional] random number generator object.
    callback: function(dict)
        [optional] callback function which will be called in each iteration of the algorithm.
        The function argument is a dictionary containing several internal variables, i.e.,
        embedding, forces, current level, current iteration, index set of the current level, stress, smoothed stress.
    verbose: bool
        [optional] if True, will print info about execution.
    stress_ratio_tol: float
        [optional] early stopping criterion: when [current stress]/[previous stress] > stress_ratio_tol, stop.
        Meaning when stress improvement is negligible, terminate the current level.

    Returns
    -------
    np.ndarray
        the low-dimensional embedding (2D array)
    """
    if rng is None:
        rng = np.random.default_rng()
    if initialization is None:
        if target_dim is None:
            target_dim = 2
        norms = np.linalg.norm(data, axis=1)
        initialization = rng.random((data.shape[0], target_dim))-.5
        initialization *= (norms/np.linalg.norm(initialization, axis=1))[:,None]
    if callback is None:
        callback = lambda *args: None
    # sanity checking
    if target_dim and initialization.shape[1] != target_dim:
        import warnings
        warnings.warn(f"provided target dimension {target_dim} does not match initialization shape[1]={initialization.shape[1]}")

    if initialization.shape[0] != data.shape[0]:
        raise Exception(f"provided initialization shape[0]={initialization.shape[0]} does not match data shape[0]={data.shape[0]}")

    embedding = initialization
    forces = np.zeros_like(embedding)
    n = data.shape[0]
    # generate randomized indices
    rand_indices = rng.permutation(n)
    # generate array for storing neighbor set of each point
    neighbors = np.zeros((n,neighbor_set_size), dtype=int)
    # generate level sizes
    level_sizes = [n]
    n_levels=0
    while level_sizes[n_levels] >= min_level_size*decimation_factor:
        level_sizes.append(level_sizes[n_levels]//decimation_factor)
        n_levels += 1
    n_levels += 1
    if verbose:
        print(f"levels: {n_levels}, level sizes: {level_sizes[::-1]}")

    # start at lowest level
    for level in range(n_levels-1, -1, -1):
        current_n = level_sizes[level]
        if verbose:
            print(f"execution on level: {level}, current n: {current_n}")
        current_index_set = rand_indices[:current_n]
        # create/update random neighbors
        if level == n_levels-1:
            # initialize neighbor sets random
            neighbors[current_index_set] = np.stack(
                [rng.choice(current_n, neighbor_set_size, replace=False) for _ in range(current_n)])
        # do the layout
        current_data = data[current_index_set]
        current_embedding = embedding[current_index_set]
        current_forces = forces[current_index_set]
        current_neighbors = neighbors[current_index_set]
        stresses = []
        sm_stress_prev = float('inf')
        for iter in range(max_iter):
            current_embedding, current_forces, stress = layout(
                current_data,
                current_embedding,
                current_forces,
                current_neighbors)
            embedding[current_index_set] = current_embedding
            forces[current_index_set] = current_forces
            # sort neighbor sets according to distance
            sort_neighbors(current_data, current_neighbors)
            # replace the latter half of the neighbors randomly
            neighbors[current_index_set, neighbor_set_size // 2:] = np.stack(
                [rng.choice(current_n, neighbor_set_size // 2, replace=False) for _ in range(current_n)])
            stresses.append(stress/current_n)
            sm_stress = smooth_stress(np.array(stresses))

            callback(dict(
                embedding=embedding,
                forces=forces,
                level=level,
                iter=iter,
                index_set=current_index_set,
                smoothed_stress=sm_stress,
                stress=stresses[-1]))

            if verbose and iter % 10 == 0:
                print(f"stress after iteration {iter}: {stresses[-1]} smoothed stress: {sm_stress}")
            if sm_stress_prev < float('inf'):
                stress_ratio = sm_stress / sm_stress_prev
                # early stopping if stress improvement is only very little
                if 1.0 >= stress_ratio > stress_ratio_tol:
                    if verbose:
                        print(f"early termination of level {level} after {iter} iterations")
                    break
            sm_stress_prev = sm_stress
            
        if level > 0:
            # initialize neighbors for next level
            next_n = level_sizes[level-1]
            next_index_set = rand_indices[:next_n]
            neighbors[next_index_set[current_n:next_n]] = np.stack(
                [rng.choice(current_n, neighbor_set_size, replace=False) for _ in range(next_n-current_n)])
            # relaxation step, only moving the new points from next level during layout
            for _ in range(8):
                embedding[next_index_set], _, _ = layout(
                    data[next_index_set],
                    embedding[next_index_set],
                    forces[next_index_set],
                    neighbors[next_index_set])

    return embedding


@nb.njit(cache=True, fastmath=True)
def sort_neighbors(data: np.ndarray, neighbors: np.ndarray):
    for i in nb.prange(data.shape[0]):
        point = data[i]
        neighbor_points = data[neighbors[i]]
        dists_squared = ((neighbor_points - point) ** 2).sum(axis=1)
        neighbors[i] = neighbors[i][np.argsort(dists_squared)]
    return neighbors


def smooth_stress(stresses: np.ndarray) -> float:
    if len(stresses) < 8:
        return float('inf')
    else:
        # TODO: convolution with kernel
        return stresses[-8:].mean()


def layout(data: np.ndarray, embedding: np.ndarray, forces: np.ndarray, neighbors: np.ndarray, start=0, end=None):
    # for each point get neighbor points and compute forces
    if end is None:
        end = data.shape[0]
    return __compute_forces_and_layout(data, embedding, forces, neighbors, start, end)


def __compute_forces_and_layout(data: np.ndarray, embedding: np.ndarray, forces: np.ndarray, neighbors: np.ndarray, start:int, end:int):
    k_neighbors = neighbors.shape[1]
    normalize_factor = 1.0 / k_neighbors
    # for each point get its k neighbor points (3D array)
    neighbor_points_hi = data[neighbors]
    neighbor_points_lo = embedding[neighbors]
    # compute differences between respective point to neighbors
    diff = neighbor_points_hi - data[:,None,:]
    delta = neighbor_points_lo - embedding[:, None, :]
    # compute distances (lengths of the differences)
    dists_hi = np.sqrt((diff**2).sum(axis=-1))
    dists_lo = np.sqrt((delta ** 2).sum(axis=-1)) + 1e-8
    stress = ((dists_hi - dists_lo) ** 2).sum()
    # compute scale factors of the deltas
    scalings = np.expand_dims(1 - dists_hi / dists_lo, axis=-1)
    delta, scalings = np.broadcast_arrays(delta, scalings)
    # compute new forces (momentum approach, reusing old forces)
    force_update = (delta * scalings).sum(axis=1) * normalize_factor
    forces_new = forces * 0.5 + force_update
    # update embedding
    embedding[start:end] += forces_new[start:end]
    return embedding, forces_new, stress



if __name__ == '__main__':
    from sklearn import preprocessing as prep
    from bokeh.sampledata import iris
    import bokeh.plotting as bkp
    # get iris data
    dataset = iris.flowers
    data = dataset.iloc[:, 0:4].values
    for _ in range(7):
        data = np.vstack((data,data+(np.random.rand(data.shape[0], data.shape[1])*0.2-.1)))
    print(data.shape)
    data = prep.StandardScaler().fit_transform(data)

    projection = execute_glimmer(data)
    p = bkp.figure()
    p.scatter(projection[:, 0], projection[:, 1], size=1)
    bkp.show(p)
