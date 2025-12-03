#pip install numpy
import numpy as np
from .util import row_wise_duplicate_indices 


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
        stress_ratio_tol = 1 - 1e-5,
        alpha=1.0
) -> tuple[np.ndarray,float]:
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
        [optional] the number of neighbors for near and far set used with every data point. The effective
        amound of neighbors will be neighbor_set_size * 2.
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
    alpha: float
        [optional] learning rate: scale factor for gradients in gradient descent.

    Returns
    -------
    tuple[np.ndarray, float]
        the low-dimensional embedding (2D array) and corresponding (smoothed) stress.
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
    neighbors = np.zeros((n,neighbor_set_size*3), dtype=int)
    # generate level sizes
    level_sizes = [n]
    n_levels=0
    while level_sizes[n_levels] >= min_level_size*decimation_factor:
        level_sizes.append(level_sizes[n_levels]//decimation_factor)
        n_levels += 1
    n_levels += 1
    if verbose:
        print(f"levels: {n_levels}, level sizes: {level_sizes[::-1]}")

    sm_stress = None
    # start at lowest level
    for level in range(n_levels-1, -1, -1):
        current_n = level_sizes[level]
        if verbose:
            print(f"execution on level: {level}, current n: {current_n}")
        current_index_set = rand_indices[:current_n]
        # create/update random neighbors
        if level == n_levels-1:
            # initialize neighbor sets random
            neighbors[current_index_set] = __rand_indices_noduplicates_on_rows(
                current_n,
                current_n,
                neighbor_set_size*3, # only need 2 but due to possible duplicates on replace we need more space
                rng)
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
                current_neighbors[:,:neighbor_set_size*2],
                alpha=alpha)
            current_embedding -= current_embedding.mean(axis=0)
            embedding[current_index_set] = current_embedding
            forces[current_index_set] = current_forces
            # sort neighbor sets according to distance
            #__sort_neighbors(current_data, current_neighbors)
            # replace the latter half of the neighbors randomly
            new_neighbor_candidates = __rand_indices_noduplicates_on_rows(
                current_n,
                current_n,
                neighbor_set_size*2,
                rng)
            __update_neighbors(current_neighbors, new_neighbor_candidates, current_data, current_data, neighbor_set_size)
            
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
            neighbors[next_index_set[current_n:next_n]] = __rand_indices_noduplicates_on_rows(
                current_n,
                next_n-current_n,
                neighbor_set_size*3,
                rng)
            # relaxation step, only moving the new points from next level during layout
            for _ in range(8):
                embedding[next_index_set], _, _ = layout(
                    data[next_index_set],
                    embedding[next_index_set],
                    forces[next_index_set],
                    neighbors[next_index_set, :neighbor_set_size*2])

    return embedding, sm_stress


def __sort_neighbors(data: np.ndarray, neighbors: np.ndarray):
    neighbor_points_hi = data[neighbors]
    # compute differences between respective point to neighbors
    diff = neighbor_points_hi - data[:, None, :]
    # compute distances (lengths of the differences)
    dists_squared = (diff ** 2).sum(axis=-1)
    sorting = np.argsort(dists_squared, axis=1)
    neighbors[:,:] = neighbors.ravel()[sorting]
    return neighbors


def __update_neighbors(curr_neighbors, new_randoms, positions, neighbor_positions, k):
  old = curr_neighbors.copy()
  curr_neighbors[:,k:] = new_randoms
  index_order = np.argsort(curr_neighbors, axis=1)
  curr_neighbors[:,:] = np.take_along_axis(curr_neighbors, index_order, axis=1)
  # find duplicate neighbors
  indices_to_mark = row_wise_duplicate_indices(curr_neighbors)  #first_index[counts > 1]
  # determine distances
  dists_sq = ((neighbor_positions[curr_neighbors] - positions[:,None,:])**2).sum(axis=-1)
  # mark duplicates as infinitely far away
  dists_sq[indices_to_mark] = 1e16  # np.inf
  # sort by distances
  order = np.argsort(dists_sq, axis=1)
  curr_neighbors[:,:] = np.take_along_axis(curr_neighbors, order, axis=1)
  diff_near = np.argwhere(old[:,:k] != curr_neighbors[:,:k])
  return curr_neighbors


def __rand_indices_noduplicates_on_rows(max_i, n, m, rng):
    if n*m < max_i:
        print("using no batches")
        return rng.choice(max_i, (n, m), replace=False)
    batches = []
    batch_size = (max_i//m)*m
    for i in range((n*m)//batch_size+1):
        batches.append(rng.choice(max_i, batch_size, replace=False, shuffle=True))
    return np.concatenate(batches)[:n*m].reshape((n,m), order='C')



def smooth_stress(stresses: np.ndarray) -> float:
    if len(stresses) < 8:
        return float('inf')
    else:
        # TODO: convolution with kernel
        return stresses[-8:].mean()


def layout(data: np.ndarray, embedding: np.ndarray, forces: np.ndarray, neighbors: np.ndarray, start=0, end=None, alpha=1.0):
    # for each point get neighbor points and compute forces
    if end is None:
        end = data.shape[0]
    return __compute_forces_and_layout(data, embedding, forces, neighbors, start, end, alpha)


def __compute_forces_and_layout(data: np.ndarray, embedding: np.ndarray, forces: np.ndarray, neighbors: np.ndarray, start:int, end:int, alpha=1.0):
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
    embedding[start:end] += forces_new[start:end] * alpha
    return embedding, forces_new, stress
