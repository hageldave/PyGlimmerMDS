#pip install numpy numba
import numpy as np
import numba as nb
import math

def glimmer(
        data: np.ndarray,
        initialization: np.ndarray = None,
        decimation_factor=2,
        neighbor_set_size=8,
        max_iter=512,
        min_level_size=1000,
        rng=None,
        callback=None,
        verbose=True
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    if initialization is None:
        norms = np.linalg.norm(data, axis=1)
        initialization = rng.random((data.shape[0], 2))-0.5
        initialization *= (norms/np.linalg.norm(initialization, axis=1))[:,None]
    if callback is None:
        callback = noop
    embedding = initialization

    callback(embedding)

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
            current_embedding, current_forces, stress = layout(current_data, current_embedding, current_forces, current_neighbors)
            embedding[current_index_set] = current_embedding
            callback(embedding)
            forces[current_index_set] = current_forces
            # sort neighbor sets according to distance
            sort_neighbors(current_data, current_neighbors)
            # replace the latter half of the neighbors randomly
            neighbors[current_index_set, neighbor_set_size // 2:] = np.stack(
                [rng.choice(current_n, neighbor_set_size // 2, replace=False) for _ in range(current_n)])
            stresses.append(stress/current_n)
            sm_stress = smooth_stress(np.array(stresses))
            if sm_stress_prev < float('inf'):
                stress_ratio = sm_stress_prev / sm_stress
                # early stopping if stress improvement is only very little
                if 1.0 >= stress_ratio > 0.99:
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
    return layout_numba(data, embedding, forces, neighbors, start, end)


#@nb.njit(cache=True, fastmath=True)
def layout_numba(data: np.ndarray, embedding: np.ndarray, forces: np.ndarray, neighbors: np.ndarray, start:int, end:int):
    # forces_new = np.zeros_like(forces)
    # for i in nb.prange(start, end):
    #     point_hi = data[i]
    #     neighbor_points_hi = data[neighbors[i]]
    #     dists_hi = np.sqrt(((neighbor_points_hi - point_hi) ** 2).sum(axis=1))
    #     point_lo = embedding[i]
    #     neighbor_points_lo = embedding[neighbors[i]]
    #     delta = neighbor_points_lo - point_lo
    #     dists_lo = np.sqrt((delta**2).sum(axis=1)) + 1e-8
    #     scalings = np.expand_dims(1 - dists_hi/dists_lo, axis=1)
    #     delta, scalings = np.broadcast_arrays(delta, scalings)
    #     force_update = (delta * scalings).sum(axis=0)
    #     forces_new[i] = forces[i]*0.1 + 0.1*force_update
    # embedding[start:end] += forces_new[start:end]
    # return embedding, forces_new

    forces_new = np.zeros_like(forces)
    neighbor_points_hi = data[neighbors]
    n_neighbors = neighbors.shape[1]
    normalize_factor = 1.0/n_neighbors
    diff = neighbor_points_hi - data[:,None,:]
    dists_hi = np.sqrt((diff**2).sum(axis=-1))
    neighbor_points_lo = embedding[neighbors]
    delta = neighbor_points_lo - embedding[:,None,:]
    dists_lo = np.sqrt((delta ** 2).sum(axis=-1)) + 1e-8
    scalings = np.expand_dims(1 - dists_hi / dists_lo, axis=-1)
    delta, scalings = np.broadcast_arrays(delta, scalings)
    force_update = (delta * scalings).sum(axis=1) * normalize_factor
    forces_new = forces * 0.5 + force_update
    embedding[start:end] += forces_new[start:end]
    stress = ((dists_hi - dists_lo)**2).sum()
    return embedding, forces_new, stress


def noop(*args):
    pass


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

    #embedding = glimmer(data, decimation_factor=2, max_iter=512, neighbor_set_size=16)
    embedding = glimmer(data)
    p = bkp.figure()
    p.scatter(embedding[:,0], embedding[:,1], size=1)
    bkp.show(p)
