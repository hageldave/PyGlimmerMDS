# pip install cupy-cuda12x  (adjust cuda version suffix to match your driver)
# pip install numpy
import numpy as np
import cupy as cp


# ---------------------------------------------------------------------------
# Fisher-Yates RawKernel
# ---------------------------------------------------------------------------
# Generates (n x m) unique random indices per row from [0, max_i) entirely
# on the GPU.  Each thread owns one row and runs a partial Fisher-Yates
# shuffle with an inline xorshift128+ RNG — no large scratch matrix, no
# CPU round-trip, VRAM cost is O(n*m) only.
# Constraint: m <= 100  (matches neighbor_set_size*3 <= 100 expectation).
# ---------------------------------------------------------------------------
_FISHER_YATES_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void partial_fisher_yates(int* out, int n, int max_i, int m,
                          unsigned long long seed) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    // Unique per-thread xorshift128+ state
    unsigned long long s0 = seed ^ (unsigned long long)row * 6364136223846793005ULL;
    unsigned long long s1 = s0 ^ 0xdeadbeefcafeULL;
    // warm up
    s0 ^= s0 << 23; s0 ^= s0 >> 17; s0 ^= s1 ^ (s1 >> 26);
    s1 ^= s1 << 23;

    // Stack-allocated swap buffer — holds virtual positions of already-swapped
    // elements so we never materialise the full [0, max_i) array.
    // m <= 100 is guaranteed by the caller.
    int buf[100];
    for (int i = 0; i < m; i++) buf[i] = -1;  // -1 = "not yet touched"

    for (int i = 0; i < m; i++) {
        // xorshift128+ step
        unsigned long long t = s0;
        s0 = s1;
        t ^= t << 23;
        t ^= t >> 17;
        t ^= s0 ^ (s0 >> 26);
        s1 = t;
        unsigned long long r = s0 + s1;

        // Draw j uniformly from [i, max_i)
        int j = i + (int)(r % (unsigned long long)(max_i - i));

        // Resolve virtual value at position j:
        // if buf[j] has been set we use it, otherwise the virtual value is j.
        // Since buf only tracks indices 0..m-1 we check the range.
        int val_j = (j < m && buf[j] != -1) ? buf[j] : j;
        int val_i = (buf[i] != -1) ? buf[i] : i;

        // Perform the virtual swap: store val_j at position i (our draw),
        // store val_i at position j so future iterations can find it.
        buf[i] = val_j;
        if (j < m) buf[j] = val_i;

        out[row * m + i] = val_j;
    }
}
''', 'partial_fisher_yates')

_THREADS_PER_BLOCK = 256


def _rand_indices_gpu(max_i: int, n: int, m: int, seed: int = 0) -> cp.ndarray:
    """
    Returns a (n, m) CuPy int32 array of unique random indices per row,
    drawn without replacement from [0, max_i).

    Replaces __rand_indices_noduplicates_on_rows entirely — no CPU work,
    no PCIe transfer, O(n*m) VRAM.
    """
    assert m <= 100, "Fisher-Yates kernel supports m <= 100 only"
    out = cp.empty((n, m), dtype=cp.int32)
    blocks = (n + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    _FISHER_YATES_KERNEL(
        (blocks,), (_THREADS_PER_BLOCK,),
        (out, np.int32(n), np.int32(max_i), np.int32(m),
         np.uint64(seed))
    )
    return out


# ---------------------------------------------------------------------------
# Vectorized duplicate mask — replaces Numba row_wise_duplicate_indices
# ---------------------------------------------------------------------------

def _row_wise_duplicate_mask(ar: cp.ndarray) -> cp.ndarray:
    """
    Returns a boolean mask (same shape as ar): True where an entry is either
    a duplicate of the previous entry in the same row (ar is pre-sorted) or
    equals the row's own index (point is its own neighbour).

    Replaces the @nb.njit row_wise_duplicate_indices — fully vectorized on GPU,
    no CPU round-trip.
    """
    n = ar.shape[0]
    row_idx = cp.arange(n, dtype=ar.dtype)[:, None]   # (N, 1)
    is_self = (ar == row_idx)

    # Adjacent duplicates in the sorted array: compare col j with col j-1.
    # Pad with a sentinel column of -1 on the left so column 0 is never flagged.
    sentinel = cp.full((n, 1), -1, dtype=ar.dtype)
    shifted = cp.concatenate([sentinel, ar[:, :-1]], axis=1)  # (N, m)
    is_dup = (ar == shifted)

    return is_self | is_dup


# ---------------------------------------------------------------------------
# Internal helpers (GPU versions)
# ---------------------------------------------------------------------------

def _sort_neighbors(data: cp.ndarray, neighbors: cp.ndarray) -> cp.ndarray:
    """Sort each row of neighbors by ascending hi-dim distance (unused by default)."""
    neighbor_points_hi = data[neighbors]
    diff = neighbor_points_hi - data[:, None, :]
    dists_squared = (diff ** 2).sum(axis=-1)
    sorting = cp.argsort(dists_squared, axis=1)
    neighbors[:, :] = cp.take_along_axis(neighbors, sorting, axis=1)
    return neighbors


def _update_neighbors(curr_neighbors: cp.ndarray,
                      new_randoms: cp.ndarray,
                      positions: cp.ndarray,
                      neighbor_positions: cp.ndarray,
                      k: int,
                      iter_seed: int = 0) -> cp.ndarray:
    """
    Refresh the latter half of the neighbor set with new random candidates,
    de-duplicate, then re-sort by hi-dim distance so the k nearest stay first.

    Changes vs. original:
    - row_wise_duplicate_indices (Numba CPU) replaced by _row_wise_duplicate_mask (GPU).
    - np.argwhere result (diff_near) was computed but never used — dropped.
    """
    curr_neighbors[:, k:] = new_randoms
    index_order = cp.argsort(curr_neighbors, axis=1)
    curr_neighbors[:, :] = cp.take_along_axis(curr_neighbors, index_order, axis=1)

    # Mark duplicates and self-references as infinitely far away
    dup_mask = _row_wise_duplicate_mask(curr_neighbors)
    dists_sq = ((neighbor_positions[curr_neighbors] - positions[:, None, :]) ** 2).sum(axis=-1)
    dists_sq[dup_mask] = 1e16

    # Sort by distance so the k best neighbours end up in the first k columns
    order = cp.argsort(dists_sq, axis=1)
    curr_neighbors[:, :] = cp.take_along_axis(curr_neighbors, order, axis=1)
    return curr_neighbors


def _smooth_stress(stresses: list) -> float:
    """Rolling mean over the last 32 stress values. Kept on CPU — tiny array."""
    width = 32
    if len(stresses) < width:
        return float('inf')
    return float(np.mean(stresses[-width:]))


def _layout(data: cp.ndarray,
            embedding: cp.ndarray,
            forces: cp.ndarray,
            neighbors: cp.ndarray,
            start: int = 0,
            end: int = None,
            alpha: float = 1.0):
    """Thin wrapper kept for API compatibility."""
    if end is None:
        end = data.shape[0]
    return _compute_forces_and_layout(data, embedding, forces, neighbors, start, end, alpha)


def _compute_forces_and_layout(data: cp.ndarray,
                                embedding: cp.ndarray,
                                forces: cp.ndarray,
                                neighbors: cp.ndarray,
                                start: int,
                                end: int,
                                alpha: float = 1.0):
    """
    Core force + gradient step — drop-in CuPy replacement.
    All tensor operations run on the GPU unchanged from the NumPy original.
    """
    k_neighbors = neighbors.shape[1]
    normalize_factor = 1.0 / k_neighbors

    neighbor_points_hi = data[neighbors]
    neighbor_points_lo = embedding[neighbors]

    diff  = neighbor_points_hi - data[:, None, :]
    delta = neighbor_points_lo - embedding[:, None, :]

    dists_hi = cp.sqrt((diff  ** 2).sum(axis=-1))
    dists_lo = cp.sqrt((delta ** 2).sum(axis=-1)) + 1e-8

    stress = float(((dists_hi - dists_lo) ** 2).sum())   # pull scalar to CPU immediately

    scalings = cp.expand_dims(1 - dists_hi / dists_lo, axis=-1)
    delta, scalings = cp.broadcast_arrays(delta, scalings)

    force_update = (delta * scalings).sum(axis=1) * normalize_factor
    forces_new   = forces * 0.5 + force_update

    embedding[start:end] += forces_new[start:end] * alpha
    return embedding, forces_new, stress


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def execute_glimmer_gpu(
        data: np.ndarray,
        initialization: np.ndarray = None,
        target_dim: int = None,
        decimation_factor: int = 2,
        neighbor_set_size: int = 8,
        max_iter: int = 512,
        min_level_size: int = 1000,
        rng=None,
        callback=None,
        verbose: bool = True,
        stress_ratio_tol: float = 1 - 1e-5,
        alpha: float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    Execute the Glimmer algorithm with GPU acceleration via CuPy.

    Parameters
    ----------
    data : np.ndarray
        High-dimensional data set (2-D, CPU array).  Moved to GPU internally.
    initialization : np.ndarray, optional
        Initial low-dimensional embedding (2-D, CPU array).
    target_dim : int, optional
        Embedding dimensionality.  Only used when initialization is None.
    decimation_factor : int
        Coarsening factor between levels.
    neighbor_set_size : int
        Number of near/far neighbours per point (effective = neighbor_set_size * 2).
        Must satisfy neighbor_set_size * 3 <= 100 (Fisher-Yates kernel limit).
    max_iter : int
        Maximum iterations per level.
    min_level_size : int
        Minimum points in the smallest level.
    rng : np.random.Generator, optional
        CPU RNG used only for the initial permutation and embedding noise.
    callback : function(dict)
        [optional] callback function which will be called in each iteration of the algorithm.
        The function argument is a dictionary containing several internal variables, i.e.,
        embedding, forces, current level, current iteration, index set of the current level, stress, smoothed stress.
        The objects embedding, forces, and index_set are callables that return CPU arrays when invoked,
        e.g. emb = dict['embedding']() to avoid unnecessary GPU->CPU transfers when not accessed in callback.
    verbose : bool
        Print level/iteration info.
    stress_ratio_tol : float
        Early-stopping threshold on smoothed stress improvement.
    alpha : float
        Learning rate.

    Returns
    -------
    tuple[np.ndarray, float]
        Low-dimensional embedding (CPU ndarray) and smoothed stress.
    """
    assert neighbor_set_size * 3 <= 100, (
        f"neighbor_set_size={neighbor_set_size} requires neighbor_set_size*3="
        f"{neighbor_set_size*3} <= 100 for the Fisher-Yates GPU kernel."
    )

    # --- CPU setup (rng, level sizes, initialization) ----------------------
    if rng is None:
        rng = np.random.default_rng()

    if initialization is None:
        if target_dim is None:
            target_dim = 2
        norms = np.linalg.norm(data, axis=1)
        initialization = rng.random((data.shape[0], target_dim)) - 0.5
        initialization *= (norms / np.linalg.norm(initialization, axis=1))[:, None]

    _no_callback = callback is None
    if callback is None:
        callback = lambda *args: None

    if target_dim and initialization.shape[1] != target_dim:
        import warnings
        warnings.warn(
            f"provided target dimension {target_dim} does not match "
            f"initialization shape[1]={initialization.shape[1]}"
        )

    if initialization.shape[0] != data.shape[0]:
        raise ValueError(
            f"initialization shape[0]={initialization.shape[0]} does not "
            f"match data shape[0]={data.shape[0]}"
        )

    n = data.shape[0]

    # Randomised row order — done on CPU, then moved to GPU as int32
    rand_indices_cpu = rng.permutation(n).astype(np.int32)

    # Level sizes (CPU — just a short list)
    level_sizes = [n]
    n_levels = 0
    while level_sizes[n_levels] >= min_level_size * decimation_factor:
        level_sizes.append(level_sizes[n_levels] // decimation_factor)
        n_levels += 1
    n_levels += 1

    if verbose:
        print(f"levels: {n_levels}, level sizes: {level_sizes[::-1]}")

    # --- Move everything to GPU --------------------------------------------
    data_gpu        = cp.asarray(data,           dtype=cp.float32)
    embedding_gpu   = cp.asarray(initialization, dtype=cp.float32)
    forces_gpu      = cp.zeros_like(embedding_gpu)
    rand_indices    = cp.asarray(rand_indices_cpu, dtype=cp.int32)
    neighbors_gpu   = cp.zeros((n, neighbor_set_size * 3), dtype=cp.int32)

    # Seed counter for the Fisher-Yates kernel — incremented each call so
    # successive calls produce independent random draws.
    _kernel_seed = 0

    def _next_rand_indices(max_i: int, rows: int, m: int) -> cp.ndarray:
        nonlocal _kernel_seed
        result = _rand_indices_gpu(max_i, rows, m, seed=_kernel_seed)
        _kernel_seed += rows   # advance seed so next call differs
        return result

    # --- Main multi-level loop ---------------------------------------------
    sm_stress = None

    for level in range(n_levels - 1, -1, -1):
        current_n = level_sizes[level]
        if verbose:
            print(f"execution on level: {level}, current n: {current_n}")

        current_index_set = rand_indices[:current_n]   # view, stays on GPU

        # Initialise neighbour sets at the coarsest level
        if level == n_levels - 1:
            neighbors_gpu[current_index_set] = _next_rand_indices(
                current_n, current_n, neighbor_set_size * 3
            )

        # Slices for this level — views into the full GPU arrays
        current_data      = data_gpu[current_index_set]
        current_embedding = embedding_gpu[current_index_set]
        current_forces    = forces_gpu[current_index_set]
        current_neighbors = neighbors_gpu[current_index_set]

        stresses: list[float] = []
        sm_stress_prev = float('inf')

        for it in range(max_iter):
            current_embedding, current_forces, stress = _layout(
                current_data,
                current_embedding,
                current_forces,
                current_neighbors[:, :neighbor_set_size * 2],
                alpha=alpha,
            )
            current_embedding -= current_embedding.mean(axis=0)

            # Write views back into the full arrays
            embedding_gpu[current_index_set] = current_embedding
            forces_gpu[current_index_set]    = current_forces

            # Refresh random neighbour candidates (on GPU, no transfer)
            new_neighbor_candidates = _next_rand_indices(
                current_n, current_n, neighbor_set_size * 2
            )
            _update_neighbors(
                current_neighbors, new_neighbor_candidates,
                current_data, current_data, neighbor_set_size,
            )

            # stress is already a Python float (extracted in _compute_forces_and_layout)
            stresses.append(stress / current_n)
            sm_stress = _smooth_stress(stresses)

            # Callback — transfer GPU arrays to CPU only when a real callback was provided,
            # avoiding unnecessary PCIe transfers in the common no-callback case.
            if not _no_callback:
                def get_embedding() -> np.ndarray:
                    return cp.asnumpy(embedding_gpu)
                def get_forces() -> np.ndarray:
                    return cp.asnumpy(forces_gpu)
                def get_index_set() -> np.ndarray:
                    return cp.asnumpy(current_index_set)
                
                callback(dict(
                    embedding=get_embedding,
                    forces=get_forces,
                    level=level,
                    iter=it,
                    index_set=get_index_set,
                    smoothed_stress=sm_stress,
                    stress=stresses[-1],
                ))

            if verbose and it % 10 == 0:
                print(f"stress after iteration {it}: {stresses[-1]:.6f}  "
                      f"smoothed stress: {sm_stress}")

            if sm_stress_prev < float('inf'):
                stress_ratio = sm_stress / sm_stress_prev
                if 1.0 >= stress_ratio > stress_ratio_tol:
                    if verbose:
                        print(f"early termination of level {level} after {it} iterations")
                    break

            sm_stress_prev = sm_stress

        if level > 0:
            next_n = level_sizes[level - 1]
            next_index_set = rand_indices[:next_n]

            # Initialise neighbours for the new points joining at the next level
            neighbors_gpu[next_index_set[current_n:next_n]] = _next_rand_indices(
                current_n, next_n - current_n, neighbor_set_size * 3
            )

            # Relaxation: settle new points while keeping existing ones fixed
            for _ in range(8):
                embedding_gpu[next_index_set], _, _ = _layout(
                    data_gpu[next_index_set],
                    embedding_gpu[next_index_set],
                    forces_gpu[next_index_set],
                    neighbors_gpu[next_index_set, :neighbor_set_size * 2],
                )

    # Transfer final embedding back to CPU
    return cp.asnumpy(embedding_gpu), sm_stress
