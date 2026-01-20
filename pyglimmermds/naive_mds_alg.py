import numpy as np
import math
import numba as nb
from numba import cuda
from typing import Callable
from sklearn import metrics


def euqal_weights(*_) -> float:
    return 1.0


@nb.njit(cache=True, parallel=True)
def pairwise_distances_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute distances between block a and all points in b"""
    dists = np.zeros((a.shape[0], b.shape[0]))
    for i in nb.prange(a.shape[0]):
        a_i, b_ = np.broadcast_arrays(a[i], b)
        d = a_i - b_
        dists[i,:] = np.sqrt((d*d).sum(axis=1))
        #for j in range(b.shape[0]):
        #    d = a[i] - b[j]
        #    dists[i, j] = np.sqrt((d*d).sum())
    return dists


@nb.njit(cache=True, fastmath=True, parallel=True)
def gradient_block_no_weights(
    x_block: np.ndarray,      # High-dim data for current block
    x_all: np.ndarray,         # All high-dim data
    y_block: np.ndarray,       # Low-dim positions for current block
    y_all: np.ndarray,         # All low-dim positions
    block_indices: np.ndarray  # Indices of points in current block
) -> np.ndarray:
    """
    Compute gradient for a block of points against all other points.
    Optimized version without weights.
    
    Returns gradient array of shape (block_size, low_dim)
    """
    eps = 1e-8
    block_size = x_block.shape[0]
    n_total = x_all.shape[0]
    g_block = np.zeros(y_block.shape)
    
    # Compute high-dim distances for this block against all points
    pd_block = pairwise_distances_block(x_block, x_all)
    #pd_block = metrics.pairwise_distances(x_block, x_all)

    for i in nb.prange(block_size):
        global_i = block_indices[i]
        gi = y_block[i] * 0  # zeros in correct shape
        
        for j in range(n_total):
            if global_i == j:
                continue
            
            # Low-dim distance
            d_low = y_block[i] - y_all[j]
            dist_low = math.sqrt((d_low * d_low).sum())
            
            # High-dim distance (already computed)
            dist_high = pd_block[i, j]
            
            # Gradient contribution (weight = 1.0 implicit)
            gi += 2 * d_low * (1 - (dist_high / (dist_low + eps)))
        
        g_block[i] = gi
    
    return g_block


@nb.njit(cache=True, fastmath=True, parallel=True)
def gradient_block_with_weights(
    x_block: np.ndarray,
    x_all: np.ndarray,
    y_block: np.ndarray,
    y_all: np.ndarray,
    block_indices: np.ndarray,
    weights_block: np.ndarray  # Precomputed weights for this block (block_size x n)
) -> np.ndarray:
    """
    Compute gradient with precomputed weights for this block.
    Now JIT-compilable since weights are passed as array.
    """
    eps = 1e-8
    block_size = x_block.shape[0]
    n_total = x_all.shape[0]
    g_block = np.zeros(y_block.shape)
    
    # Compute high-dim distances for this block against all points
    pd_block = pairwise_distances_block(x_block, x_all)
    
    for i in nb.prange(block_size):
        global_i = block_indices[i]
        gi = y_block[i] * 0
        
        for j in range(n_total):
            if global_i == j:
                continue
            
            w = weights_block[i, j]
            if w == 0.0:
                continue
            
            # Low-dim distance
            d_low = y_block[i] - y_all[j]
            dist_low = math.sqrt((d_low * d_low).sum())
            
            # High-dim distance
            dist_high = pd_block[i, j]
            
            # Gradient contribution
            gi += 2 * d_low * (w * (1 - (dist_high / (dist_low + eps))))
        
        g_block[i] = gi
    
    return g_block


def execute_mds_minibatch(
    x: np.ndarray,
    y_init: np.ndarray,
    weights: Callable[[int, int], float] = None,
    iterations: int = 100,
    a: float = 0.001,
    block_size: int = 128
) -> np.ndarray:
    """
    Run MDS with mini-batch gradient computation.
    
    Parameters:
    -----------
    x : np.ndarray, shape (n, high_dim)
        High-dimensional data
    y_init : np.ndarray, shape (n, low_dim)
        Initial low-dimensional positions
    weights : Callable[[int, int], float] or None
        Optional weight function. If None, uses uniform weights (faster).
        Function should take two indices (i, j) and return weight value.
    iterations : int
        Number of gradient descent iterations
    a : float
        Learning rate
    block_size : int
        Number of points to process in each block
    
    Returns:
    --------
    y : np.ndarray, shape (n, low_dim)
        Final low-dimensional positions
    """
    n = x.shape[0]
    y = y_init.copy()
    
    # Determine number of blocks
    n_blocks = int(np.ceil(n / block_size))
    
    # Choose gradient computation path
    use_weights = weights is not None
    
    for iteration in range(iterations):
        # Accumulate gradients for all points
        g_full = np.zeros_like(y)
        
        # Process each block
        for block_idx in range(n_blocks):
            start_idx = block_idx * block_size
            end_idx = min(start_idx + block_size, n)
            block_indices = np.arange(start_idx, end_idx)
            
            # Extract block data
            x_block = x[start_idx:end_idx]
            y_block = y[start_idx:end_idx]
            
            # Compute gradient for this block
            if use_weights:
                # Compute weights for this block only (block_size x n)
                weights_block = np.zeros((end_idx - start_idx, n))
                for i in range(end_idx - start_idx):
                    for j in range(n):
                        weights_block[i, j] = weights(start_idx + i, j)
                
                g_block = gradient_block_with_weights(
                    x_block, x, y_block, y, block_indices, weights_block
                )
            else:
                g_block = gradient_block_no_weights(
                    x_block, x, y_block, y, block_indices
                )
            
            # Store in full gradient array
            g_full[start_idx:end_idx] = g_block
        
        # Update all positions at once
        y = y - a * g_full
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}/{iterations}")
    
    return y


@nb.njit(cache=True)
def stress_minibatch(x: np.ndarray, y: np.ndarray, block_size: int = 128) -> float:
    """
    Compute stress in a memory-efficient way using mini-batches.
    Assumes uniform weights (weight = 1.0 for all pairs).
    Vectorized for better performance.
    """
    n = x.shape[0]
    n_blocks = int(np.ceil(n / block_size))
    total_stress = 0.0
    
    for block_idx in range(n_blocks):
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, n)
        
        x_block = x[start_idx:end_idx]
        y_block = y[start_idx:end_idx]
        
        # Compute high-dim distances for this block against all points
        pd_block = pairwise_distances_block(x_block, x)
        
        # Compute low-dim distances for this block against all points
        dist_low_block = pairwise_distances_block(y_block, y)
        
        # Compute stress contributions (vectorized)
        # Diagonal elements are already 0 by definition (distance to self = 0)
        stress_contributions = (pd_block - dist_low_block) ** 2  # (block_size, n)
        
        # Sum all contributions
        total_stress += stress_contributions.sum()
    
    return total_stress / 2


if __name__ == '__main__':
    # Usage example
    hi_dim = 10
    n_points = 10000
    
    # Generate high dimensional data consisting of normally distributed clusters
    data = np.vstack([
        np.random.multivariate_normal(np.zeros(hi_dim)+i, np.eye(hi_dim)*0.1, size=n_points//4) 
        for i in range(4)
    ])
    
    print(f"Performing mini-batch MDS on data of shape {data.shape}")
    
    lo_dim = 2
    projection = np.random.rand(data.shape[0], lo_dim)  # random init
    
    # Run mini-batch MDS with block_size=200 (no weights = faster)
    projection = execute_mds_minibatch(
        data,
        y_init=projection,
        weights=None,  # No custom weights = uniform weights
        iterations=10,
        a=0.0001,
        block_size=256
    )
    
    # Compute final stress
    final_stress = stress_minibatch(data, projection, block_size=256)
    print(f"Final stress: {math.sqrt(final_stress)/data.shape[0]}")
    
    # Example with custom weights (slower):
    # def distance_weight(i, j):
    #     # Weight inversely proportional to distance
    #     return 1.0 / (1.0 + abs(i - j))
    # 
    # projection = run_mds_minibatch(
    #     data,
    #     y_init=projection,
    #     weights=distance_weight,
    #     iterations=500,
    #     a=0.0001,
    #     block_size=200
    # )
    
    # Plot the projection
    import matplotlib.pyplot as plt
    plt.scatter(projection[:,0], projection[:,1], alpha=0.5, s=1)
    plt.axis("equal")
    plt.title(f"Mini-batch MDS (n={data.shape[0]}, block_size=200)")
    plt.show()