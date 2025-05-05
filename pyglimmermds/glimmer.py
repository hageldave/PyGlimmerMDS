from . import execute_glimmer
import numpy as np

class Glimmer:
    """
    Class to perform multidimensional scaling using the Glimmer algorithm.

    Attributes
    ----------
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
    """

    def __init__(self,
        target_dim = 2,
        decimation_factor = 2,
        neighbor_set_size = 8,
        max_iter = 512,
        min_level_size = 1000,
        rng = None,
        callback = None,
        verbose = True,
        stress_ratio_tol = 1-1e-5
    ):
        self.target_dim = target_dim
        self.decimation_factor = decimation_factor
        self.neighbor_set_size = neighbor_set_size
        self.max_iter = max_iter
        self.min_level_size = min_level_size
        self.rng = rng
        self.callback = callback
        self.verbose = verbose
        self.stress_ratio_tol = stress_ratio_tol


    def fit_transform(self, data: np.ndarray, init: np.ndarray=None) -> np.ndarray:
        """
        Fits a low-dimensional embedding to the data.

        Parameters
        ----------
        data: np.ndarray
            the high-dimensional data set for which multidimensional scaling is performed. (2D array)
        init: np.ndarray
            [optional] initial low-dimensional embedding (2D array). If None, random initialization will be used.

        Returns
        -------
        np.ndarray
            the low-dimensional embedding (2D array)
        """
        return execute_glimmer(
            data,
            initialization=init,
            target_dim=self.target_dim,
            decimation_factor=self.decimation_factor,
            neighbor_set_size=self.neighbor_set_size,
            max_iter=self.max_iter,
            min_level_size=self.min_level_size,
            rng=self.rng,
            callback=self.callback,
            verbose=self.verbose,
            stress_ratio_tol=self.stress_ratio_tol
        )

