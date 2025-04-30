from glimmer_alg import execute_glimmer
import numpy as np

class Glimmer:

    def __init__(self,
        target_dim = 2,
        decimation_factor = 2,
        neighbor_set_size = 8,
        max_iter = 512,
        min_level_size = 1000,
        rng = None,
        callback = None,
        verbose = True
    ):
        self.target_dim = target_dim
        self.decimation_factor = decimation_factor
        self.neighbor_set_size = neighbor_set_size
        self.max_iter = max_iter
        self.min_level_size = min_level_size
        self.rng = rng
        self.callback = callback
        self.verbose = verbose


    def fit_transform(self, data: np.ndarray, init: np.ndarray=None) -> np.ndarray:
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
            verbose=self.verbose
        )

