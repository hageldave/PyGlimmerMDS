from .glimmer_alg import execute_glimmer
from .glimmer_alg_pd import execute_glimmer_pd
try:
    from .glimmer_alg_gpu import execute_glimmer_gpu
    HAS_GPU_IMPL = True
except ImportError:
    HAS_GPU_IMPL = False
from .glimmer import Glimmer

