"""
CUDA extensions for the radioft package.
"""

import torch

# Try to import CUDA extension
try:
    if torch.cuda.is_available():
        from . import kernels  # This will be your compiled CUDA code

        # Export the functions your CUDA module provides
        # Replace these with the actual functions exposed by your CUDA code
        __all__ = ['kernels']

        # You can create wrapper functions here to provide a cleaner interface
        # For example:
        # def direct_dft_2d(input, u, v):
        #     return kernels.your_cuda_function(input, u, v)
    else:
        __all__ = []
except ImportError:
    __all__ = []

def is_available():
    """Check if CUDA extensions are available."""
    return len(__all__) > 0
