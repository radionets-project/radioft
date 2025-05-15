"""
radioft - Fourier transform implementations for radio interferometry applications.
"""

__version__ = "0.1.0"

import os
import torch

# Try to import CUDA extension, but make package still usable without it
try:
    if torch.cuda.is_available():
        from .cuda import kernels  # Import your CUDA kernels
        _has_cuda_extension = True
    else:
        _has_cuda_extension = False
except ImportError:
    _has_cuda_extension = False

def has_cuda_extension():
    """Check if CUDA extension is available."""
    return _has_cuda_extension

# Import main modules
from . import dft
from . import utils

# Only import CUDA module if available
if _has_cuda_extension:
    from . import cuda
