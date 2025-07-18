"""
radioft - Fourier transform implementations for radio interferometry applications.
"""

from . import dft, utils
from .version import __version__

try:
    import torch

    if torch.cuda.is_available():
        from .cuda import kernels  # NOQA

        _has_cuda_extension = True
    else:
        _has_cuda_extension = False
except ImportError:
    _has_cuda_extension = False


def has_cuda_extension():
    """Check if CUDA extension is available."""
    return _has_cuda_extension


__all__ = ["__version__", "dft", "utils"]

if _has_cuda_extension:
    from . import cuda  # NOQA

    __all__.append("cuda")
