"""
radioft - Fourier transform implementations for radio interferometry applications.
"""

from .version import __version__
from . import dft
from . import utils


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
