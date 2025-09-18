"""
CUDA extensions for the radioft package.
"""

import torch

try:
    if torch.cuda.is_available():
        from . import kernels  # noqa: F401

        __all__ = ["kernels"]

    else:
        __all__ = []
except ImportError:
    __all__ = []


def is_available():
    """Check if CUDA extensions are available."""
    return len(__all__) > 0
