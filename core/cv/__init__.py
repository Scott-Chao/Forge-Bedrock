"""
core/cv — Computer Vision modules (Phase 7).

Builds on PyTorch (Phase 5+ convention) to implement vision primitives
from first principles.

Modules
-------
Conv2d
    2D convolution via the im2col + GEMM approach.
"""

from .conv2d import Conv2d

__all__ = [
    "Conv2d",
]
