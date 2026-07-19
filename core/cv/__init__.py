"""
core/cv — Computer Vision modules (Phase 7).

Builds on PyTorch (Phase 5+ convention) to implement vision primitives
from first principles.

Modules
-------
Conv2d
    2D convolution via the im2col + GEMM approach.
MaxPool2d
    2D max-pooling via unfold + reduction.
AvgPool2d
    2D average-pooling via unfold + reduction.
"""

from .conv2d import Conv2d
from .pooling import AvgPool2d, MaxPool2d

__all__ = [
    "AvgPool2d",
    "Conv2d",
    "MaxPool2d",
]
