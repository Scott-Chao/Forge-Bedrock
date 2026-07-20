"""
core/cv — Computer Vision modules (Phase 7).

Builds on PyTorch (Phase 5+ convention) to implement vision primitives
from first principles.

Modules
-------
Conv2d
    2D convolution via the im2col + GEMM approach.
ConvTranspose2d
    2D transposed convolution via matmul + fold (col2im).
MaxPool2d
    2D max-pooling via unfold + reduction.
AvgPool2d
    2D average-pooling via unfold + reduction.
BatchNorm2d
    2D batch normalisation over (N, H, W) per channel.
"""

from .conv2d import Conv2d
from .conv_transpose import ConvTranspose2d
from .normalization import BatchNorm2d
from .pooling import AvgPool2d, MaxPool2d

__all__ = [
    "AvgPool2d",
    "BatchNorm2d",
    "Conv2d",
    "ConvTranspose2d",
    "MaxPool2d",
]
