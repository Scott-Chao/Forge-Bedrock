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
BasicBlock
    ResNet basic residual block: two 3×3 convs + skip connection.
BottleneckBlock
    ResNet bottleneck block: 1×1 → 3×3 → 1×1 + skip connection.
ResNet
    Configurable residual network (ResNet-18/34/50/101/152).
DownBlock
    U-Net encoder stage: double conv → max-pool downsampling, returns
    (downsampled, skip) for decoder skip connection.
UpBlock
    U-Net decoder stage: conv-transpose upsample → skip concat →
    double conv.
UNet
    Full U-shaped encoder-decoder with skip connections.
"""

from .conv2d import Conv2d
from .conv_transpose import ConvTranspose2d
from .normalization import BatchNorm2d
from .pooling import AvgPool2d, MaxPool2d
from .resnet import BasicBlock, BottleneckBlock, ResNet
from .unet import DownBlock, UNet, UpBlock

__all__ = [
    "AvgPool2d",
    "BasicBlock",
    "BatchNorm2d",
    "BottleneckBlock",
    "Conv2d",
    "ConvTranspose2d",
    "DownBlock",
    "MaxPool2d",
    "ResNet",
    "UNet",
    "UpBlock",
]
