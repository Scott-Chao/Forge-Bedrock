"""
core/cv/resnet.py — Residual Network building blocks.

ResNet introduces the **skip connection** (residual connection), allowing
very deep networks to train without degradation:

    y = F(x) + x

where F(x) is a small stack of Conv2d → BN → ReLU layers.  If the
dimensions of F(x) and x don't match, the skip path applies a 1×1 conv
to project x to the correct shape.

Two block types are defined here:

    BasicBlock
        Two 3×3 conv layers — used in ResNet-18, ResNet-34.

    BottleneckBlock
        Three layers: 1×1 (reduce) → 3×3 (spatial) → 1×1 (expand).
        Used in ResNet-50, ResNet-101, ResNet-152.  The 1×1 convs
        control the channel count so the expensive 3×3 operates on a
        smaller ``C_mid = C_out // 4``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv2d import Conv2d
from .normalization import BatchNorm2d


class _ResBlock(nn.Module):
    """Abstract base shared by BasicBlock and BottleneckBlock.

    Handles the skip-connection logic (1×1 projection when dimensions
    change).  Subclasses implement ``_make_layers()`` which returns
    the sequential trunk.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()

        self.trunk = self._make_layers(in_channels, out_channels, stride)

        if stride != 1 or in_channels != out_channels:
            self.skip = Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.skip = nn.Identity()

    def _make_layers(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int],
    ) -> nn.Sequential:
        """Build the trunk layers.  Override in subclasses."""
        raise NotImplementedError("Override in subclass")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: F(x) + x (with optional skip projection)."""
        identity = self.skip(x)
        residual = self.trunk(x)
        out = identity + residual
        return torch.relu(out)


class BasicBlock(_ResBlock):
    """Two 3×3 conv layers with BatchNorm + ReLU.

    Channel flow: C_in → C_out (first conv may use stride > 1).

    The second conv always uses ``stride=1`` so the spatial size
    only changes on the first conv.
    """

    expansion: int = 1
    """Factor relating ``out_channels`` to the block's internal width.
    For BasicBlock, output channels = internal width (expansion=1)."""

    def _make_layers(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int],
    ) -> nn.Sequential:
        """Build two conv layers.
        .. caution::
            ``bias=False`` is used because BatchNorm has its own bias
            parameter (β).  Having both biases is redundant.
        """
        return nn.Sequential(
            Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            BatchNorm2d(out_channels),
        )


class BottleneckBlock(_ResBlock):
    """Three-layer bottleneck: 1×1 → 3×3 → 1×1.

    Channel flow: C_in → C_mid → C_mid → C_out
    where ``C_mid = C_out // self.expansion``.

    The 1×1 reduce / expand phases are computationally cheap; the
    expensive 3×3 operates on the reduced channel count.
    """

    expansion: int = 4
    """C_out = expansion * (bottleneck width).  The middle 3×3 conv
    operates on ``C_out // expansion`` channels."""

    def _make_layers(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int],
    ) -> nn.Sequential:
        """Build three conv layers.

        Note that the first 1×1 always has ``stride=1``; the spatial
        downsampling happens only in the 3×3 conv (Layer 2).
        """
        mid_channels = out_channels // self.expansion
        return nn.Sequential(
            Conv2d(in_channels, mid_channels, 1, stride=1, bias=False),
            BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False),
            BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            Conv2d(mid_channels, out_channels, 1, stride=1, bias=False),
            BatchNorm2d(out_channels),
        )
