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
from .pooling import MaxPool2d


class _ResBlock(nn.Module):
    """Abstract base shared by BasicBlock and BottleneckBlock.

    Handles the skip-connection logic (1×1 projection when dimensions
    change).  Subclasses implement ``_make_layers()`` which returns
    the sequential trunk.
    """

    expansion: int = 1
    """Output / bottleneck width ratio.  Subclasses override:
    BasicBlock → 1, BottleneckBlock → 4."""

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


# ============================================================================
# ResNet — network assembly
# ============================================================================


# ── Pre-defined configurations ──────────────────────────────────────────────
_RESNET_CONFIGS: dict[int, tuple[type[_ResBlock], list[int]]] = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (BottleneckBlock, [3, 4, 6, 3]),
    101: (BottleneckBlock, [3, 4, 23, 3]),
    152: (BottleneckBlock, [3, 8, 36, 3]),
}


class ResNet(nn.Module):
    """Residual Network with configurable depth.

    Parameters
    ----------
    depth : int, default=18
        ResNet variant.  Supported: ``{18, 34, 50, 101, 152}``.
    num_classes : int, default=1000
        Number of output classes.
    zero_init_residual : bool, default=False
        If True, initialise the last BN weight of each block to 0.
        This makes the residual branch start as identity, improving
        training stability at initialisation (Goyal et al. 2017,
        "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour").
    """

    def __init__(
        self,
        depth: int = 18,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()

        # ── Resolve block type & layer config ──────────────────────
        if depth not in _RESNET_CONFIGS:
            raise ValueError(f"Unsupported depth: {depth}")
        self.depth = depth
        block_type, num_blocks = _RESNET_CONFIGS[depth]

        # ── Stem ───────────────────────────────────────────────────
        self.stem = nn.Sequential(
            Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
            MaxPool2d(3, stride=2, padding=1),
        )

        # ── Stages 1–4 ─────────────────────────────────────────────
        self.inplanes = 64
        self.layer1 = self._make_layer(block_type, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block_type, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_type, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_type, 512, num_blocks[3], stride=2)

        # ── Head ───────────────────────────────────────────────────
        last_channels = 512 * block_type.expansion
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(last_channels, num_classes),
        )

        # ── Initialisation ─────────────────────────────────────────
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.trunk[4].weight)
                elif isinstance(m, BottleneckBlock):
                    nn.init.zeros_(m.trunk[7].weight)

    def _make_layer(
        self,
        block_type: type[_ResBlock],
        planes: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Build one ResNet stage.

        Parameters
        ----------
        block_type : type[_ResBlock]
            Which residual block to use (BasicBlock or BottleneckBlock).
        planes : int
            Bottleneck width (NOT the output channels).  Output channels
            = ``planes * block_type.expansion``.
        num_blocks : int
            Number of residual blocks in this stage.
        stride : int
            Stride of the **first** block in the stage.  Use 2 for
            downsampling stages (layer2–4), 1 for layer1.

        Returns
        -------
        nn.Sequential
        """
        layers = []
        layers.append(block_type(self.inplanes, planes * block_type.expansion, stride))
        self.inplanes = planes * block_type.expansion
        for _ in range(1, num_blocks):
            layers.append(block_type(self.inplanes, self.inplanes, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (N, 3, 224, 224) or similar.

        Returns
        -------
        torch.Tensor, shape (N, num_classes)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x

    def extra_repr(self) -> str:
        """Return a formatted string for ``print(resnet)``."""
        total = sum(p.numel() for p in self.parameters())
        return f"ResNet-{self.depth} ({total / 1e6:.1f}M params)"
