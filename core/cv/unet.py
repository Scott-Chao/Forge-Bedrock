"""
core/cv/unet.py — U-Net encoder-decoder building blocks.

U-Net (Ronneberger et al., 2015) is a symmetric encoder-decoder
architecture designed for biomedical image segmentation.  Its key
innovation is **skip connections** that concatenate encoder feature maps
to the corresponding decoder layers, preserving fine spatial detail
that would otherwise be lost during downsampling.

DownBlock
    One encoder stage: residual block + max-pool downsampling.  The
    pre-pool feature map is saved and returned for the skip connection.

UpBlock
    One decoder stage: transposed-conv upsampling → skip concatenation
    → residual block.

UNet
    Full U-shape: encoder (4x DownBlock) → bottleneck → decoder (4x
    UpBlock) → 1×1 output convolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv2d import Conv2d
from .conv_transpose import ConvTranspose2d
from .normalization import BatchNorm2d
from .pooling import MaxPool2d

# ============================================================================
# DownBlock — one encoder stage
# ============================================================================


class _DoubleConv(nn.Module):
    """Two consecutive Conv2d → BN → ReLU blocks.

    This is the core convolutional unit used throughout U-Net.  It
    extracts features at a fixed spatial resolution.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (same for both conv layers).
    mid_channels : int | None, default=None
        Channels of the middle hidden representation.  If None, defaults
        to ``out_channels``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
    ) -> None:
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            Conv2d(in_channels, mid_channels, 3, stride=1, padding=1, bias=False),
            BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            Conv2d(mid_channels, out_channels, 3, stride=1, padding=1, bias=False),
            BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply double convolution."""
        return self.conv(x)


class DownBlock(nn.Module):
    """One encoder stage: feature extraction → max-pool downsampling.

    The forward pass returns **two** tensors:

        1. ``out``  — the downsampled feature map  (N, C_out, H/2, W/2)
        2. ``skip`` — the feature before pooling   (N, C_out, H, W)

    ``skip`` is passed to the corresponding decoder stage via a
    skip connection (concatenation along the channel dimension).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (after the residual block).
    use_residual : bool, default=True
        If True, add a skip connection around the double conv
        (i.e., ``relu(double_conv(x) + x)``).  Without the residual,
        this becomes the classic U-Net "two convs → maxpool" pattern.

    Shape
    -----
    Input:  (N, C_in, H, W)
    Output: (N, C_out, H/2, W/2), (N, C_out, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        self.use_residual = use_residual

        self.double_conv = _DoubleConv(in_channels, out_channels)
        self.pool = MaxPool2d(kernel_size=2, stride=2)

        if self.use_residual and in_channels != out_channels:
            self.skip_proj = Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        out : torch.Tensor
            Downsampled feature map, shape (N, C_out, H/2, W/2).
        skip : torch.Tensor
            Feature map before pooling, shape (N, C_out, H, W).
            Used for the decoder skip connection.
        """
        h = self.double_conv(x)
        if self.use_residual:
            h = torch.relu(h + self.skip_proj(x))
        else:
            h = torch.relu(h)
        return self.pool(h), h

    def extra_repr(self) -> str:
        """Return a formatted string for ``print(down_block)``."""
        return f"residual={self.use_residual}"


# ============================================================================
# UpBlock — one decoder stage
# ============================================================================


class UpBlock(nn.Module):
    """One decoder stage: transposed-conv upsample → skip concat → double conv.

    Takes a low-resolution feature map from the previous decoder level
    (or the bottleneck), upsamples it 2×, concatenates the corresponding
    encoder's skip connection feature along the channel dimension, then
    applies a double convolution to fuse the information.

    Parameters
    ----------
    in_channels : int
        Number of channels of the input from the previous decoder level
        (or bottleneck).
    out_channels : int
        Target number of output channels.  **Must** equal the number of
        channels of the matching encoder skip connection for a symmetric
        U-Net.

    Shape
    -----
    Input:
        x : (N, in_channels, H, W)  — decoder input (from below)
        skip : (N, out_channels, 2H, 2W) — encoder skip connection

    Output:
        (N, out_channels, 2H, 2W)

        Note: ``skip_channels == out_channels`` for a symmetric U-Net.
        If the spatial sizes of ``x_up`` and ``skip`` differ slightly
        (due to odd input dimensions), the code should centre-crop or
        pad ``skip`` before concatenation.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.up_conv = ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = _DoubleConv(2 * out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Decoder input (from the previous level or bottleneck).
            Shape: (N, in_channels, H, W).
        skip : torch.Tensor
            Encoder skip connection feature.
            Shape: (N, out_channels, 2H, 2W).

        Returns
        -------
        torch.Tensor
            Output feature map of shape (N, out_channels, 2H, 2W).
        """
        x = self.up_conv(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y or diff_x:
            skip = skip[
                :,
                :,
                diff_y // 2 : skip.size(2) - (diff_y - diff_y // 2),
                diff_x // 2 : skip.size(3) - (diff_x - diff_x // 2),
            ]
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x
