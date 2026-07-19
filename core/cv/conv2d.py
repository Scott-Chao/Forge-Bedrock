"""
core/cv/conv2d.py — 2D Convolution layer via im2col + matrix multiply.

Implements the forward pass of a 2D convolution using the im2col
(image-to-column) trick:

    1.  Extract local patches from the input via unfold.
    2.  Reshape patches into a 2D matrix (one row per spatial location).
    3.  Perform a single matrix multiply (GEMM) with the flattened
        weight matrix to produce the output.
    4.  Reshape back to (N, C_out, H_out, W_out).

This avoids explicit nested loops over the spatial dimensions and
delegates the heavy lifting to highly optimised BLAS routines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Conv2d(nn.Module):
    """2D convolution on (N, C_in, H, W) tensors.

    The forward pass uses **im2col**: ``F.unfold`` extracts all local
    patches into a matrix, then a single ``matmul`` with the flattened
    weight produces the output.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (number of kernels).
    kernel_size : int | tuple[int, int]
        Spatial size of the convolution kernel.
    stride : int | tuple[int, int], default=1
        Stride of the convolution.
    padding : int | tuple[int, int], default=0
        Zero-padding added to both sides of each spatial axis.
    dilation : int | tuple[int, int], default=1
        Spacing between kernel elements (dilated / atrous convolution).
    bias : bool, default=True
        If True, adds a learnable bias to the output.
    device : torch.device | None, optional
        Device to place the parameters on.
    dtype : torch.dtype | None, optional
        Data type of the parameters.

    Shape
    -----
    Input:  (N, C_in, H_in, W_in)
    Output: (N, C_out, H_out, W_out)

    where

        H_out = floor((H_in + 2 * padding[0]
                       - dilation[0] * (kernel_size[0] - 1) - 1)
                      / stride[0] + 1)
        W_out = floor((W_in + 2 * padding[1]
                       - dilation[1] * (kernel_size[1] - 1) - 1)
                      / stride[1] + 1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        k_h, k_w = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
        p_h, p_w = (padding, padding) if isinstance(padding, int) else padding
        d_h, d_w = (dilation, dilation) if isinstance(dilation, int) else dilation

        # ── Learnable parameters ──────────────────────────────────
        shape = (out_channels, in_channels, k_h, k_w)
        self.weight = nn.Parameter(torch.empty(shape, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(
                torch.empty((out_channels,), device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        # ── Stored hyper-parameters ───────────────────────────────
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (k_h, k_w)
        self.stride = (s_h, s_w)
        self.padding = (p_h, p_w)
        self.dilation = (d_h, d_w)

        # ── Initialise parameters ─────────────────────────────────
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise weight and bias following the Kaiming/He uniform
        recipe (designed for layers typically followed by ReLU)."""
        init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / (fan_in**0.5)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the 2D convolution via im2col + matmul.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C_in, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, C_out, H_out, W_out).
        """

        x_patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        w = self.weight.view(self.out_channels, -1)
        out = w @ x_patches
        if self.bias is not None:
            out += self.bias.view(1, -1, 1)

        H_out = (
            x.shape[2]
            + 2 * self.padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1
        W_out = (
            x.shape[3]
            + 2 * self.padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1
        out = out.reshape(x.shape[0], self.out_channels, H_out, W_out)
        return out

    def extra_repr(self) -> str:
        """Return a formatted string for ``print(conv_layer)``."""
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"bias={self.bias is not None}"
        )
