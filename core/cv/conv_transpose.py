"""
core/cv/conv_transpose.py — Transposed 2D Convolution (a.k.a. deconvolution).

ConvTranspose2d is the **transpose (adjoint)** of the linear operator
induced by Conv2d.  While Conv2d maps a large input to a smaller output
(downsampling via stride), ConvTranspose2d maps a small input to a
larger output (upsampling).

The forward pass uses the **col2im (fold)** approach:

    1.  Multiply the (C_in,)-per-position input by the transposed
        weight matrix to produce patch contributions.
    2.  Use ``F.fold`` to assemble those patches back into a spatial
        output, summing overlapping contributions.

This is the exact inverse of the Conv2d im2col pipeline:

    Conv2d:          unfold(x) @ W_flat          → fold the result
    ConvTranspose2d:  x_flat @ W_flat.t()  → col2im (fold)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ConvTranspose2d(nn.Module):
    """Transposed 2D convolution.

    Also referred to as a "deconvolution" (though it is not the
    mathematical inverse of convolution, only the transpose).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int | tuple[int, int]
        Size of the convolution kernel.
    stride : int | tuple[int, int], default=1
        Stride — acts as an **upsampling factor** in transposed conv.
    padding : int | tuple[int, int], default=0
        "Dilation" of the output — effectively crops ``padding`` pixels
        from each edge of the output.
    output_padding : int | tuple[int, int], default=0
        Additional padding added to one side of the output.  Used to
        resolve ambiguity when ``stride > 1`` and the output size
        formula is underspecified.
    dilation : int | tuple[int, int], default=1
        Spacing between kernel elements.
    bias : bool, default=True
        If True, adds a learnable bias to the output.
    device : torch.device | None, optional
    dtype : torch.dtype | None, optional

    Shape
    -----
    Input:  (N, C_in, H_in, W_in)
    Output: (N, C_out, H_out, W_out)

    where

        H_out = (H_in - 1) * stride[0]
                - 2 * padding[0]
                + dilation[0] * (kernel_size[0] - 1)
                + output_padding[0] + 1

        W_out = (W_in - 1) * stride[1]
                - 2 * padding[1]
                + dilation[1] * (kernel_size[1] - 1)
                + output_padding[1] + 1

    .. note::
        **Weight shape** is ``(in_channels, out_channels, k_h, k_w)``.
        This is the **transpose** of Conv2d's weight shape
        ``(out_channels, in_channels, k_h, k_w)`` — the two dimensions
        are swapped.

    The relationship with Conv2d for the **same** (weight, bias):

        Conv2d(x, weight, bias)   — maps  C_in → C_out,  downsamples
        ConvTranspose2d(x, weight, bias) — maps C_out → C_in, upsamples

    provides an explicit check pair.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
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
        op_h, op_w = (
            (output_padding, output_padding)
            if isinstance(output_padding, int)
            else output_padding
        )
        d_h, d_w = (dilation, dilation) if isinstance(dilation, int) else dilation

        # ── Learnable parameters ──────────────────────────────────
        shape = (in_channels, out_channels, k_h, k_w)
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
        self.output_padding = (op_h, op_w)
        self.dilation = (d_h, d_w)

        # ── Initialise parameters ─────────────────────────────────
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise weight and bias following Kaiming uniform.

        The fan for ConvTranspose2d is ``in_channels * k_h * k_w``
        (same as Conv2d's fan_in from a data-flow perspective: each
        input channel connects to ``k_h * k_w`` spatial locations).
        """
        init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / (fan_in**0.5)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the transposed convolution via matmul + fold.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C_in, H_in, W_in).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, C_out, H_out, W_out).
        """
        N, C_in, H_in, W_in = x.shape
        H_out = (
            (H_in - 1) * self.stride[0]
            - 2 * self.padding[0]
            + self.dilation[0] * (self.kernel_size[0] - 1)
            + self.output_padding[0]
            + 1
        )
        W_out = (
            (W_in - 1) * self.stride[1]
            - 2 * self.padding[1]
            + self.dilation[1] * (self.kernel_size[1] - 1)
            + self.output_padding[1]
            + 1
        )

        w = self.weight.view(C_in, -1)  # (C_in, C_out * k_h * k_w)
        x_flat = x.view(N, C_in, -1)  # (N, C_in, H_in * W_in)

        # For each spatial position in the input, produce a
        # (C_out * k_h * k_w)-sized patch contribution.
        patches = (x_flat.transpose(1, 2) @ w).transpose(1, 2)
        #   (N, L_in, C_in) @ (C_in, C_out*kh*kw) → (N, L_in, C_out*kh*kw)
        #   → transpose → (N, C_out*k_h*k_w, L_in)

        out = F.fold(
            patches,
            output_size=(H_out, W_out),
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )

        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)

        return out

    def extra_repr(self) -> str:
        """Return a formatted string for ``print(conv_layer)``."""
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, output_padding={self.output_padding}, "
            f"dilation={self.dilation}, bias={self.bias is not None}"
        )
