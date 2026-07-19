"""
core/cv/pooling.py — 2D Pooling layers: MaxPool2d & AvgPool2d.

Pooling performs a sliding-window reduction over the spatial dimensions
of a feature map.  Two canonical reductions:

    MaxPool2d — keep only the maximum activation in each window.
                Preserves the strongest feature response; introduces
                local translation invariance.

    AvgPool2d — take the arithmetic mean of the window.
                Smooth downsampling; preserves the average energy of
                the feature response.

Both can be viewed through the same lens as Conv2d:
    1. Extract local patches via ``F.unfold`` (im2col-style).
    2. Reduce over the ``C_in * k_h * k_w`` dimension of each patch.
    3. Reshape back to (N, C_in, H_out, W_out).

Unlike Conv2d, pooling operates **per-channel independently** — it
never mixes information across channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Pool2d(nn.Module):
    """Base class for 2D pooling layers.

    Shares the common interface that MaxPool2d and AvgPool2d inherit:
    same parameter normalisation, same output-size formula, same
    ``F.unfold``-based patch extraction.  Subclasses need only define
    ``_reduction(patch)`` — a static method that reduces the last
    dimension of the extracted patches.

    Parameters
    ----------
    kernel_size : int | tuple[int, int]
        Size of the sliding window.
    stride : int | tuple[int, int], default=None
        Stride of the window.  If ``None``, defaults to ``kernel_size``
        (no overlap — typical for pooling).
    padding : int | tuple[int, int], default=0
        Implicit zero-padding on both spatial sides.

    Shape
    -----
    Input:  (N, C, H_in, W_in)
    Output: (N, C, H_out, W_out)

    where

        H_out = floor((H_in + 2 * padding[0] - kernel_size[0]) / stride[0] + 1)
        W_out = floor((W_in + 2 * padding[1] - kernel_size[1]) / stride[1] + 1)
    """

    _pad_value = 0.0

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()

        k_h, k_w = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        p_h, p_w = (padding, padding) if isinstance(padding, int) else padding
        if stride is None:
            stride = kernel_size
        s_h, s_w = (stride, stride) if isinstance(stride, int) else stride

        # ── Stored hyper-parameters ───────────────────────────────
        self.kernel_size = (k_h, k_w)
        self.stride = (s_h, s_w)
        self.padding = (p_h, p_w)

    # ── Output size helpers ──────────────────────────────────────────

    @staticmethod
    def _output_size(
        h_in: int,
        w_in: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
    ) -> tuple[int, int]:
        """Compute (H_out, W_out) for a given input spatial size."""
        h_out = (h_in - kernel_size[0]) // stride[0] + 1
        w_out = (w_in - kernel_size[1]) // stride[1] + 1
        return (h_out, w_out)

    # ── Reduction — override in subclasses ─────────────────────────

    @staticmethod
    def _reduction(patch: torch.Tensor, C: int) -> torch.Tensor:
        """Reduce over the ``C * k_h * k_w`` dimension of a patch.

        ``patch`` has raw unfold shape ``(N, C * k_h * k_w, L)``.
        Must return shape ``(N, C, L)``.

        This is the **only** method that differs between MaxPool2d and
        AvgPool2d.  Each subclass splits out the channel dimension
        internally before applying its reduction.
        """
        raise NotImplementedError("Override in subclass")

    # ── Forward ──────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the 2D pooling via unfold + reduction.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, C, H_out, W_out).
        """
        if self.padding != (0, 0):
            x = F.pad(
                x,
                (self.padding[1], self.padding[1], self.padding[0], self.padding[0]),
                value=self._pad_value,
            )
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        N, C, H, W = x.shape
        out = self._reduction(patches, C)
        H_out, W_out = self._output_size(H, W, self.kernel_size, self.stride)
        return out.view(N, C, H_out, W_out)

    def extra_repr(self) -> str:
        """Return a formatted string for ``print(pool_layer)``."""
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )


class MaxPool2d(_Pool2d):
    """2D max pooling.

    For each spatial window, keep the **maximum** value.  This acts as
    a "feature selector": it retains only the strongest activation in
    each local region, discarding weaker responses.
    """

    _pad_value = float("-inf")

    @staticmethod
    def _reduction(patch: torch.Tensor, C: int) -> torch.Tensor:
        """Max over the ``C * k_h * k_w`` dimension."""
        N, _, L = patch.shape
        return patch.view(N, C, -1, L).amax(dim=2)


class AvgPool2d(_Pool2d):
    """2D average pooling.

    For each spatial window, compute the **mean** of all values.
    This produces a smooth downsampled response without discarding
    information — but also without the sharp "feature detection"
    character of max pooling.
    """

    @staticmethod
    def _reduction(patch: torch.Tensor, C: int) -> torch.Tensor:
        """Average over the ``C * k_h * k_w`` dimension."""
        N, _, L = patch.shape
        return patch.view(N, C, -1, L).mean(dim=2)
