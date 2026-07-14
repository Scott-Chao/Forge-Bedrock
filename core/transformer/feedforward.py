"""
core/transformer/feedforward.py — Position-wise FeedForward Network.

A two-layer MLP with ReLU activation, applied independently to each
position in the sequence (no communication between tokens here).

    x -> Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model)

Used as the default FFN inside GPTBlock; MoEFFN in ``moe.py`` wraps
multiple ``FeedForward`` instances as its experts.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise FeedForward Network.

    A two-layer MLP with ReLU activation::

        x -> Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model)

    Applied independently to each position in the sequence.

    Parameters
    ----------
    d_model : int
        Input and output feature dimension.
    d_ff : int | None, optional (default=None)
        Hidden (intermediate) dimension. If None, defaults to ``4 * d_model``.
    bias : bool, optional (default=True)
        Whether to use bias in both linear layers.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, bias: bool = True):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.d_model = d_model
        self.d_ff = d_ff

        self.w_1 = nn.Linear(d_model, d_ff, bias=bias)
        self.relu = nn.ReLU()
        self.w_2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feedforward network.

        Parameters
        ----------
        x : (batch_size, seq_len, d_model)
            Input tensor.

        Returns
        -------
        out : (batch_size, seq_len, d_model)
            Output after expansion, ReLU, and compression.
        """
        hidden = self.w_1(x)
        hidden = self.relu(hidden)
        out = self.w_2(hidden)
        return out
