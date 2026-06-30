"""
core/transformer/feedforward.py — Position-wise FeedForward Network.

The FeedForward Network (FFN) is a simple 2-layer MLP applied independently
to each token position. After tokens exchange information via attention, the
FFN processes each token's representation through:

    FFN(x) = W2 * ReLU(W1 * x + b1) + b2

The hidden dimension d_ff is typically 4 * d_model, giving the network
enough capacity to learn complex feature interactions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise FeedForward Network.

    A two-layer MLP with ReLU activation:

        x -> Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model)

    Applied independently to each position in the sequence
    (no communication between tokens here).

    Parameters
    ----------
    d_model : int
        Input and output feature dimension.
    d_ff : int
        Hidden (intermediate) dimension. Typically 4 * d_model.
    bias : bool, optional (default=True)
        Whether to use bias in both linear layers.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, bias: bool = True):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.d_model = d_model
        self.d_ff = d_ff

        self.w_1 = torch.nn.Linear(d_model, d_ff, bias=bias)
        self.relu = torch.nn.ReLU()
        self.w_2 = torch.nn.Linear(d_ff, d_model, bias=bias)

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
