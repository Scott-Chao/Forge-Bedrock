"""
core/nn/linear.py — Fully-connected (Linear) layer for neural networks.

Implements the affine transformation:

    y = xW^T + b

where W is a weight matrix of shape (out_features, in_features) and b is
a bias vector of shape (out_features,).  This is the same convention used
by PyTorch's torch.nn.Linear.
"""

from __future__ import annotations

import math

import numpy as np

from core.autograd import Value
from core.nn.init import kaiming_uniform_
from core.nn.module import Module
from core.nn.parameter import Parameter


class Linear(Module):
    """A fully connected layer: y = x @ W.T + b.

    Parameters
    ----------
    in_features : int
        Number of input features (columns of x).
    out_features : int
        Number of output features (columns of y).
    bias : bool, default=True
        Whether to include a learnable bias term.

    Attributes
    ----------
    weight : Parameter
        Shape (out_features, in_features).  The weight matrix.
    bias : Parameter or None
        Shape (out_features,).  The bias vector.  None if bias=False.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = Parameter(np.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(np.empty((out_features,)))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        kaiming_uniform_(self.weight)
        if self.bias is not None:
            fan_in = self.weight.data.shape[1]
            bound = 1 / math.sqrt(fan_in)
            self.bias.data = np.random.uniform(-bound, bound, size=self.bias.data.shape)

    def forward(self, x: Value) -> Value:
        y = x @ self.weight.T
        if self.bias is not None:
            y += self.bias
        return y

    def __repr__(self) -> str:
        return f"Linear(in={self.weight.data.shape[1]}, out={self.weight.data.shape[0]}, bias={self.bias is not None})"
