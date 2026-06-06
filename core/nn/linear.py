"""
core/nn/linear.py — Fully-connected (Linear) layer for neural networks.

Implements the affine transformation:

    y = xW^T + b

where W is a weight matrix of shape (out_features, in_features) and b is
a bias vector of shape (out_features,).  This is the same convention used
by PyTorch's torch.nn.Linear.
"""

import numpy as np

from core.nn.parameter import Parameter


class Linear:
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
        self.weight = Parameter(np.random.uniform(-1, 1, (out_features, in_features)))
        if bias:
            self.bias = Parameter(np.random.uniform(-1, 1, (out_features,)))
        else:
            self.bias = None

    def forward(self, x):
        y = x @ self.weight.T
        if self.bias is not None:
            y += self.bias
        return y

    def parameters(self):
        yield self.weight
        if self.bias is not None:
            yield self.bias

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Linear(in={self.weight.data.shape[1]}, out={self.weight.data.shape[0]}, bias={self.bias is not None})"
