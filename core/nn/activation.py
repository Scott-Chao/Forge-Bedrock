"""
core/nn/activation.py — Activation function wrappers for use as layers.

Each class wraps a function from core/autograd/functional into a
callable object with the same interface as Linear, so they can be
used interchangeably in a Sequential pipeline.
"""

from __future__ import annotations

from core.autograd import Value
from core.autograd import functional as F
from core.nn.module import Module


class ReLU(Module):
    """Rectified Linear Unit activation: max(0, x)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Value) -> Value:
        return F.relu(x)

    def __repr__(self) -> str:
        return "ReLU()"


class Tanh(Module):
    """Hyperbolic tangent activation: (e^x - e^-x) / (e^x + e^-x)

    Output range: (-1, 1)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Value) -> Value:
        return F.tanh(x)

    def __repr__(self) -> str:
        return "Tanh()"


class Sigmoid(Module):
    """Logistic sigmoid activation: 1 / (1 + e^-x)

    Output range: (0, 1).  Commonly used in the output layer for
    binary classification.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Value) -> Value:
        return F.sigmoid(x)

    def __repr__(self) -> str:
        return "Sigmoid()"
