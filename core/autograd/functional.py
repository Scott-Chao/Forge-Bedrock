"""
core/autograd/functional.py — Functions built on the Value class.

Each activation is an atomic operation: it creates a single Value node with
its own _backward closure, rather than composing simpler operations.
This keeps the computation graph compact and avoids intermediate nodes.
"""

import numpy as np
from .value import Value


def relu(x: Value) -> Value:
    """Rectified Linear Unit: max(0, x)."""
    out_data = np.maximum(0, x.data)
    out = Value(out_data, (x,), "ReLU")

    def _backward():
        x.grad += (out.data > 0) * out.grad

    out._backward = _backward
    return out


def sigmoid(x: Value) -> Value:
    """Sigmoid / logistic function: 1 / (1 + e^{-x})."""
    out_data = 1 / (1 + np.exp(-x.data))
    out = Value(out_data, (x,), "Sigmoid")

    def _backward():
        x.grad += (out.data * (1 - out.data)) * out.grad

    out._backward = _backward
    return out


def tanh(x: Value) -> Value:
    """Hyperbolic tangent: (e^{x} - e^{-x}) / (e^{x} + e^{-x})."""
    out_data = np.tanh(x.data)
    out = Value(out_data, (x,), "Tanh")

    def _backward():
        x.grad += (1 - out.data**2) * out.grad

    out._backward = _backward
    return out
