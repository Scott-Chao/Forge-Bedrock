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


def exp(x: Value) -> Value:
    """Exponential: e^{x}."""
    out_data = np.exp(x.data)
    out = Value(out_data, (x,), "Exp")

    def _backward():
        x.grad += out.data * out.grad

    out._backward = _backward
    return out


def log(x: Value) -> Value:
    """Natural logarithm: ln(x)."""
    out_data = np.log(x.data)
    out = Value(out_data, (x,), "Log")

    def _backward():
        x.grad += (1 / x.data) * out.grad

    out._backward = _backward
    return out


def sqrt(x: Value) -> Value:
    """Square root: x^{1/2}."""
    out_data = np.sqrt(x.data)
    out = Value(out_data, (x,), "Sqrt")

    def _backward():
        x.grad += 0.5 / out.data * out.grad

    out._backward = _backward
    return out


def softmax(x: Value) -> Value:
    """
    Softmax with stable log-sum-exp trick.

    p_i = exp(x_i) / sum_j exp(x_j)

    For numerical stability, subtract the max before exponentiating:
    shifted = x - max(x)
    p = exp(shifted) / sum(exp(shifted))

    The backward (Jacobian-vector product):
        ∂p_i / ∂x_j = p_i * (δ_ij - p_j)
        grad_x = p * (grad_out - dot(p, grad_out))
    """
    shifted = x.data - np.max(x.data)
    out_data = np.exp(shifted) / np.sum(np.exp(shifted))
    out = Value(out_data, (x,), "Softmax")

    def _backward():
        x.grad += out_data * (out.grad - np.dot(out_data, out.grad))

    out._backward = _backward
    return out


def log_softmax(x: Value) -> Value:
    """
    Log-softmax with stable log-sum-exp trick.

    log(p_i) = x_i - log-sum-exp(x)

    where log-sum-exp(x) = max(x) + log(sum(exp(x - max(x)))).

    The backward (Jacobian-vector product):
        ∂log(p_i) / ∂x_j = δ_ij - p_j
        grad_x = grad_out - sum(grad_out) * p
    """
    log_sum_exp_x = np.max(x.data) + np.log(np.sum(np.exp(x.data - np.max(x.data))))
    out_data = x.data - log_sum_exp_x
    out = Value(out_data, (x,), "LogSoftmax")

    def _backward():
        x.grad += out.grad - np.sum(out.grad) * np.exp(out.data)

    out._backward = _backward
    return out
