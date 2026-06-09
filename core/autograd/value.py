"""
core/autograd/value.py — The scalar-valued autograd building block.

Inspired by Karpathy's micrograd, this module implements reverse-mode
automatic differentiation via a dynamic computation graph (DAG) that is
constructed on-the-fly through Python operator overloading.

Every arithmetic operation creates a new Value node wired to its parents,
recording the operation type.  When backward() is eventually called, the
graph is traversed in topological order and the chain rule is applied at
each node to compute gradients.

Phase 2 uses raw NumPy ndarrays (not the Matrix class from Phase 1).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class Value:
    """A node in a dynamic computation graph that supports autograd.

    Each node stores:
      - data  : the numerical value (scalar or NumPy ndarray)
      - grad  : accumulated gradient from backward passes
      - _children : set of parent Value nodes that produced this node
      - _op   : the operation label (e.g. '+', '*', '**')
      - _backward : callable that computes local gradients w.r.t. parents

    The DAG is built implicitly: every arithmetic operation (+, *, **, etc.)
    constructs a new Value, links it to its inputs via _children, and stores
    a _backward closure that will later implement the chain rule for this
    specific operation.

    Parameters
    ----------
    data : int | float | np.ndarray
        The numerical data wrapped by this node.
    children : tuple[Value, ...], optional
        Parent nodes this value was derived from.
    op : str, optional
        A short string label for the operation that created this node.
    """

    def __init__(
        self,
        data,
        children: Tuple["Value", ...] = (),
        op: str = "",
    ):
        self.data = data
        self.grad = 0.0
        self._children = set(children)
        self._op = op
        self._backward = lambda: None

    def _ensure_value(self, other):
        """Wrap a Python scalar as a Value if it isn't one already.

        HINT: This helper is used in __add__, __mul__, etc. so that
              `Value(2) + 3` works without the user explicitly wrapping 3.
        """
        if not isinstance(other, Value):
            return Value(other)
        return other

    def _expand_to(self, target_shape):
        """Return a new Value whose data is broadcast to `target_shape`.

        Forward:  self.data is "stretched" to target_shape by replicating
                  elements along dimensions where the original size is 1,
                  or by inserting new leading dimensions.

        Backward: The gradient must be reduced back to the original shape.
                  This is the inverse of replication — sum along every axis
                  that was added or repeated.

        Parameters
        ----------
        target_shape : tuple of int
            The shape to broadcast to.
        """
        out = Value(np.broadcast_to(self.data, target_shape), (self,), "expand")

        def _backward():
            grad = out.grad
            orig_data = np.asarray(self.data)
            # Collapse extra leading dims created by broadcasting
            while grad.ndim > orig_data.ndim:
                grad = np.sum(grad, axis=0)
            # Collapse dims where original size == 1 (was replicated)
            for d in range(orig_data.ndim):
                if orig_data.shape[d] == 1 and grad.shape[d] != 1:
                    grad = np.sum(grad, axis=d, keepdims=True)
            self.grad += grad.reshape(orig_data.shape)

        out._backward = _backward

        return out

    @staticmethod
    def _broadcast_pair(a: "Value", b: "Value") -> tuple["Value", "Value"]:
        """Broadcast two Values to a common shape for element-wise ops."""
        a_data = np.asarray(a.data)
        b_data = np.asarray(b.data)
        target_shape = np.broadcast_shapes(a_data.shape, b_data.shape)
        if a_data.shape != target_shape:
            a = a._expand_to(target_shape)
        if b_data.shape != target_shape:
            b = b._expand_to(target_shape)
        return a, b

    def __add__(self, other):
        other = self._ensure_value(other)
        self, other = self._broadcast_pair(self, other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = self._ensure_value(other)
        self, other = self._broadcast_pair(self, other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        out = Value(self.data**other, (self,), "**")

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** (-1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def __rtruediv__(self, other):
        return self**-1 * other

    @staticmethod
    def _topological_sort(root: "Value"):
        """
        Produce a topological ordering of the DAG rooted at `root`.

        Uses DFS: visit children first, then append the current node.
        The returned list has parents *before* children, which is the
        standard topological order.
        """
        visited = set()
        order = []

        def _dfs(node):
            if node in visited:
                return
            visited.add(node)
            for child in node._children:
                _dfs(child)
            order.append(node)

        _dfs(root)
        return order

    def backward(self, gradient=None):
        """Kick off reverse-mode autograd from this node.

        Parameters
        ----------
        gradient : Value-compatible type, optional
            The initial gradient.  Required when the output is non-scalar
            (e.g. a matrix).  For scalar outputs, defaults to 1.0, matching
            PyTorch's convention that backward() on a scalar needs no argument.
        """
        order = self._topological_sort(self)
        for node in order:
            if node._op:
                node.grad = 0.0
        if gradient is None:
            self.grad = 1.0
        else:
            self.grad = gradient
        for node in reversed(order):
            node._backward()

    @property
    def T(self):
        out = Value(self.data.T, (self,), "T")

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = self._ensure_value(other)
        out = Value(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
