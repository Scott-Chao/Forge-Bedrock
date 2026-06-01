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

from typing import Optional, Set, Tuple


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

    def __add__(self, other):
        other = self._ensure_value(other)
        out = Value(self.data + other.data, (self, other), "+")
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = self._ensure_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        out = Value(self.data ** other, (self,), "**")

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
        return self ** -1 * other

    def backward(self):
        """Kick off reverse-mode autograd from this node.

        The full implementation requires:
          1. Topological sort of the DAG (next roadmap item).
          2. Traversing in reverse order, calling each node's _backward.
          3. Gradient accumulation.

        For now this is a placeholder.
        """
        # TODO: implement topological sort
        # TODO: traverse in reverse order, calling _backward on each node
        # HINT: self.grad should initialise to 1.0 (dself/dself = 1)
        return NotImplemented

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
