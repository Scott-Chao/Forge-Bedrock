"""
core/nn/parameter.py — Parameter class for neural network modules.

A Parameter is a Value subclass that marks a tensor as a trainable
parameter.  The primary purpose is semantic: when an optimizer walks
the module tree, it collects Parameter instances to update, ignoring
intermediate values in the computation graph.

Inspired by PyTorch's torch.nn.Parameter.
"""

from core.autograd import Value


class Parameter(Value):
    """A trainable parameter that is automatically registered by a Module.

    Parameters are Value nodes that represent weights, biases, or any
    other learned quantity in a neural network layer.  They are always
    leaf nodes in the computation graph (no parents / no operation).

    Parameters
    ----------
    data : int | float | np.ndarray
        The initial numerical data.  For a Linear layer this would be
        a weight matrix or bias vector.
    """

    def __init__(self, data):
        super().__init__(data)

    def __repr__(self):
        return f"Parameter(data={self.data}, grad={self.grad})"
