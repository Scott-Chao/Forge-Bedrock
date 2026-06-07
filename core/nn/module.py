"""
core/nn/module.py — Base class for all neural network modules.

Provides automatic parameter and sub-module registration via
__setattr__, recursive parameter iteration, gradient zeroing,
and train/eval mode toggling.

Inspired by PyTorch's torch.nn.Module.
"""

from core.nn.parameter import Parameter


class Module:
    """Base class for all neural network modules.

    Subclasses should call super().__init__() and assign Parameter
    instances (or child Module instances) as attributes.  The
    __setattr__ hook automatically registers them so that
    parameters() and zero_grad() work recursively.
    """

    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_training", True)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self):
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def train(self, mode=True):
        self._training = mode
        for m in self._modules.values():
            m.train(mode)

    def eval(self):
        """Set the module to evaluation mode (no dropout, fixed batchnorm, etc.)."""
        return self.train(False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")
