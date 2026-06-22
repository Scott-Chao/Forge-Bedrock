"""
core/nn/sequential.py — A container for composing layers into a pipeline.

Sequential applies a list of layers one after the other: given an
input x, it returns layer_n(...(layer_2(layer_1(x)))...).

This is the primary way users define feedforward models.  Every
component in the pipeline must be callable (i.e. implement __call__
or forward), which is true of all layers in core.nn (Linear, ReLU,
Tanh, Sigmoid, and later Loss modules).
"""

from __future__ import annotations

from collections.abc import Callable

from core.nn.module import Module


class Sequential(Module):
    """A sequential container for composing layers into a pipeline.

    Layers are applied in the order they are passed.
    """

    def __init__(self, layers: list[Callable]) -> None:
        super().__init__()
        object.__setattr__(self, "layers", layers)
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)

    def forward(self, x):
        for layer in self._modules.values():
            x = layer(x)
        return x

    def __repr__(self) -> str:
        layers_str = "\n".join(
            f"  ({i}): {layer}" for i, layer in enumerate(self.layers)
        )
        return f"Sequential(\n{layers_str}\n)"

    def __getitem__(self, idx: int) -> Callable:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)
