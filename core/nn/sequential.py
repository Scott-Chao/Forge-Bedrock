"""
core/nn/sequential.py — A container for composing layers into a pipeline.

Sequential applies a list of layers one after the other: given an
input x, it returns layer_n(...(layer_2(layer_1(x)))...).

This is the primary way users define feedforward models.  Every
component in the pipeline must be callable (i.e. implement __call__
or forward), which is true of all layers in core.nn (Linear, ReLU,
Tanh, Sigmoid, and later Loss modules).
"""


class Sequential:
    """A sequential container for composing layers into a pipeline.

    Layers are applied in the order they are passed.

    Parameters
    ----------
    layers : list of callable
        The layers/modules to compose, in order.  Each layer must be
        callable with signature layer(x) -> output tensor.
    """

    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                yield from layer.parameters()

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        layers_str = "\n".join(
            f"  ({i}): {layer}" for i, layer in enumerate(self.layers)
        )
        return f"Sequential(\n{layers_str}\n)"

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)
