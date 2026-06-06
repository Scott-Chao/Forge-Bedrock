"""
core/nn/activation.py — Activation function wrappers for use as layers.

Each class wraps a function from core/autograd/functional into a
callable object with the same interface as Linear, so they can be
used interchangeably in a Sequential pipeline.
"""

from core.autograd import functional as F


class ReLU:
    """Rectified Linear Unit activation: max(0, x)."""

    def forward(self, x):
        return F.relu(x)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "ReLU()"


class Tanh:
    """Hyperbolic tangent activation: (e^x - e^-x) / (e^x + e^-x)

    Output range: (-1, 1)
    """

    def forward(self, x):
        return F.tanh(x)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "Tanh()"


class Sigmoid:
    """Logistic sigmoid activation: 1 / (1 + e^-x)

    Output range: (0, 1).  Commonly used in the output layer for
    binary classification.
    """

    def forward(self, x):
        return F.sigmoid(x)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "Sigmoid()"
