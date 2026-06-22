"""
core/nn/loss.py — Loss function modules.

Each loss module wraps the loss computation into a callable object
with a consistent interface, so it can be used seamlessly in
training loops alongside other nn components.
"""

from __future__ import annotations

from core.autograd import Value
from core.nn.module import Module


class MSELoss(Module):
    """Mean Squared Error loss for regression tasks.

    Computes:

        L = (1 / N) * sum((prediction - target)^2)

    where N is the total number of elements (batch_size * num_outputs).

    This is implemented purely via the existing autograd operations
    (subtraction, power, and reduction), so its backward pass is
    handled automatically by the computation graph.
    """

    def forward(self, prediction: Value, target: Value) -> Value:
        diff = prediction - target
        squared = diff**2
        return squared.mean()

    def __repr__(self) -> str:
        return "MSELoss()"
