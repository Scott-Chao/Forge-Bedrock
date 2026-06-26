"""
core/nn/loss.py — Loss function modules.

Each loss module wraps the loss computation into a callable object
with a consistent interface, so it can be used seamlessly in
training loops alongside other nn components.
"""

from __future__ import annotations

import numpy as np
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


class L1Loss(Module):
    """L1 loss / Mean Absolute Error.

    Computes:

        L = (1 / N) * sum(|prediction - target|)

    Also called Mean Absolute Error (MAE).
    MLE derivation: Laplacian NLL → L1 loss.

    The gradient is constant (±1), making it robust to outliers compared
    to MSE, whose gradient grows linearly with the error.
    """

    def forward(self, prediction: Value, target: Value) -> Value:
        """Compute L1 = mean(|pred - target|)."""
        diff_data = prediction.data - target.data
        N = diff_data.size
        out_data = np.mean(np.abs(diff_data))
        out = Value(out_data, (prediction, target), "L1Loss")

        def _backward():
            grad = np.sign(diff_data) / N
            prediction.grad += grad

        out._backward = _backward

        return out

    def __repr__(self) -> str:
        return "L1Loss()"


class HuberLoss(Module):
    """Huber loss — smooth L1, quadratic near 0, linear far from 0.

        L = (1/N) * sum( huber(d_i) )

    where d_i = prediction_i - target_i and

        huber(d) = 0.5 * d²           if |d| <= delta
                   delta * |d| - 0.5 * delta²   if |d| > delta

    This combines the best of MSE (smooth at zero) and L1 (robust to
    outliers). The parameter delta controls where the transition happens.
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self._delta = delta

    def forward(self, prediction: Value, target: Value) -> Value:
        """Compute Huber loss."""
        diff_data = prediction.data - target.data
        N = diff_data.size
        out_data = np.mean(
            np.where(
                np.abs(diff_data) <= self._delta,
                0.5 * diff_data**2,
                self._delta * np.abs(diff_data) - 0.5 * self._delta**2,
            )
        )
        out = Value(out_data, (prediction, target), "HuberLoss")

        def _backward():
            grad = (
                np.where(
                    np.abs(diff_data) <= self._delta,
                    diff_data,
                    np.sign(diff_data) * self._delta,
                )
                / N
            )
            prediction.grad += grad

        out._backward = _backward

        return out

    def __repr__(self) -> str:
        return f"HuberLoss(delta={self._delta})"


class BCELoss(Module):
    """Binary Cross-Entropy loss for binary classification.

        L = -(1/N) * sum(y * log(p) + (1-y) * log(1-p))

    where p is the predicted probability (output of sigmoid) and
    y is the binary target {0, 1}.

    MLE derivation: Bernoulli NLL → BCE.
    Information theory: H(y, p) = H(y) + D_KL(y || p).

    NOTE: This implementation expects p to already be in (0, 1) — the
          sigmoid should be applied *before* passing to this loss.
    """

    def forward(self, prediction: Value, target: Value) -> Value:
        """Compute BCE loss."""
        eps = 1e-12
        p = np.clip(prediction.data, eps, 1 - eps)
        N = p.size
        out_data = -np.mean(target.data * np.log(p) + (1 - target.data) * np.log(1 - p))
        out = Value(out_data, (prediction, target), "BCELoss")

        def _backward():
            grad = -(target.data / p - (1 - target.data) / (1 - p)) / N
            prediction.grad += grad

        out._backward = _backward

        return out

    def __repr__(self) -> str:
        return "BCELoss()"


class CrossEntropyLoss(Module):
    """Cross-Entropy loss for multi-class classification.

    Fused Softmax + Negative Log-Likelihood for numerical stability.

        L = -(1/N) * sum_i log( exp(z_i,y_i) / sum_j exp(z_i,j) )

    where z are raw logits (not probabilities) and y are class indices.

    MLE derivation: Categorical NLL → CrossEntropy.
    Information theory: H(y, z) = H(y) + D_KL(y || softmax(z)).

    NUMERICAL STABILITY — the reason this is fused:
        If we computed softmax(z) then log(softmax(z)) separately, we'd
        lose precision when softmax(z) is very close to 0.
        The fused version computes log(p) directly via the identity:
            log(softmax(z)_k) = z_k - logsumexp(z)
        where logsumexp(z) = max(z) + log(sum(exp(z - max(z)))).

    NOTE: Takes raw logits, not probabilities.
          Targets are class indices (int), not one-hot vectors.
    """

    def forward(self, logits: Value, target: Value) -> Value:
        """Compute fused Cross-Entropy loss.

        Parameters
        ----------
        logits : Value
            Raw class scores, shape (batch_size, num_classes).
        target : Value
            Ground-truth class indices, shape (batch_size,).
            Internally converted to one-hot for gradient computation.
        """
        z = logits.data
        t = target.data.ravel()  # ensure 1-D: (batch_size,) — NOT (batch_size, 1)

        z_max = np.max(z, axis=1, keepdims=True)
        z_shifted = z - z_max

        log_sum_exp = z_max + np.log(np.sum(np.exp(z_shifted), axis=1, keepdims=True))
        log_probs = z - log_sum_exp

        batch_size = z.shape[0]
        correct_log_probs = log_probs[np.arange(batch_size), t]

        loss_data = -np.mean(correct_log_probs)
        out = Value(loss_data, (logits, target), "CrossEntropyLoss")

        def _backward():
            p = np.exp(z_shifted) / np.sum(np.exp(z_shifted), axis=1, keepdims=True)
            one_hot = np.zeros_like(p)
            one_hot[np.arange(batch_size), t] = 1.0
            grad = (p - one_hot) / batch_size
            logits.grad += grad

        out._backward = _backward

        return out

    def __repr__(self) -> str:
        return "CrossEntropyLoss()"
