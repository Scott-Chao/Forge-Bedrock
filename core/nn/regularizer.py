"""
core/nn/regularizer.py — Regularization penalties for training.

Regularization adds a penalty term to the loss function that discourages
overly complex models.  From a Bayesian perspective, this corresponds to
a prior distribution over the parameters.

Different priors give different regularizers:
    Gaussian prior  → L2 penalty  (λ/2 · Σ θ²)  — shrinkage, no sparsity
    Laplace prior   → L1 penalty  (λ · Σ |θ|)   — sparsity (drives θ → 0)
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from core.autograd import Value
from core.nn.parameter import Parameter


def l2_penalty(
    parameters: Iterator[Parameter],
    lambda_: float = 0.0,
) -> Value:
    """L2 weight penalty (Ridge / Gaussian prior / weight decay).

    Computes and returns a Value representing:

        L2 = (lambda_ / 2) * sum(param ** 2)

    summed across all elements of all provided parameters.

    The returned Value is wired into the autograd graph so its gradient
    (lambda_ * param for each parameter) flows naturally during backward().

    Parameters
    ----------
    parameters : iterable of Parameter
        The model parameters to regularise.  Typically model.parameters()
        or model.trainable_params().
    lambda_ : float, default=0.0
        Regularisation strength.  lambda_ > 0 activates the penalty.
        Typical values: 1e-4 to 0.1.

    Returns
    -------
    Value
        A scalar Value node representing the total L2 penalty.  Its
        backward pass adds lambda_ * param.data to each param.grad.
    """
    total = 0.0
    for param in parameters:
        total += np.sum(param.data**2)
    out = Value(total * lambda_ / 2, tuple(parameters), "L2Reg")

    def _backward():
        for param in parameters:
            if param.grad is not None:
                param.grad += lambda_ * param.data

    out.backward = _backward

    return out


def l1_penalty(
    parameters: Iterator[Parameter],
    lambda_: float = 0.0,
) -> Value:
    """L1 weight penalty (Lasso / Laplace prior).

    Computes and returns a Value representing:

        L1 = lambda_ * sum(|param|)

    summed across all elements of all provided parameters.

    L1 regularisation drives parameters toward exactly zero (sparsity),
    unlike L2 which only shrinks them but rarely reaches zero.  This is
    because the subgradient of |theta| at 0 is the interval [-lambda, lambda],
    which can "trap" the parameter at zero if the data-gradient is smaller
    than lambda.

    Parameters
    ----------
    parameters : iterable of Parameter
        The model parameters to regularise.
    lambda_ : float, default=0.0
        Regularisation strength.  Larger lambda_ = more sparsity.
        Typical values: 1e-5 to 0.01.

    Returns
    -------
    Value
        A scalar Value node representing the total L1 penalty.  Its
        backward pass adds lambda_ * sign(param.data) to each param.grad.
    """
    total = 0.0
    for param in parameters:
        total += np.sum(np.abs(param.data))
    out = Value(total * lambda_, tuple(parameters), "L1Reg")

    def _backward():
        for param in parameters:
            if param.grad is not None:
                param.grad += lambda_ * np.sign(param.data)

    out.backward = _backward

    return out
