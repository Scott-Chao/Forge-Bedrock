"""
core/nn/optim.py — Optimizers for neural network training.

A curriculum from vanilla SGD to velocity-based updates, covering the
evolution of first-order optimization methods.

Implementations follow the PyTorch optimizer convention.
"""

from collections.abc import Iterator

import numpy as np
from core.nn.parameter import Parameter


class Optimizer:
    """Base class for all optimizers.

    Provides common state management shared by every optimizer:
      - Stores parameter list and learning rate.
      - Provides zero_grad() so subclasses don't reimplement it.

    Subclasses must implement step().
    """

    def __init__(self, parameters: Iterator[Parameter], lr: float) -> None:
        self.params = list(parameters)
        self.lr = lr

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = 0.0


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    The simplest update rule: step along the negative gradient direction.

        param.data -= lr * param.grad

    Parameters
    ----------
    parameters : iterable of Parameter
        The model parameters to optimize.  Typically obtained from
        model.parameters().
    lr : float, default=0.001
        Learning rate (step size).
    """

    def __init__(self, parameters: Iterator[Parameter], lr: float = 0.001) -> None:
        super().__init__(parameters, lr)

    def step(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad


class Momentum(Optimizer):
    """Momentum optimizer — velocity-based updates to dampen ravine oscillation.

    Instead of stepping directly along the gradient, Momentum maintains a
    velocity vector v that accumulates past gradients with exponential decay:

        v_{t+1} = beta * v_t + lr * grad_t
        param.data -= v_{t+1}

    Intuition: in a ravine the gradient oscillates perpendicular to the
    valley.  Since oscillations alternate sign (+ then -), they cancel out
    in the velocity accumulator.  Meanwhile, in the consistent direction,
    velocity builds up like a ball rolling downhill — the effective learning
    rate is amplified by ~ 1/(1-beta).

    Parameters
    ----------
    parameters : iterable of Parameter
        The model parameters to optimize.
    lr : float, default=0.01
        Learning rate (step size).  Typical range [0.001, 0.1].
    momentum : float, default=0.9
        Momentum coefficient (beta).  Controls how much past gradients
        influence the current velocity.
            beta = 0    → equivalent to plain SGD
            beta = 0.9  → typical starting value
            beta = 0.99 → strong smoothing, may overshoot
    """

    def __init__(
        self,
        parameters: Iterator[Parameter],
        lr: float = 0.01,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(parameters, lr)
        self.beta = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for param, v in zip(self.params, self.velocities):
            if param.grad is not None:
                v[...] = self.beta * v + self.lr * param.grad
                param.data -= v


class NAG(Optimizer):
    """NAG optimizer — Nesterov Accelerated Gradient with look-ahead correction.

    Textbook NAG (Sutskever et al., 2013) evaluates the gradient at an
    approximate future position:

        v_new = beta * v + lr * grad(theta - beta * v)
        theta -= v_new

    We implement the PyTorch-style parameterization that uses only the
    already-computed gradient (a change of variables makes it equivalent):

        v_new = beta * v + lr * g
        theta -= beta * v_new + lr * g                     # Nesterov correction

    The extra beta * v_new term decelerates v *before* overshooting:
    if the lookahead has started climbing the opposite ravine wall,
    the gradient correction already points backward.

    Theory: NAG achieves O(1/k²) convergence for smooth convex functions,
    versus O(1/k) for standard momentum.

    Parameters
    ----------
    parameters : iterable of Parameter
        The model parameters to optimize.
    lr : float, default=0.01
        Learning rate (step size).  Typical range [0.001, 0.1].
    momentum : float, default=0.9
        Momentum coefficient (beta) — same role as in Momentum.
    """

    def __init__(
        self,
        parameters: Iterator[Parameter],
        lr: float = 0.01,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(parameters, lr)
        self.beta = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for param, v in zip(self.params, self.velocities):
            if param.grad is not None:
                v[...] = self.beta * v + self.lr * param.grad
                param.data -= self.beta * v + self.lr * param.grad


class AdaGrad(Optimizer):
    """AdaGrad optimizer — per-parameter adaptive learning rates.

    Scales the learning rate for each parameter *inversely* to the
    accumulated magnitude of its past gradients:

        cache += g²
        theta -= lr / (sqrt(cache) + eps) * g

    This is diagonal preconditioning: diag(1/√cache) approximates inverse
    curvature, analogous to the Hessian in Newton's method but at trivial cost.

    Limitation: cache grows monotonically → effective LR tends to zero.
    Fine for convex problems but often kills learning before convergence
    in deep networks.

    Parameters
    ----------
    parameters : iterable of Parameter
        The model parameters to optimize.
    lr : float, default=0.01
        Global learning rate.  Often used with lr ~ 0.01.
    eps : float, default=1e-8
        Small constant for numerical stability when cache is near zero.
    """

    def __init__(
        self,
        parameters: Iterator[Parameter],
        lr: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, lr)
        self.eps = eps
        self.caches = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for param, c in zip(self.params, self.caches):
            if param.grad is not None:
                c[...] += param.grad**2
                adjusted_lr = self.lr / (np.sqrt(c) + self.eps)
                param.data -= adjusted_lr * param.grad


class RMSProp(Optimizer):
    """RMSProp optimizer — adaptive learning rates with sliding window.

    Fixes AdaGrad's monotonic decay by replacing the full sum of squared
    gradients with an exponentially weighted moving average (EWMA):

        cache = beta * cache + (1 - beta) * g²
        theta -= lr / (sqrt(cache) + eps) * g

    When gradient magnitudes shrink (e.g. after a steep region), the
    cache decays and LR recovers — something AdaGrad cannot do.

    Effective window ≈ 1 / (1 - beta) steps:
        beta = 0.9   →  ~10 steps
        beta = 0.99  →  ~100 steps

    Parameters
    ----------
    parameters : iterable of Parameter
        The model parameters to optimize.
    lr : float, default=0.001
        Global learning rate.  Typical range [0.0001, 0.01].
    beta : float, default=0.9
        Decay rate for the EWMA.
    eps : float, default=1e-8
        Small constant for numerical stability.
    """

    def __init__(
        self,
        parameters: Iterator[Parameter],
        lr: float = 0.001,
        beta: float = 0.9,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, lr)
        self.beta = beta
        self.eps = eps
        self.caches = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for param, c in zip(self.params, self.caches):
            if param.grad is not None:
                c[...] = self.beta * c + (1 - self.beta) * param.grad**2
                adjusted_lr = self.lr / (np.sqrt(c) + self.eps)
                param.data -= adjusted_lr * param.grad


class Adam(Optimizer):
    """Adam optimizer — adaptive moment estimation.

    Combines Momentum-style velocity smoothing with RMSProp-style
    per-parameter adaptive scaling, plus bias correction:

        m = beta1 * m + (1 - beta1) * g              # 1st moment (velocity)
        v = beta2 * v + (1 - beta2) * g**2            # 2nd moment (uncentered var)

        m_hat = m / (1 - beta1**t)                    # bias correction
        v_hat = v / (1 - beta2**t)

        theta -= lr * m_hat / (sqrt(v_hat) + eps)    # adaptive step

    Bias correction is critical at t=1: m and v are biased toward zero
    (m₁ ≈ 0.1g when β₁=0.9), producing vanishingly small updates without it.

    Parameters
    ----------
    parameters : iterable of Parameter
        The model parameters to optimize.
    lr : float, default=0.001
        Learning rate.  0.001 works well across many tasks.
    betas : tuple of float, default=(0.9, 0.999)
        Coefficients for running averages of gradient and its square.
    eps : float, default=1e-8
        Small constant for numerical stability.
    """

    def __init__(
        self,
        parameters: Iterator[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # step counter, used for bias correction
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is not None:
                m[...] = self.beta1 * m + (1 - self.beta1) * p.grad
                v[...] = self.beta2 * v + (1 - self.beta2) * p.grad**2
                m_hat = m / (1 - self.beta1**self.t)
                v_hat = v / (1 - self.beta2**self.t)
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """AdamW — Adam with decoupled weight decay.

    AdamW fixes a subtle interaction in the original Adam: when L2
    regularization (+λθ on the gradient) is passed through the adaptive
    scaler (m̂/√v̂), the regularization becomes non-uniform — parameters
    with large recent gradients get *more* shrinkage.  AdamW decouples:

        theta -= lr * (m_hat / (sqrt(v_hat) + eps))     # Adam update (inherited)
        theta -= lr * wd * theta                         # uniform weight decay

    Parameters
    ----------
    parameters : iterable of Parameter
        The model parameters to optimize.
    lr : float, default=0.001
        Learning rate.
    betas : tuple of float, default=(0.9, 0.999)
        Coefficients for running averages of gradient and its square.
    eps : float, default=1e-8
        Small constant for numerical stability.
    weight_decay : float, default=0.01
        Weight decay coefficient.  Applied uniformly to every active
        parameter after the adaptive step.
    """

    def __init__(
        self,
        parameters: Iterator[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(parameters, lr, betas, eps)
        self.wd = weight_decay

    def step(self) -> None:
        super().step()
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * self.wd * p.data
