"""
core/nn/optim.py — Optimizers for neural network training.

A curriculum from vanilla SGD to velocity-based updates, covering the
evolution of first-order optimization methods.

Implementations follow the PyTorch optimizer convention.
"""

from __future__ import annotations

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
    """Nesterov Accelerated Gradient optimizer — look-ahead correction.

    The textbook NAG (Sutskever et al., 2013) evaluates the gradient at
    an approximate future position rather than the current one:

        lookahead = theta - beta * v
        v_new = beta * v + lr * grad(lookahead)
        theta -= v_new

    However, our autograd already computed grad(theta), not grad(lookahead).
    The trick: a change of variables makes the update equivalent while using
    only the available gradient.  The PyTorch-style parameterization is:

        v_new = beta * v + lr * g                         # same as Momentum
        theta -= beta * v_new + lr * g                     # Nesterov correction

    The correction term (beta * v_new) decelerates v before overshooting:
    if v_new points downhill but the lookahead would overshoot, the beta
    scaling of v_new adds a counteracting component.

    Intuition: standard momentum barrels downhill and only discovers it
    overshot after the fact.  NAG "peeks ahead" — if the lookahead has
    started climbing the opposite wall, the gradient correction already
    points backward, so deceleration happens *before* the overshoot.

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

    AdaGrad scales the learning rate for each parameter *inversely* to
    the square root of the sum of its past squared gradients:

        cache += g²
        theta -= lr / (sqrt(cache) + eps) * g

    Parameters that have received large gradients are closer to being
    well-optimised (or lie on a steep slope), so their step size is
    suppressed.  Parameters with small gradients get larger updates.

    This is a form of diagonal preconditioning: diag(1 / sqrt(cache))
    approximates the inverse curvature, analogous to what the Hessian
    provides in Newton's method but at trivial cost.

    Limitations
    -----------
    cache is monotonic (it sums squares, never decreases), so the
    effective learning rate *lr / sqrt(cache)* tends to zero over time.
    This is fine for convex problems but often kills learning before
    convergence in deep networks.

    Parameters
    ----------
    parameters : iterable of Parameter
        The model parameters to optimize.
    lr : float, default=0.01
        Global learning rate.  AdaGrad is often used with lr ~ 0.01.
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

    RMSProp fixes AdaGrad's monotonic decay by replacing the full sum of
    squared gradients with an exponentially weighted moving average (EWMA):

        cache = beta * cache + (1 - beta) * g²
        theta -= lr / (sqrt(cache) + eps) * g

    The EWMA gives RMSProp a *sliding window* over recent gradient history.
    When gradient magnitudes shrink (e.g. after passing a steep region),
    the cache decays and the learning rate recovers — something AdaGrad
    cannot do because its cache only grows.

    Effective window length ≈ 1 / (1 - beta) steps:
        beta = 0.9   →   ~10 steps
        beta = 0.99  →  ~100 steps
        beta = 0.999 → ~1000 steps (rarely used, very slow adaptation)

    Typically the same update rule is written without the (1 - beta) factor
    in PyTorch — check the source to see why this convention varies.

    Parameters
    ----------
    parameters : iterable of Parameter
        The model parameters to optimize.
    lr : float, default=0.001
        Global learning rate.  Typical range [0.0001, 0.01].
    beta : float, default=0.9
        Decay rate for the EWMA.  Controls how far back the cache
        remembers squared gradients.
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
