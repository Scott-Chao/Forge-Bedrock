"""
core/nn/lr_scheduler.py — Learning rate schedules for training.

LR schedulers are orthogonal to optimizers: they don't change the update
rule, they adjust the learning rate over time.

Usage pattern in a training loop:

    optimizer = SGD(model.parameters(), lr=0.1)
    scheduler = StepDecay(optimizer, step_size=30, gamma=0.1)

    for epoch in range(epochs):
        for batch in dataloader:
            loss = ...               # forward + backward
            optimizer.step()         # update params with current lr
            scheduler.step()         # adjust lr for the next iteration

Schedulers follow the PyTorch convention: step() is called *after*
each optimizer step, incrementing an internal counter and recomputing lr.
"""

from __future__ import annotations

import numpy as np
from core.nn.optim import Optimizer


class LRScheduler:
    """Base class for all learning rate schedulers.

    Every scheduler wraps an Optimizer instance and modifies its .lr
    attribute each time step() is called.

    Subclasses must implement _get_lr() -> float, which computes the
    new learning rate from the current step count.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer whose learning rate will be adjusted.
    """

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self.base_lr = optimizer.lr  # snapshot: used by every subclass
        self._step_count = 0

    def step(self) -> None:
        self._step_count += 1
        self.optimizer.lr = self._get_lr()

    def _get_lr(self) -> float:
        raise NotImplementedError


class StepDecay(LRScheduler):
    """Step decay learning rate schedule.

    Reduces the learning rate by a factor gamma every step_size steps:

        lr_t = base_lr * gamma ** floor(t / step_size)

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer whose learning rate will be adjusted.
    step_size : int, default=30
        Number of steps between successive LR drops.
    gamma : float, default=0.1
        Multiplicative factor applied at each drop.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int = 30,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def _get_lr(self) -> float:
        exponent = self._step_count // self.step_size
        return self.base_lr * self.gamma**exponent


class CosineAnnealing(LRScheduler):
    """Cosine annealing learning rate schedule.

    Smoothly decreases the learning rate from base_lr to eta_min over
    T_max steps following a cosine curve:

        lr_t = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(t / T_max * pi))

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer whose learning rate will be adjusted.
    T_max : int, default=100
        Maximum number of steps in the annealing cycle.
    eta_min : float, default=0.0
        Minimum (final) learning rate.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int = 100,
        eta_min: float = 0.0,
    ) -> None:
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def _get_lr(self) -> float:
        progress = np.clip(self._step_count / self.T_max, 0.0, 1.0)
        cos_weight = (1 + np.cos(progress * np.pi)) / 2
        return self.eta_min + (self.base_lr - self.eta_min) * cos_weight


class Warmup(LRScheduler):
    """Linear warmup learning rate schedule.

    Linearly increases the learning rate from 0 to base_lr over the first
    warmup_steps, then stays at base_lr:

        lr_t = base_lr * (t / warmup_steps),    t <= warmup_steps
        lr_t = base_lr,                          t > warmup_steps

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer whose learning rate will be adjusted.
    warmup_steps : int, default=10
        Number of steps over which the LR linearly increases.
    """

    def __init__(self, optimizer: Optimizer, warmup_steps: int = 10) -> None:
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps

    def _get_lr(self) -> float:
        if self._step_count <= self.warmup_steps:
            return self.base_lr * (self._step_count / self.warmup_steps)
        else:
            return self.base_lr


class WarmupCosine(LRScheduler):
    """Warmup + Cosine Annealing — the modern standard schedule.

    Combines linear warmup with cosine decay in a two-phase schedule:

        Phase 1 — Warmup (t < warmup_steps):
            lr_t = base_lr * (t / warmup_steps)

        Phase 2 — Cosine annealing (t >= warmup_steps):
            lr_t = eta_min + 0.5 * (base_lr - eta_min)
                   * (1 + cos((t - warmup_steps) / (total_steps - warmup_steps) * pi))

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer whose learning rate will be adjusted.
    warmup_steps : int, default=500
        Number of steps for linear warmup.
    total_steps : int, default=5000
        Total number of training steps (warmup + cosine).
    eta_min : float, default=0.0
        Minimum (final) learning rate after cosine decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 500,
        total_steps: int = 5000,
        eta_min: float = 0.0,
    ) -> None:
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min

    def _get_lr(self) -> float:
        if self._step_count <= self.warmup_steps:
            return self.base_lr * (self._step_count / self.warmup_steps)
        else:
            progress = np.clip(
                (self._step_count - self.warmup_steps)
                / (self.total_steps - self.warmup_steps),
                0.0,
                1.0,
            )
            cos_weight = (1 + np.cos(progress * np.pi)) / 2
            return self.eta_min + (self.base_lr - self.eta_min) * cos_weight
