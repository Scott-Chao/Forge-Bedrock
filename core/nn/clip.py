"""
core/nn/clip.py — Gradient clipping utilities for training stability.

Gradient clipping prevents "gradient explosion" — rare mini-batches that
produce pathologically large gradients can throw the parameters into a
distant region of the loss surface, undoing hours of training.

Two strategies are provided:
- clip_grad_value_ : element-wise clamping of each gradient component
- clip_grad_norm_  : norm-based scaling that preserves gradient direction

Usage in a training loop (note: clip after backward(), before step()):

    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

The trailing underscore follows the PyTorch convention for in-place
operations.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from core.nn.parameter import Parameter


def clip_grad_value_(
    params: Iterator[Parameter],
    threshold: float,
) -> None:
    """Clip gradient values element-wise (in-place).

    Each component of every parameter's gradient is clamped to the range
    [-threshold, threshold]:

        g_i = clip(g_i, -threshold, threshold)

    This is simple and cheap, but it *changes the gradient direction*:
    dimensions with large components are clipped more aggressively than
    small ones, skewing the vector's angle.

    Parameters
    ----------
    params : iterable of Parameter
        The model parameters whose gradients will be clipped in-place.
    threshold : float
        Maximum absolute value for any single gradient element.
        Must be > 0.

    Raises
    ------
    ValueError
        If threshold <= 0.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")

    for param in params:
        if param.grad is not None:
            param.grad = np.clip(param.grad, -threshold, threshold)


def clip_grad_norm_(
    params: Iterator[Parameter],
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """Clip gradient norms (in-place), preserving gradient direction.

    If the total L2 norm of all parameter gradients exceeds max_norm,
    scale *every* gradient component by the same factor:

        total_norm = sqrt(sum_i g_i**2)
        if total_norm > max_norm:
            scale = max_norm / total_norm
            for each g_i:
                g_i *= scale

    This preserves the gradient *direction* while limiting the step
    magnitude — unlike value clipping which distorts the direction.

    The function returns the total norm *before* clipping, which is useful
    for logging (you can monitor how often clipping fires).

    Parameters
    ----------
    params : iterable of Parameter
        The model parameters whose gradients will be clipped in-place.
    max_norm : float
        Maximum allowed total norm.  If the norm exceeds this, the
        gradients are scaled down proportionally.
    norm_type : float, default=2.0
        Type of norm.  Only norm_type=2.0 (Euclidean / L2) is standard;
        other values are left for exploration.

    Returns
    -------
    float
        The total norm of all gradients *before* clipping (useful for
        tracking how often clipping triggers during training).
    """
    total_norm = _total_grad_norm(params, norm_type)
    if total_norm > max_norm and total_norm > 0:
        scale = max_norm / total_norm
        for param in params:
            if param.grad is not None:
                param.grad *= scale
    return total_norm


def _total_grad_norm(
    params: Iterator[Parameter],
    norm_type: float = 2.0,
) -> float:
    """Compute the total norm of all gradients (helper, no clipping).

    This utility function is split out so it can be reused (e.g. for
    logging gradient norms during training without modifying them).

    Parameters
    ----------
    params : iterable of Parameter
        Model parameters.
    norm_type : float, default=2.0
        Type of norm (typically 2.0 for Euclidean norm).

    Returns
    -------
    float
        The computed norm across all parameter gradients.
    """
    total = 0.0
    for param in params:
        if param.grad is not None:
            total += np.sum(param.grad**norm_type)
    return float(total ** (1 / norm_type))
