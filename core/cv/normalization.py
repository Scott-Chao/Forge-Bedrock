"""
core/cv/normalization.py — Batch Normalisation for 2D convolutions.

BatchNorm2d normalises the activations over the (N, H, W) dimensions
independently for each channel.  During training it uses the current
mini-batch statistics; during evaluation it uses running averages
accumulated over training.

This reduces internal covariate shift, allowing higher learning rates
and providing a mild regularisation effect.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init


class BatchNorm2d(nn.Module):
    """Batch normalisation over (N, H, W) for 2D feature maps.

    Parameters
    ----------
    num_features : int
        Number of input channels (C).
    eps : float, default=1e-5
        Small constant added to the variance to avoid division by zero.
    momentum : float, default=0.1
        Factor for updating the running statistics:

            running_var = momentum * running_var + (1 - momentum) * batch_var

    affine : bool, default=True
        If True, learn the per-channel affine transform (γ, β).
    track_running_stats : bool, default=True
        If True, maintain running mean/variance for use during
        evaluation.  Only disable for special cases (e.g. when
        training a very small model where running stats introduce
        unwanted state).

    Shape
    -----
    Input:  (N, C, H, W)
    Output: (N, C, H, W) — same shape, only values are normalised.

    Training vs Evaluation
    ----------------------
    Training forward:

        1. Compute μ_c, σ²_c over (N, H, W) of the current batch.
        2. Normalise: x̂ = (x - μ_c) / √(σ²_c + ε).
        3. Update running stats (if track_running_stats=True).
        4. Scale & shift: y = γ * x̂ + β.

    Evaluation forward:

        1. Use stored running_mean, running_var instead of batch stats.
        2. Normalise: x̂ = (x - running_mean_c) / √(running_var_c + ε).
        3. Scale & shift: y = γ * x̂ + β   (same γ, β as training).
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # ── Learnable scale & shift parameters ─────────────────────
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_features))
            self.bias = nn.Parameter(torch.empty(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # ── Running statistics (not gradients, just buffers) ───────
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        # ── Initialise parameters ──────────────────────────────────
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise γ ≈ 1 and β ≈ 0 so the normalised output is
        approximately identity at initialisation."""
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch normalisation."""
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), correction=0)
            if self.track_running_stats:
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var, self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        mean, var = mean[None, :, None, None], var[None, :, None, None]
        x_norm = (x - mean) * (var + self.eps).rsqrt()
        if self.affine:
            x_norm = (
                self.weight[None, :, None, None] * x_norm
                + self.bias[None, :, None, None]
            )
        return x_norm

    def extra_repr(self) -> str:
        """Return a formatted string for ``print(bn_layer)``."""
        return (
            f"num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )
