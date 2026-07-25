"""
core/gen/diffusion.py — Denoising Diffusion Probabilistic Models (Phase 8).

Building blocks for DDPM, following Ho, Jain & Abbeel (2020).

Components
----------
NoiseScheduler
    Manages the forward diffusion process: linear β schedule, cumulative
    ᾱ, and the closed-form noising formula xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε.
TimeEmbedding
    Sinusoidal positional encoding of timestep t (same formula as RoPE
    from Phase 5), projected through a small MLP.
"""

import math

import torch
import torch.nn as nn


class NoiseScheduler(nn.Module):
    """Manages the forward diffusion noise schedule.

    Implements the linear β schedule from DDPM:
        β₁ = 1e-4, β_T = 0.02,  linearly spaced in between

    Provides pre-computed buffers for fast closed-form noising:
        αₜ = 1 - βₜ
        ᾱₜ = ∏_{s=1}^t αₛ          (cumulative product)
        √ᾱₜ                        (signal scale)
        √(1 - ᾱₜ)                  (noise scale)
        √(1/αₜ)                    (reverse-step mean coefficient)
        (1 - αₜ) / √(1 - ᾱₜ)      (reverse-step noise coefficient)

    Parameters
    ----------
    timesteps : int
        Number of diffusion steps T (DDPM uses 1000).
    beta_start : float
        Starting noise rate β₁ (DDPM: 1e-4).
    beta_end : float
        Ending noise rate β_T (DDPM: 0.02).

    Usage
    -----
        scheduler = NoiseScheduler(timesteps=1000)
        scheduler = scheduler.to(device)            # move all buffers at once
        x_t = scheduler.add_noise(x_0, t)           # forward noising
    """

    def __init__(
        self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02
    ):
        super().__init__()
        self.timesteps = timesteps
        beta = torch.linspace(beta_start, beta_end, timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = torch.cat([alpha_bar.new_ones(1), alpha_bar[:-1]])

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())
        self.register_buffer("sqrt_recip_alpha", (1.0 / alpha).sqrt())
        self.register_buffer(
            "posterior_variance",
            beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar),
        )

    def add_noise(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffuse x₀ to xₜ in one step.

        Args:
            x_0: clean image  (batch, channels, H, W)
            t:   timestep indices  (batch,) — values in [0, timesteps-1]

        Returns:
            x_t: noised image at timestep t  (same shape as x_0)
            ε:   the Gaussian noise added    (same shape as x_0)
        """
        eps = torch.randn_like(x_0)
        signal_scale = self.sqrt_alpha_bar[t].view(-1, *([1] * (x_0.dim() - 1)))
        noise_scale = self.sqrt_one_minus_alpha_bar[t].view(
            -1, *([1] * (x_0.dim() - 1))
        )
        x_t = signal_scale * x_0 + noise_scale * eps
        return x_t, eps


class TimeEmbedding(nn.Module):
    """Sinusoidal time-step embedding for diffusion models.

    Composes ``SinusoidalPosEmbedding`` (raw sin/cos encoding) with a
    2-layer MLP (Linear → SiLU → Linear) to produce the final embedding.

    Parameters
    ----------
    dim : int
        Embedding dimension (common: 256, matches U-Net base channels).
    max_period : float
        Maximum period of the sinusoids (DDPM: 10000, same as Transformer).
    """

    def __init__(self, dim: int = 256, max_period: float = 10000.0):
        super().__init__()
        self.raw = SinusoidalPosEmbedding(dim, max_period)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timestep(s) into a continuous vector space.

        Args:
            t: timestep(s)  (batch,) — integer indices in [0, T-1]

        Returns:
            emb: shape (batch, dim) — time embedding vector(s)
        """
        return self.mlp(self.raw(t))


class SinusoidalPosEmbedding(nn.Module):
    """Raw sinusoidal encoding (no MLP), shared by several diffusion U-Net implementations.

    Applies the same sin/cos formula as TimeEmbedding but returns the
    raw encoding — useful when the projection is done inside a
    modulation block (e.g., FiLM scale/shift).

    Parameters
    ----------
    dim : int
        Embedding dimension.
    max_period : float
        Maximum period of the sinusoids (DDPM: 10000).
    """

    def __init__(self, dim: int = 256, max_period: float = 10000.0):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Raw sinusoidal encoding for timestep(s).

        Args:
            t: timestep(s)  (batch,) — integer indices in [0, T-1]

        Returns:
            emb: shape (batch, half*2) — raw sin/cos encoding
        """
        t_float = t.float().unsqueeze(-1)
        angles = t_float * self.freqs.unsqueeze(0)
        sin, cos = torch.sin(angles), torch.cos(angles)
        return torch.stack([sin, cos], dim=-1).flatten(-2)
