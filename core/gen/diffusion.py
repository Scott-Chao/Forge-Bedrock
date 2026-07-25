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


# ============================================================================
# Helpers for TimeConditionedUNet
# ============================================================================


def _film(h: torch.Tensor, gamma_beta: torch.Tensor) -> torch.Tensor:
    """Apply FiLM modulation: γ · h + β."""
    gamma, beta = gamma_beta.chunk(2, dim=-1)
    gamma = gamma.unsqueeze(-1).unsqueeze(-1)
    beta = beta.unsqueeze(-1).unsqueeze(-1)
    return gamma * h + beta


class _Block(nn.Module):
    """Single conv block with time conditioning: Conv → GN → SiLU → FiLM."""

    def __init__(
        self, in_channels: int, out_channels: int, time_dim: int, stride: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        self.film = nn.Linear(time_dim, out_channels * 2)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv(h)
        h = self.norm(h)
        h = torch.nn.functional.silu(h)
        h = _film(h, self.film(t_emb))
        return h


class _EncoderStage(nn.Module):
    """One encoder stage: two conv blocks, second with stride-2 downsampling.

    Stores skip feature (output of block 1) before downsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.block1 = _Block(in_channels, out_channels, time_dim, stride=1)
        self.block2 = _Block(out_channels, out_channels, time_dim, stride=2)

    def forward(
        self, h: torch.Tensor, t_emb: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        h = self.block1(h, t_emb)
        skips.append(h)
        h = self.block2(h, t_emb)
        return h


class _DecoderStage(nn.Module):
    """One decoder stage: upsample → concat skip → conv block."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv_block = _Block(in_channels, out_channels, time_dim)

    def forward(
        self, h: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        h = nn.functional.interpolate(h, scale_factor=2.0, mode="nearest")
        if h.shape[-1] != skip.shape[-1]:
            h = nn.functional.interpolate(h, size=skip.shape[-2:], mode="nearest")
        h = torch.cat([skip, h], dim=1)
        return self.conv_block(h, t_emb)


class TimeConditionedUNet(nn.Module):
    """U-Net with time-step conditioning via FiLM (scale/shift modulation).

    DDPM-style U-Net for MNIST (28×28):
        Encoder: 3 stages (64 → 128 → 256, stride-2 downsampling)
        Bottleneck: 256 → 256
        Decoder: 3 stages (256+skip → 128+skip → 64, upsample + concat)
        Output: 64 → out_channels

    At every conv block, time is injected via FiLM:
        γ(t) · h + β(t)
    where γ, β are predicted from the time embedding by per-stage MLPs.

    Uses GroupNorm + SiLU (DDPM convention) instead of BN + ReLU.

    Parameters
    ----------
    in_channels : int
        Input image channels (1 for grayscale).
    out_channels : int
        Output channels (same as in_channels for noise prediction).
    base_channels : int
        Channels at the first encoder stage (doubles each stage).
    depth : int
        Number of encoder/decoder stages.
    time_dim : int
        Time embedding dimension.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 3,
        time_dim: int = 256,
    ):
        super().__init__()
        self.depth = depth
        self.base_channels = base_channels
        self.time_dim = time_dim

        self.time_embed = SinusoidalPosEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ── Encoder ─────────────────────────────────────────────────
        self.encoder = nn.ModuleList()
        in_ch = in_channels
        for level in range(depth):
            out_ch = base_channels * (2**level)
            self.encoder.append(_EncoderStage(in_ch, out_ch, time_dim))
            in_ch = out_ch

        # ── Bottleneck ──────────────────────────────────────────────
        b_ch = base_channels * (2 ** (depth - 1))
        self.bottleneck = nn.ModuleList(
            [_Block(b_ch, b_ch, time_dim) for _ in range(2)]
        )

        # ── Decoder ─────────────────────────────────────────────────
        self.decoder = nn.ModuleList()
        for level in range(depth - 1, -1, -1):
            skip_ch = base_channels * (2**level)
            prev_ch = b_ch if level == depth - 1 else base_channels * (2 ** (level + 1))
            conv_in = prev_ch + skip_ch
            conv_out = skip_ch if level > 0 else base_channels
            self.decoder.append(_DecoderStage(conv_in, conv_out, time_dim))

        # ── Output ──────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise given noisy image and timestep.

        Args:
            x: noised image   (batch, in_channels, H, W)
            t: timestep       (batch,)

        Returns:
            out: predicted noise (batch, out_channels, H, W)
        """
        t_emb = self.time_mlp(self.time_embed(t))

        # ── Encoder ─────────────────────────────────────────────────
        skips = []
        h = x
        for stage in self.encoder:
            h = stage(h, t_emb, skips)

        # ── Bottleneck ──────────────────────────────────────────────
        for block in self.bottleneck:
            h = block(h, t_emb)

        # ── Decoder ─────────────────────────────────────────────────
        for stage in self.decoder:
            skip = skips.pop()
            h = stage(h, skip, t_emb)

        return self.out_conv(h)
