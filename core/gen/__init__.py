"""
core/gen — Generative Models (Phase 8).

Three generative paradigms built on top of Phase 7's conv/conv-transpose backbone:

    VAE         — latent-variable inference via variational bound (ELBO)
    GAN         — adversarial equilibrium via min-max training
    Diffusion   — denoising score matching via forward/reverse process

Each module uses PyTorch (Phase 5+ convention) and reuses components from
core/cv (Conv2d, ConvTranspose2d, etc.) where appropriate.
"""

from .diffusion import (
    DDPM,
    NoiseScheduler,
    SinusoidalPosEmbedding,
    TimeConditionedUNet,
    TimeEmbedding,
)
from .gan import Discriminator, Generator
from .vae import VAE, Decoder, Encoder

__all__ = [
    "DDPM",
    "Decoder",
    "Discriminator",
    "Encoder",
    "Generator",
    "NoiseScheduler",
    "SinusoidalPosEmbedding",
    "TimeConditionedUNet",
    "TimeEmbedding",
    "VAE",
]
