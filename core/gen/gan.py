"""
core/gen/gan.py — Generative Adversarial Network (Phase 8).

GAN pits a Generator against a Discriminator in a min-max game:

    min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]

Components
----------
Generator
    Noise z → Linear → ConvTranspose2d upsamples → image (Tanh output).
Discriminator
    Image → Conv2d downsamples → Linear → real/fake logit.

Both follow DCGAN conventions: BatchNorm after conv layers (except
the first/last), no fully-connected hidden layers after the stem.
"""

import torch.nn as nn


class Generator(nn.Module):
    """Generator: noise z → image via ConvTranspose2d upsampling.

    DCGAN-style architecture for MNIST (28×28):
        Linear → Reshape → ConvTranspose2d → ConvTranspose2d → Tanh

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the noise vector z (common: 100).
    out_channels : int
        Number of output image channels (1 for grayscale, 3 for RGB).
    img_size : int
        Height/width of the square output image (must be divisible by 4).
    ngf : int
        Base channel count for generator (DCGAN convention: filters double
        as we go deeper; ngf is the *deepest* layer count.  Common: 64).
    """

    def __init__(
        self,
        latent_dim: int = 100,
        out_channels: int = 1,
        img_size: int = 28,
        ngf: int = 64,
    ):
        super().__init__()
        H0 = W0 = img_size // 4

        self.net = nn.Sequential(
            # Project noise to initial feature map
            nn.Linear(latent_dim, ngf * 4 * H0 * W0),
            nn.Unflatten(-1, (ngf * 4, H0, W0)),
            # Upsample: 7 → 14
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Upsample: 14 → 28
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Output layer (no BN, Tanh to match DCGAN convention)
            nn.ConvTranspose2d(ngf, out_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        """Generate an image from noise.

        Args:
            z: shape (batch, latent_dim)  — typically sampled from N(0, I)

        Returns:
            out: shape (batch, out_channels, img_size, img_size)
                 Values in (-1, 1) via Tanh.
        """
        return self.net(z)


class Discriminator(nn.Module):
    """Discriminator: image → real/fake logit via Conv2d downsampling.

    DCGAN-style (mirrors the Generator):
        Conv2d → LeakyReLU(0.2)  (no BN on first layer)
        Conv2d → BN → LeakyReLU
        Conv2d → BN → LeakyReLU
        Flatten → Linear → sigmoid (or raw logit)

    Parameters
    ----------
    in_channels : int
        Number of input image channels (1 for grayscale, 3 for RGB).
    img_size : int
        Height/width of the square input image (must be divisible by 4).
    ndf : int
        Base channel count for discriminator (DCGAN convention:
        filters double as we go deeper; common: 64).
    """

    def __init__(self, in_channels: int = 1, img_size: int = 28, ndf: int = 64):
        super().__init__()
        final_H = final_W = img_size // 8

        self.net = nn.Sequential(
            # Conv2d block 1 — no BN (DCGAN convention)
            nn.Conv2d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            # Conv2d block 2
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # Conv2d block 3
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            # Classifier
            nn.Flatten(),
            nn.Linear(ndf * 4 * final_H * final_W, 1),
        )

    def forward(self, x):
        """Classify image as real (high) or fake (low).

        Args:
            x: shape (batch, in_channels, img_size, img_size)

        Returns:
            out: shape (batch,) — raw logit (before sigmoid).
                 Positive → real, negative → fake.
        """
        return self.net(x).flatten()
