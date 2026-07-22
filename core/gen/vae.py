"""
core/gen/vae.py — Variational Autoencoder (Phase 8).

VAE marries an encoder network (inference) with a decoder network (generation)
via a latent bottleneck, trained by maximising the Evidence Lower BOund (ELBO).

Components
----------
Encoder
    Conv → Linear → (mu, logvar)  — maps input to latent distribution parameters.
Decoder
    Linear → ConvTranspose2d      — maps latent sample z back to reconstructed image.
VAE
    Encoder + Reparameterise + Decoder + ELBO loss in one Module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Convolutional encoder: image → (mu, logvar).

    Architecture (square input, e.g. 28×28 MNIST):
        Conv(1→32, k=4, s=2, p=1) → ReLU          # H → H/2
        Conv(32→64, k=4, s=2, p=1) → ReLU         # H/2 → H/4
        Flatten → Linear → ReLU
        |→ fc_mu     (Linear, no activation)
        |→ fc_logvar (Linear, no activation)

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale, 3 for RGB).
    hidden_dim : int
        Width of the penultimate hidden layer.
    latent_dim : int
        Dimensionality of the latent space.
    img_size : int
        Height/width of the square input image.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 256,
        latent_dim: int = 20,
        img_size: int = 28,
    ):
        super().__init__()
        conv_hw = img_size // 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * conv_hw * conv_hw, hidden_dim),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input into latent distribution parameters.

        Args:
            x: shape (batch, in_channels, H, W)

        Returns:
            mu:     shape (batch, latent_dim) — mean of q(z|x)
            logvar: shape (batch, latent_dim) — log-variance of q(z|x)
        """
        h = self.stem(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Convolutional decoder: latent z → reconstructed image.

    Mirrors Encoder in reverse:
        Linear → ReLU → Linear → ReLU → Reshape
        ConvTranspose2d(64→32, k=4, s=2, p=1) → ReLU   # H/4 → H/2
        ConvTranspose2d(32→out, k=4, s=2, p=1) → Sigmoid  # H/2 → H

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the input latent vector z.
    hidden_dim : int
        Width of the penultimate hidden layer (mirrors Encoder).
    out_channels : int
        Number of output channels (1 for grayscale, 3 for RGB).
    img_size : int
        Height/width of the square output image.
    """

    def __init__(
        self,
        latent_dim: int = 20,
        hidden_dim: int = 256,
        out_channels: int = 1,
        img_size: int = 28,
    ):
        super().__init__()
        conv_hw = img_size // 4
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 64 * conv_hw * conv_hw),
            nn.ReLU(True),
            nn.Unflatten(-1, (64, conv_hw, conv_hw)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent sample into a reconstructed image.

        Args:
            z: shape (batch, latent_dim)

        Returns:
            x_recon: shape (batch, out_channels, H, W) — values in (0, 1)
                     via Sigmoid (suitable for BCE loss).
        """
        return self.net(z)


class VAE(nn.Module):
    """Variational Autoencoder: Encoder → Reparameterise → Decoder.

    Provides a ``loss_function`` that computes the ELBO:
        L = -E_q[log p(x|z)] + KL(q(z|x) || p(z))

    Usage
    -----
        vae = VAE(latent_dim=20)
        x = torch.randn(4, 1, 28, 28)         # MNIST batch
        recon, mu, logvar = vae(x)             # forward (reparameterised)
        loss = vae.loss_function(recon, x, mu, logvar)
        loss.backward()
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 20,
        hidden_dim: int = 256,
        img_size: int = 28,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim, img_size)
        self.decoder = Decoder(latent_dim, hidden_dim, in_channels, img_size)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick: z = mu + sigma * eps.

        Moves stochasticity outside the computation graph so gradients
        can flow back to the encoder.

        Args:
            mu:     shape (batch, latent_dim)
            logvar: shape (batch, latent_dim)

        Returns:
            z: shape (batch, latent_dim)
        """
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward pass with reparameterised sampling.

        Args:
            x: shape (batch, in_channels, H, W)

        Returns:
            recon:  reconstructed image  (batch, in_channels, H, W)
            mu:     latent mean          (batch, latent_dim)
            logvar: latent log-variance  (batch, latent_dim)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @staticmethod
    def loss_function(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the negative ELBO as a training loss.

        L = BCE(recon_x, x) + KL(N(mu, sigma²) || N(0, 1))

        Args:
            recon_x: reconstructed image  (batch, in_channels, H, W)  — in (0,1)
            x:       original image       (batch, in_channels, H, W)  — in [0,1]
            mu:      latent mean          (batch, latent_dim)
            logvar:  latent log-variance  (batch, latent_dim)

        Returns:
            dict with keys 'loss', 'recon_loss', 'kl_loss'.
        """
        batch_size = x.size(0)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / batch_size
        kl_loss = 0.5 * (mu**2 + torch.exp(logvar) - logvar - 1).sum() / batch_size
        return {
            "loss": recon_loss + kl_loss,
            "recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }
