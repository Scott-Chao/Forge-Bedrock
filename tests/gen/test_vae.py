"""tests/gen/test_vae.py — Tests for VAE building blocks."""

import pytest
import torch
from core.gen import VAE, Decoder, Encoder


class TestEncoder:
    @pytest.mark.parametrize(
        "batch,channels,img_size,latent_dim",
        [
            (2, 1, 28, 20),  # MNIST standard
            (4, 3, 32, 16),  # CIFAR-style
        ],
    )
    def test_output_shape(self, batch, channels, img_size, latent_dim):
        enc = Encoder(in_channels=channels, latent_dim=latent_dim, img_size=img_size)
        mu, logvar = enc(torch.randn(batch, channels, img_size, img_size))
        assert mu.shape == (batch, latent_dim)
        assert logvar.shape == (batch, latent_dim)


class TestDecoder:
    @pytest.mark.parametrize(
        "batch,latent_dim,out_channels,img_size",
        [
            (2, 20, 1, 28),
            (4, 16, 3, 32),
        ],
    )
    def test_output_shape(self, batch, latent_dim, out_channels, img_size):
        dec = Decoder(
            latent_dim=latent_dim, out_channels=out_channels, img_size=img_size
        )
        recon = dec(torch.randn(batch, latent_dim))
        assert recon.shape == (batch, out_channels, img_size, img_size)
        assert recon.min() >= 0.0 and recon.max() <= 1.0  # Sigmoid output


class TestVAE:
    """Full VAE: forward shapes, reparameterise, loss, backward."""

    @pytest.mark.parametrize(
        "latent_dim,batch,img_size",
        [
            (20, 2, 28),  # MNIST standard
            (4, 1, 28),  # small latent + single sample
        ],
    )
    def test_forward_shapes(self, latent_dim, batch, img_size):
        vae = VAE(latent_dim=latent_dim, img_size=img_size)
        recon, mu, logvar = vae(torch.randn(batch, 1, img_size, img_size))
        assert recon.shape == (batch, 1, img_size, img_size)
        assert mu.shape == (batch, latent_dim)
        assert logvar.shape == (batch, latent_dim)

    def test_reparameterize_adds_noise(self):
        """z differs from mu when logvar=0 (unit noise added)."""
        vae = VAE(latent_dim=20)
        mu = torch.zeros(4, 20)
        logvar = torch.zeros(4, 20)  # σ = 1
        z = vae.reparameterize(mu, logvar)
        assert z.shape == (4, 20)
        assert not torch.allclose(z, mu, atol=1e-3)  # noise injected

    def test_loss_contract(self):
        """Loss dict has right keys, positive values, KL=0 when mu=logvar=0."""
        vae = VAE(latent_dim=20)
        x = torch.rand(2, 1, 28, 28)  # uniform [0,1] — valid for BCE
        recon, mu, logvar = vae(x)
        losses = vae.loss_function(recon, x, mu, logvar)

        assert isinstance(losses, dict)
        assert set(losses.keys()) == {"loss", "recon_loss", "kl_loss"}
        assert losses["loss"].item() > 0
        assert losses["recon_loss"].item() > 0

        # KL divergence: μ=0, log σ²=0 → KL = ½ Σ(0 + 1 - 0 - 1) = 0
        losses_zero = vae.loss_function(
            recon, x, torch.zeros(2, 20), torch.zeros(2, 20)
        )
        assert losses_zero["kl_loss"].item() < 1e-6

    def test_gradient_flows(self):
        """Loss backward reaches all parameters in encoder and decoder."""
        vae = VAE(latent_dim=20)
        x = torch.rand(2, 1, 28, 28)  # uniform [0,1] — valid for BCE
        recon, mu, logvar = vae(x)
        vae.loss_function(recon, x, mu, logvar)["loss"].backward()
        for name, param in vae.named_parameters():
            assert param.grad is not None, f"{name} has no grad"
            assert param.grad.shape == param.shape
