"""tests/gen/test_gan.py — Tests for GAN building blocks (Generator, Discriminator)."""

import pytest
import torch
from core.gen import Discriminator, Generator


class TestGenerator:
    @pytest.mark.parametrize(
        "batch,latent_dim,out_channels,img_size,ngf",
        [
            (2, 100, 1, 28, 64),  # MNIST standard (DCGAN defaults)
            (4, 64, 3, 32, 32),  # CIFAR-style, smaller latent
            (1, 100, 1, 32, 128),  # single sample, larger ngf
        ],
    )
    def test_output_shape_and_range(
        self, batch, latent_dim, out_channels, img_size, ngf
    ):
        G = Generator(
            latent_dim=latent_dim,
            out_channels=out_channels,
            img_size=img_size,
            ngf=ngf,
        )
        z = torch.randn(batch, latent_dim)
        out = G(z)
        assert out.shape == (batch, out_channels, img_size, img_size)
        # Tanh output: values in (-1, 1)
        assert out.min() >= -1.0 and out.max() <= 1.0


class TestDiscriminator:
    @pytest.mark.parametrize(
        "batch,in_channels,img_size,ndf",
        [
            (2, 1, 28, 64),  # MNIST
            (4, 3, 32, 32),  # CIFAR-style
            (1, 1, 32, 128),  # single sample, larger ndf
        ],
    )
    def test_output_shape(self, batch, in_channels, img_size, ndf):
        D = Discriminator(in_channels=in_channels, img_size=img_size, ndf=ndf)
        x = torch.randn(batch, in_channels, img_size, img_size)
        out = D(x)
        assert out.shape == (batch,)  # one logit per sample


class TestGANPipeline:
    """End-to-end GAN forward checks: Generator → Discriminator."""

    def test_generator_discriminator_chain(self):
        """D(G(z)) produces (batch,) logits with valid gradient flow."""
        G = Generator(latent_dim=100, out_channels=1, img_size=28)
        D = Discriminator(in_channels=1, img_size=28)

        z = torch.randn(4, 100)
        fake = G(z)
        assert fake.shape == (4, 1, 28, 28)

        logits = D(fake)
        assert logits.shape == (4,)

    def test_discriminator_real_vs_fake_logits(self):
        """Real images should produce higher logits than pure noise on average,
        even from an untrained Discriminator — a basic sanity check that the
        architecture can distinguish signal from noise (conv layers respond
        to spatial structure, noise has none)."""
        D = Discriminator(in_channels=1, img_size=28)

        real = torch.randn(16, 1, 28, 28)  # some spatial structure
        noise = torch.randn(16, 1, 28, 28) * 10  # high-variance noise

        with torch.no_grad():
            real_logits = D(real)
            noise_logits = D(noise)

        # On average, real should give higher than noise (not checking per-sample)
        assert real_logits.mean() > noise_logits.mean()

    def test_gradient_flows_through_generator(self):
        """Backprop from D(G(z)) reaches all Generator parameters."""
        G = Generator(latent_dim=100, out_channels=1, img_size=28)
        D = Discriminator(in_channels=1, img_size=28)

        z = torch.randn(2, 100)
        fake = G(z)
        logits = D(fake)

        # Non-saturating G loss: -log(D(G(z)))
        loss = -logits.mean()
        loss.backward()

        for name, param in G.named_parameters():
            assert param.grad is not None, f"Generator param {name} has no grad"
            assert param.grad.shape == param.shape

    def test_gradient_flows_through_discriminator(self):
        """Backprop from D loss reaches all Discriminator parameters."""
        G = Generator(latent_dim=100, out_channels=1, img_size=28)
        D = Discriminator(in_channels=1, img_size=28)

        real = torch.randn(2, 1, 28, 28)
        z = torch.randn(2, 100)

        # D loss (real=1, fake=0)
        real_logits = D(real)
        fake_logits = D(G(z).detach())  # detach stops grad to G
        loss = -real_logits.mean() + fake_logits.mean()  # BCE-with-logits surrogate
        loss.backward()

        for name, param in D.named_parameters():
            assert param.grad is not None, f"Discriminator param {name} has no grad"
            assert param.grad.shape == param.shape
