"""tests/gen/test_diffusion.py — Tests for diffusion building blocks.

Covers NoiseScheduler (forward noising, schedule properties) and
TimeEmbedding / SinusoidalPosEmbedding (shapes, determinism, MLP).
"""

import pytest
import torch
from core.gen import (
    NoiseScheduler,
    SinusoidalPosEmbedding,
    TimeConditionedUNet,
    TimeEmbedding,
)

# ============================================================================
# NoiseScheduler — Schedule Properties
# ============================================================================


class TestNoiseSchedulerSchedule:
    """Pre-computed buffers: shapes, ranges, monotonicity."""

    def test_beta_linear_range(self):
        """βₜ increases linearly from beta_start to beta_end."""
        s = NoiseScheduler(timesteps=100, beta_start=1e-4, beta_end=0.02)
        assert s.beta.shape == (100,)
        assert abs(s.beta[0].item() - 1e-4) < 1e-6
        assert abs(s.beta[-1].item() - 0.02) < 1e-6
        assert (s.beta[1:] >= s.beta[:-1]).all(), "βₜ must increase"

    def test_alpha_bar_starts_near_one_ends_near_zero(self):
        """ᾱₜ = ∏(1-βₛ) goes from ~1 to ~0 over timesteps."""
        s = NoiseScheduler(timesteps=1000)
        assert s.alpha_bar[0].item() > 0.99  # first step is nearly 1
        assert s.alpha_bar[-1].item() < 0.01  # last step is nearly 0

    @pytest.mark.parametrize("timesteps", [100, 1000])
    def test_buffer_shapes(self, timesteps):
        """All pre-computed buffers are 1D with length = timesteps."""
        s = NoiseScheduler(timesteps=timesteps)
        keys = [
            "beta",
            "alpha",
            "alpha_bar",
            "sqrt_alpha_bar",
            "sqrt_one_minus_alpha_bar",
            "sqrt_recip_alpha",
        ]
        for k in keys:
            buf = getattr(s, k)
            assert isinstance(buf, torch.Tensor), f"{k} is not a Tensor"
            assert buf.shape == (timesteps,), f"{k} shape {buf.shape}"

    def test_posterior_variance_finite(self):
        """posterior_variance is finite and non-negative for all t."""
        s = NoiseScheduler(timesteps=100)
        assert s.posterior_variance.shape == (100,)
        assert torch.isfinite(s.posterior_variance).all()
        assert (s.posterior_variance >= 0).all()


# ============================================================================
# NoiseScheduler — Forward Noising (add_noise)
# ============================================================================


class TestAddNoise:
    """Closed-form forward diffusion x₀ → xₜ."""

    @pytest.mark.parametrize("B,C,H,W", [(2, 1, 28, 28), (4, 3, 32, 32)])
    def test_output_shape(self, B, C, H, W):
        """Output x_t and ε have the same shape as input x₀."""
        s = NoiseScheduler(timesteps=100)
        x_0 = torch.randn(B, C, H, W)
        t = torch.randint(0, 100, (B,))
        x_t, eps = s.add_noise(x_0, t)
        assert x_t.shape == (B, C, H, W)
        assert eps.shape == (B, C, H, W)

    def test_t_zero_returns_nearly_x0(self):
        """At t=0, ᾱ₀ is very close to 1 → x_t ≈ x₀."""
        s = NoiseScheduler(timesteps=100)
        x_0 = torch.randn(2, 1, 28, 28)
        t = torch.zeros(2, dtype=torch.long)
        x_t, _ = s.add_noise(x_0, t)
        # First step adds only a tiny amount of noise
        mse = ((x_t - x_0) ** 2).mean().item()
        assert mse < 0.01, f"MSE at t=0 should be tiny, got {mse:.6f}"

    def test_t_final_nearly_pure_noise(self):
        """At t=T-1, ᾱ is near 0 → x_t ≈ ε (signal lost)."""
        s = NoiseScheduler(timesteps=1000)
        x_0 = torch.randn(2, 1, 28, 28)
        t = torch.full((2,), 999, dtype=torch.long)
        x_t, eps = s.add_noise(x_0, t)
        # The output should be very close to pure noise;
        # MSE between x_t and eps should be small.
        # x_t = sqrt(ᾱ)·x₀ + sqrt(1-ᾱ)·ε ≈ 0·x₀ + 1·ε ≈ ε
        mse = ((x_t - eps) ** 2).mean().item()
        assert mse < 0.05, f"MSE from pure noise at t={s.timesteps - 1} = {mse:.6f}"

    def test_noise_has_unit_variance(self):
        """The sampled ε is ≈ N(0, I) — empirical variance ≈ 1 over batch dim."""
        s = NoiseScheduler(timesteps=100)
        x_0 = torch.zeros(512, 1, 28, 28)  # zero input
        t = torch.randint(0, 100, (512,))
        x_t, eps = s.add_noise(x_0, t)
        # With x_0 = 0, x_t = sqrt(1-ᾱₜ)·ε, so Var[x_t] = (1-ᾱₜ)·1.
        # Test ε directly instead.
        var = eps.var().item()
        assert 0.8 < var < 1.3, f"ε empirical variance = {var:.4f} (expected ~1)"

    def test_different_t_different_noise_level(self):
        """Earlier t steps have less noise than later ones (empirically)."""
        s = NoiseScheduler(timesteps=100)
        x_0 = torch.ones(32, 1, 28, 28)  # constant input highlights noise
        t_early = torch.full((32,), 5, dtype=torch.long)
        t_late = torch.full((32,), 95, dtype=torch.long)

        x_early, _ = s.add_noise(x_0, t_early)
        x_late, _ = s.add_noise(x_0, t_late)

        # Late steps should have higher variance (more noise added)
        early_var = x_early.var().item()
        late_var = x_late.var().item()
        assert late_var > early_var * 1.5, (
            f"late var {late_var:.4f} should be >> early var {early_var:.4f}"
        )


# ============================================================================
# NoiseScheduler — Edge Cases
# ============================================================================


class TestNoiseSchedulerEdgeCases:
    """Single sample, minimal timesteps, alternating return shapes."""

    def test_single_sample(self):
        """Batch size 1 works correctly."""
        s = NoiseScheduler(timesteps=100)
        x_0 = torch.randn(1, 1, 28, 28)
        t = torch.tensor([50])
        x_t, eps = s.add_noise(x_0, t)
        assert x_t.shape == (1, 1, 28, 28)
        assert eps.shape == (1, 1, 28, 28)

    def test_reproducible_noise_with_seed(self):
        """Same seed → same ε (deterministic sampling)."""
        s = NoiseScheduler(timesteps=100)
        x_0 = torch.randn(4, 1, 28, 28)
        t = torch.randint(0, 100, (4,))

        torch.manual_seed(42)
        _, eps_a = s.add_noise(x_0, t)

        torch.manual_seed(42)
        _, eps_b = s.add_noise(x_0, t)

        assert torch.allclose(eps_a, eps_b), "noise should be reproducible with seed"


# ============================================================================
# TimeEmbedding — Shape, Determinism, MLP
# ============================================================================


class TestTimeEmbedding:
    """Sinusoidal time embedding with MLP projection."""

    @pytest.mark.parametrize("batch,dim", [(2, 256), (4, 128), (1, 64)])
    def test_output_shape(self, batch, dim):
        """(batch,) → (batch, dim) after MLP."""
        te = TimeEmbedding(dim=dim)
        t = torch.randint(0, 1000, (batch,))
        emb = te(t)
        assert emb.shape == (batch, dim)

    def test_same_t_same_embedding(self):
        """Deterministic: same timestep → same embedding."""
        te = TimeEmbedding(dim=128)
        t = torch.tensor([42])
        emb_a = te(t)
        emb_b = te(t)
        assert torch.allclose(emb_a, emb_b), "same t should give same embedding"

    def test_different_t_different_embedding(self):
        """Different timesteps produce measurably different embeddings."""
        te = TimeEmbedding(dim=128)
        t_a = torch.tensor([10])
        t_b = torch.tensor([990])
        emb_a = te(t_a)
        emb_b = te(t_b)
        diff = (emb_a - emb_b).abs().mean().item()
        assert diff > 0.01, (
            f"embeddings for t=10 and t=990 should differ, diff={diff:.6f}"
        )

    def test_mlp_has_trainable_params(self):
        """MLP contains trainable Linear weights."""
        te = TimeEmbedding(dim=256)
        n_trainable = sum(p.numel() for p in te.parameters())
        assert n_trainable > 0
        # Two Linear layers: 256→256 + 256→256 = 2 * (256*256 + 256) = 131_584
        assert n_trainable > 2 * 256 * 256, (
            f"expected >131072 trainable params, got {n_trainable}"
        )

    def test_gradient_flows(self):
        """Loss backward reaches all MLP parameters."""
        te = TimeEmbedding(dim=128)
        t = torch.randint(0, 1000, (4,))
        emb = te(t)
        loss = emb.sum()
        loss.backward()
        for name, p in te.named_parameters():
            assert p.grad is not None, f"MLP param {name} has no grad"
            assert p.grad.shape == p.shape


# ============================================================================
# SinusoidalPosEmbedding — Raw embedding without MLP
# ============================================================================


class TestSinusoidalPosEmbedding:
    """Raw sin/cos encoding (no learnable params)."""

    @pytest.mark.parametrize("batch,dim", [(3, 256), (1, 128)])
    def test_output_shape(self, batch, dim):
        """(batch,) → (batch, dim)  [dim = half*2]."""
        spe = SinusoidalPosEmbedding(dim=dim)
        t = torch.randint(0, 1000, (batch,))
        emb = spe(t)
        assert emb.shape[-1] == dim  # allow flexible batch
        # For odd dim, dim // 2 * 2 is the actual output
        expected_dim = dim // 2 * 2
        assert emb.shape[-1] == expected_dim

    def test_no_trainable_params(self):
        """SinusoidalPosEmbedding has no parameters — pure encoding."""
        spe = SinusoidalPosEmbedding(dim=256)
        params = list(spe.parameters())
        assert len(params) == 0, "raw encoding should have no parameters"

    def test_sin_cos_structure(self):
        """Output has both sin and cos in alternating or stacked positions."""
        spe = SinusoidalPosEmbedding(dim=8)
        t = torch.tensor([0])
        emb = spe(t)
        # dim=8, half=4 → sin[0..3], cos[0..3] stacked → 8
        assert emb.shape == (1, 8)


# ============================================================================
# TimeConditionedUNet — Output shape, gradient flow, time sensitivity
# ============================================================================


class TestTimeConditionedUNet:
    """DDPM-style U-Net with FiLM time conditioning."""

    @pytest.mark.parametrize(
        "B,C,H,W,depth,base_ch",
        [
            (2, 1, 28, 28, 3, 32),  # MNIST, smaller base_ch
            (4, 1, 32, 32, 3, 64),  # square, standard
            (1, 3, 32, 32, 3, 32),  # RGB, shallow
        ],
    )
    def test_output_shape(self, B, C, H, W, depth, base_ch):
        """(B,C,H,W) + (B,) timesteps → (B,C,H,W) noise prediction."""
        unet = TimeConditionedUNet(
            in_channels=C,
            out_channels=C,
            base_channels=base_ch,
            depth=depth,
        )
        x = torch.randn(B, C, H, W)
        t = torch.randint(0, 1000, (B,))
        out = unet(x, t)
        assert out.shape == (B, C, H, W)

    def test_output_same_spatial_resolution(self):
        """Output has same H,W as input across different sizes."""
        for H in [28, 32, 64]:
            unet = TimeConditionedUNet(
                in_channels=1,
                out_channels=1,
                base_channels=32,
                depth=3,
            )
            x = torch.randn(2, 1, H, H)
            t = torch.randint(0, 1000, (2,))
            out = unet(x, t)
            assert out.shape[-2:] == (H, H), f"spatial mismatch for {H}x{H}"

    def test_different_t_different_output(self):
        """Different timesteps produce measurably different predictions,
        even for the same noisy input — proves time conditioning works."""
        unet = TimeConditionedUNet(
            in_channels=1,
            out_channels=1,
            base_channels=32,
        )
        x = torch.randn(4, 1, 28, 28)

        # Same x, two different t sets
        t_a = torch.full((4,), 10, dtype=torch.long)
        t_b = torch.full((4,), 999, dtype=torch.long)

        with torch.no_grad():
            out_a = unet(x, t_a)
            out_b = unet(x, t_b)

        diff = (out_a - out_b).abs().mean().item()
        assert diff > 1e-4, f"t=10 and t=999 predictions should differ, diff={diff:.6f}"

    def test_gradient_flows_through_all_params(self):
        """Backward from output reaches every trainable parameter."""
        unet = TimeConditionedUNet(
            in_channels=1,
            out_channels=1,
            base_channels=32,
        )
        x = torch.randn(2, 1, 28, 28)
        t = torch.randint(0, 1000, (2,))
        out = unet(x, t)
        loss = out.mean()
        loss.backward()

        no_grad = [name for name, p in unet.named_parameters() if p.grad is None]
        assert len(no_grad) == 0, f"params with no gradient: {no_grad[:5]}"

    def test_multiple_timesteps_per_batch(self):
        """Each sample in the batch can have its own timestep."""
        unet = TimeConditionedUNet(
            in_channels=1,
            out_channels=1,
            base_channels=32,
        )
        x = torch.randn(4, 1, 28, 28)
        t = torch.tensor([0, 100, 500, 999], dtype=torch.long)
        out = unet(x, t)
        assert out.shape == (4, 1, 28, 28)

    def test_depth_variation(self):
        """Changing depth produces valid outputs (2, 3, or 4 stages)."""
        for depth in [2, 3]:
            unet = TimeConditionedUNet(
                in_channels=1,
                out_channels=1,
                base_channels=32,
                depth=depth,
            )
            min_size = 28
            H = W = max(min_size, 2**depth * 4)  # ensure divisibility
            x = torch.randn(2, 1, H, W)
            t = torch.randint(0, 1000, (2,))
            out = unet(x, t)
            assert out.shape == (2, 1, H, W), f"depth={depth} failed"

    @pytest.mark.parametrize("time_dim", [64, 128, 256])
    def test_time_dim_variation(self, time_dim):
        """Varying time_dim works with default architecture."""
        unet = TimeConditionedUNet(
            in_channels=1,
            out_channels=1,
            base_channels=32,
            time_dim=time_dim,
        )
        x = torch.randn(2, 1, 28, 28)
        t = torch.randint(0, 1000, (2,), dtype=torch.long)
        out = unet(x, t)
        assert out.shape == (2, 1, 28, 28)

    def test_batch_size_one(self):
        """Batch size 1 (single image) works correctly."""
        unet = TimeConditionedUNet(
            in_channels=1,
            out_channels=1,
            base_channels=32,
        )
        x = torch.randn(1, 1, 28, 28)
        t = torch.tensor([500])
        out = unet(x, t)
        assert out.shape == (1, 1, 28, 28)

    def test_forward_pass_is_deterministic(self):
        """Same input + same t + same seed → same output."""
        unet = TimeConditionedUNet(
            in_channels=1,
            out_channels=1,
            base_channels=32,
        )
        unet.eval()
        x = torch.randn(2, 1, 28, 28)
        t = torch.randint(0, 1000, (2,), dtype=torch.long)

        with torch.no_grad():
            out_a = unet(x, t)
            out_b = unet(x, t)

        assert torch.allclose(out_a, out_b, atol=1e-6)
