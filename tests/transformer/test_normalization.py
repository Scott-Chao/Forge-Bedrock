"""
tests/transformer/test_normalization.py — Tests for RMSNorm.
"""

import pytest
import torch
from core.transformer.normalization import RMSNorm


class TestRMSNorm:
    """Tests for RMSNorm layer."""

    @pytest.fixture
    def simple_input(self):
        """Small deterministic input for testing."""
        return torch.randn(2, 4, 16)  # (batch, seq, d_model)

    def test_construction(self):
        """RMSNorm should initialise gamma as ones."""
        d_model = 32
        rms = RMSNorm(d_model)
        assert rms.d_model == d_model
        assert rms.gamma.shape == (d_model,)
        assert torch.allclose(rms.gamma, torch.ones(d_model)), (
            "gamma should be initialised to ones"
        )

    def test_output_shape(self, simple_input):
        """Output should have the same shape as input."""
        d_model = simple_input.size(-1)
        rms = RMSNorm(d_model)
        out = rms(simple_input)
        assert out.shape == simple_input.shape, (
            f"output shape {out.shape} != input shape {simple_input.shape}"
        )

    def test_unit_rms_after_norm(self):
        """After RMSNorm, each token vector should have RMS ≈ 1."""
        batch, seq_len, d_model = 2, 4, 16
        rms = RMSNorm(d_model)
        x = torch.randn(batch, seq_len, d_model)
        out = rms(x)

        # Compute RMS per token: sqrt(mean(out^2))
        rms_values = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms_values, torch.ones_like(rms_values), atol=1e-5), (
            f"RMS should be ~1, got range "
            f"[{rms_values.min():.6f}, {rms_values.max():.6f}]"
        )

    def test_gamma_scaling(self):
        """Scaling gamma by 2 should double the output."""
        d_model = 8
        x = torch.randn(1, 3, d_model)

        rms_orig = RMSNorm(d_model)
        out_orig = rms_orig(x)

        # Manually double gamma
        rms_scaled = RMSNorm(d_model)
        rms_scaled.gamma.data = rms_scaled.gamma * 2.0
        out_scaled = rms_scaled(x)

        assert torch.allclose(out_scaled, out_orig * 2.0, atol=1e-6), (
            "doubling gamma should double the output"
        )

    def test_eps_prevents_division_by_zero(self):
        """With eps > 0, even all-zero input should produce a finite output."""
        d_model = 16
        rms = RMSNorm(d_model, eps=1e-6)
        x = torch.zeros(1, 1, d_model)
        out = rms(x)
        assert torch.all(torch.isfinite(out)), "output should be finite for zero input"
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), (
            "zero input with zero gamma should give ... hmm, "
            "check: x=0, RMS=sqrt(0+eps)=sqrt(eps)"
        )

    def test_per_token_independence(self):
        """Each token should be normalised independently of other tokens."""
        d_model = 8
        rms = RMSNorm(d_model)
        # Make two tokens: one all-ones, one all-zeros
        x = torch.tensor([[[1.0] * d_model, [0.0] * d_model]])
        out = rms(x)

        # Token 0 (all ones): RMS = sqrt(mean(1^2)+eps) = sqrt(1+eps) ≈ 1
        # Normalised: 1 / 1 = 1 → after gamma (=1): 1
        # Actually wait: if all values are 1, mean(x^2) = 1, RMS = sqrt(1+eps)
        # So out = gamma * 1/sqrt(1+eps) ≈ 1 (slightly less)
        expected_tok_0 = torch.ones(d_model) / torch.sqrt(torch.tensor(1.0 + rms.eps))
        assert torch.allclose(out[0, 0], expected_tok_0, atol=1e-5), (
            "all-ones should normalise to 1/sqrt(1+eps)"
        )

        # Token 1 (all zeros): x=0, RMS = sqrt(0+eps) = sqrt(eps)
        # Normalised: 0 / sqrt(eps) = 0 → after gamma: 0
        expected_tok_1 = torch.zeros(d_model)
        assert torch.allclose(out[0, 1], expected_tok_1, atol=1e-6), (
            "all-zeros should give zero output"
        )

    def test_different_d_models(self):
        """RMSNorm should work with different d_model values."""
        for d_model in [4, 8, 16, 64, 128, 256]:
            rms = RMSNorm(d_model)
            x = torch.randn(2, 4, d_model)
            out = rms(x)
            assert out.shape == x.shape, f"shape mismatch for d_model={d_model}"
            # Check RMS ≈ 1
            rms_val = out.pow(2).mean(dim=-1).sqrt()
            assert torch.allclose(rms_val, torch.ones_like(rms_val), atol=1e-5), (
                f"RMS not ~1 for d_model={d_model}"
            )

    def test_forward_returns_float32(self):
        """Output dtype should match input dtype (typically float32)."""
        d_model = 16
        rms = RMSNorm(d_model)
        x = torch.randn(1, 1, d_model)
        out = rms(x)
        assert out.dtype == x.dtype, (
            f"output dtype {out.dtype} != input dtype {x.dtype}"
        )

    def test_batch_independence(self):
        """Different batch elements should be normalised independently."""
        d_model = 8
        rms = RMSNorm(d_model)
        # Two batch elements: one positive, one negative
        x = torch.stack(
            [
                torch.ones(1, d_model) * 5.0,  # batch 0: all 5s
                torch.ones(1, d_model) * (-3.0),  # batch 1: all -3s
            ]
        )
        out = rms(x)

        # Both should have RMS ≈ 1 with gamma=1
        for b in range(2):
            rms_val = out[b].pow(2).mean().sqrt()
            assert torch.allclose(rms_val, torch.ones(1), atol=1e-5), (
                f"batch {b} RMS should be ~1, got {rms_val.item():.6f}"
            )
