"""
tests/transformer/test_positional.py — Tests for RoPE (production split-half version).
"""

import pytest
import torch
from core.transformer.rope import (
    RotaryEmbedding,
    apply_rotary_emb,
    precompute_freqs_cis,
    rotate_half,
)


class TestPrecomputeFreqsCis:
    """Tests for the (max_seq_len, d_model) cos/sin table (split-half layout)."""

    def test_output_shapes(self):
        """cos and sin should be (max_seq_len, d_model)."""
        d_model, max_seq_len = 16, 8
        cos, sin = precompute_freqs_cis(d_model, max_seq_len)
        assert cos.shape == (max_seq_len, d_model), f"cos shape {cos.shape}"
        assert sin.shape == (max_seq_len, d_model), f"sin shape {sin.shape}"

    def test_even_d_models(self):
        """Should work for various even d_model values."""
        for d_model in [2, 8, 16, 32, 64]:
            cos, sin = precompute_freqs_cis(d_model, max_seq_len=4)
            assert cos.shape[-1] == d_model
            assert sin.shape[-1] == d_model

    def test_position_zero_is_identity(self):
        """At position 0: all cos=1, all sin=0."""
        d_model, max_seq_len = 8, 4
        cos, sin = precompute_freqs_cis(d_model, max_seq_len)
        assert torch.allclose(cos[0], torch.ones(d_model))
        assert torch.allclose(sin[0], torch.zeros(d_model))

    def test_split_half_layout(self):
        """First half of cos/sin should equal second half (split-half pairing).

        For d_model=8: cos[m] = [c₀, c₁, c₂, c₃, c₀, c₁, c₂, c₃]
        So cos[m][:4] == cos[m][4:]
        """
        d_model = 8
        cos, sin = precompute_freqs_cis(d_model, max_seq_len=4)
        for pos in range(4):
            assert torch.allclose(
                cos[pos, : d_model // 2], cos[pos, d_model // 2 :], atol=1e-7
            ), f"cos halves mismatch at position {pos}"
            assert torch.allclose(
                sin[pos, : d_model // 2], sin[pos, d_model // 2 :], atol=1e-7
            ), f"sin halves mismatch at position {pos}"

    def test_theta_exponential_decay(self):
        """Verify theta_i = base ** (-2i/d) pattern."""
        d_model = 8
        cos, _ = precompute_freqs_cis(d_model, max_seq_len=2)  # discard sin

        # At m=1: cos[1][i] = cos(theta_i)
        # So theta_i = arccos(cos[1][i])
        # Use first half (d//2 unique values)
        theta = torch.acos(cos[1, : d_model // 2].clamp(-1, 1))

        expected = 10000.0 ** (-2 * torch.arange(d_model // 2).float() / d_model)
        assert torch.allclose(theta, expected, atol=1e-4), (
            f"theta mismatch: {theta} vs {expected}"
        )


class TestRotateHalf:
    """Tests for the rotate_half helper."""

    def test_output_shape(self):
        x = torch.randn(2, 4, 8)
        y = rotate_half(x)
        assert y.shape == x.shape

    def test_specific_values(self):
        """rotate_half([a,b,c,d, e,f,g,h]) = [-e,-f,-g,-h, a,b,c,d]."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = rotate_half(x.unsqueeze(0)).squeeze(0)
        expected = torch.tensor([-5.0, -6.0, -7.0, -8.0, 1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(y, expected)

    def test_batched_tensor(self):
        x = torch.randn(3, 5, 2, 4, 16)
        y = rotate_half(x)
        assert y.shape == x.shape


class TestApplyRotaryEmb:
    """Tests for the rotate_half-based RoPE application."""

    def test_output_shape(self):
        batch, n_heads, seq_len, d_model = 2, 4, 6, 16
        x = torch.randn(batch, n_heads, seq_len, d_model)
        cos, sin = precompute_freqs_cis(d_model, max_seq_len=seq_len)
        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape

    def test_position_zero_no_change(self):
        """At position 0, cos=1, sin=0, so RoPE is identity."""
        batch, n_heads, seq_len, d_model = 1, 1, 4, 8
        x = torch.randn(batch, n_heads, seq_len, d_model)
        cos, sin = precompute_freqs_cis(d_model, max_seq_len=seq_len)
        out = apply_rotary_emb(x, cos, sin)
        assert torch.allclose(out[:, :, 0, :], x[:, :, 0, :], atol=1e-6)

    def test_preserves_norm(self):
        """Rotation is orthogonal — vector norm preserved."""
        batch, n_heads, seq_len, d_model = 2, 4, 6, 16
        x = torch.randn(batch, n_heads, seq_len, d_model)
        cos, sin = precompute_freqs_cis(d_model, max_seq_len=seq_len)
        out = apply_rotary_emb(x, cos, sin)
        assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-6)

    def test_manual_rotation(self):
        """Manually verify split-half pairing.

        For d=4, pairs are (0,2) and (1,3).
        Each pair (x[i], x[i+d/2]) is rotated by theta_i.

        For x = [a, b, c, d] at position m=1:
            theta₀ = 1.0, theta₁ = 10000^(-2/4) ≈ 0.1

            result[0] = a*cos(θ₀) - c*sin(θ₀)
            result[2] = a*sin(θ₀) + c*cos(θ₀)
            result[1] = b*cos(θ₁) - d*sin(θ₁)
            result[3] = b*sin(θ₁) + d*cos(θ₁)
        """
        d_model = 4
        cos, sin = precompute_freqs_cis(d_model, max_seq_len=3)

        # seq_len=2, so cos/sin for both positions 0 and 1 are used
        x = torch.tensor([[[[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]]])  # (1,1,2,4)

        c0, s0 = cos[1, 0].item(), sin[1, 0].item()  # θ₀ for pair (0,2) at m=1
        c1, s1 = cos[1, 1].item(), sin[1, 1].item()  # θ₁ for pair (1,3) at m=1

        out = apply_rotary_emb(x, cos[:2], sin[:2])

        # Position 0: identity (cos=1, sin=0)
        assert torch.allclose(out[0, 0, 0, :], x[0, 0, 0, :], atol=1e-6)

        # Position 1: apply rotation
        # Pair (x₀, x₂) = (1, 0) → rotated = (1*c0 - 0*s0, 1*s0 + 0*c0) = (c0, s0)
        assert torch.allclose(out[0, 0, 1, 0], torch.tensor(c0), atol=1e-6)
        assert torch.allclose(out[0, 0, 1, 2], torch.tensor(s0), atol=1e-6)

        # Pair (x₁, x₃) = (0, 1) → rotated = (0*c1 - 1*s1, 0*s1 + 1*c1) = (-s1, c1)
        assert torch.allclose(out[0, 0, 1, 1], torch.tensor(-s1), atol=1e-6)
        assert torch.allclose(out[0, 0, 1, 3], torch.tensor(c1), atol=1e-6)

    def test_relative_position_property(self):
        """KEY RoPE property: q_m @ k_n depends only on (n-m)."""
        d_model = 32
        seq_len = 6
        cos, sin = precompute_freqs_cis(d_model, max_seq_len=seq_len)

        # Same vector at every position
        q_vec = torch.randn(1, 1, 1, d_model)
        k_vec = torch.randn(1, 1, 1, d_model)
        q = q_vec.expand(1, 1, seq_len, -1).contiguous()
        k = k_vec.expand(1, 1, seq_len, -1).contiguous()

        q_rope = apply_rotary_emb(q, cos, sin)
        k_rope = apply_rotary_emb(k, cos, sin)
        scores = q_rope @ k_rope.transpose(-2, -1)

        for offset in range(1, seq_len - 1):
            v1 = scores[0, 0, 0, offset]
            v2 = scores[0, 0, 1, offset + 1]
            assert torch.allclose(v1, v2, atol=1e-6), (
                f"relative position property failed for offset={offset}"
            )


class TestRotaryEmbeddingModule:
    """Tests for the nn.Module wrapper."""

    def test_construction(self):
        d_model, max_seq_len = 16, 32
        rope = RotaryEmbedding(d_model, max_seq_len=max_seq_len)
        assert rope.cos.shape == (max_seq_len, d_model)
        assert rope.sin.shape == (max_seq_len, d_model)

    def test_buffers_not_parameters(self):
        rope = RotaryEmbedding(16, max_seq_len=32)
        assert len(list(rope.parameters())) == 0

    def test_forward_shape(self):
        batch, n_heads, seq_len, d_model = 2, 4, 8, 16
        rope = RotaryEmbedding(d_model, max_seq_len=seq_len + 16)
        x = torch.randn(batch, n_heads, seq_len, d_model)
        out = rope(x)
        assert out.shape == x.shape

    def test_forward_short_sequence(self):
        batch, n_heads, seq_len, d_model = 2, 4, 3, 16
        rope = RotaryEmbedding(d_model, max_seq_len=256)
        x = torch.randn(batch, n_heads, seq_len, d_model)
        out = rope(x)
        assert out.shape == x.shape

    def test_device_movement(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        rope = RotaryEmbedding(16, max_seq_len=32)
        rope = rope.to("cuda")
        assert rope.cos.device.type == "cuda"
        assert rope.sin.device.type == "cuda"
