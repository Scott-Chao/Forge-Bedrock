"""
tests/transformer/test_attention.py — Tests for attention mechanisms.
"""

import pytest
import torch
from core.transformer.attention import (
    MultiHeadAttention,
    _create_causal_mask,
    scaled_dot_product_attention,
)


class TestCreateCausalMask:
    """Tests for the causal mask helper."""

    def test_shape(self):
        """Mask should be (seq_len, seq_len) for any seq_len."""
        for seq_len in [1, 2, 4, 16]:
            mask = _create_causal_mask(seq_len)
            assert mask.shape == (seq_len, seq_len), (
                f"shape mismatch for seq_len={seq_len}"
            )
            assert mask.dtype == torch.bool, f"mask should be boolean, got {mask.dtype}"

    def test_causal_property(self):
        """Position i should be able to attend to positions j <= i only."""
        mask = _create_causal_mask(4)
        # i >= j → True (allowed)
        assert mask[2, 0].item() is True  # row 3, col 1 — past, should be allowed
        assert mask[2, 1].item() is True  # row 3, col 2 — past
        assert mask[2, 2].item() is True  # row 3, col 3 — current
        # i < j → False (forbidden)
        assert mask[1, 2].item() is False  # row 2, col 3 — future, should be forbidden
        assert mask[0, 3].item() is False


class TestScaledDotProductAttention:
    """Tests for the core attention mechanism."""

    @pytest.fixture
    def simple_inputs(self):
        """Create small deterministic inputs for testing."""
        batch, seq_len, d_k = 2, 4, 8
        q = torch.randn(batch, seq_len, d_k)
        k = torch.randn(batch, seq_len, d_k)
        v = torch.randn(batch, seq_len, d_k)
        return q, k, v

    def test_output_shape(self, simple_inputs):
        """Output should have the same shape as V."""
        q, k, v = simple_inputs
        out = scaled_dot_product_attention(q, k, v)
        assert out.shape == v.shape, f"output shape {out.shape} != expected {v.shape}"

    def test_attention_weights_sum_to_one(self):
        """With V=ones, output should also be ones (weights sum to 1)."""
        q = torch.randn(1, 1, 4)
        k = torch.randn(1, 4, 4)
        v_ones = torch.ones(1, 4, 4)
        out = scaled_dot_product_attention(q, k, v_ones)
        assert torch.allclose(out, torch.ones_like(out), atol=1e-5), (
            "attention with V=ones should output ones"
        )

    def test_causal_mask_prevents_looking_ahead(self):
        """With causal mask, position i should not depend on future inputs.

        We can test this by zeroing out the value at position j > i and
        checking that the output at position i is unchanged.
        """
        batch, seq_len, d_k = 1, 4, 8
        q = torch.randn(batch, seq_len, d_k)
        k = torch.randn(batch, seq_len, d_k)
        v = torch.randn(batch, seq_len, d_k)

        mask = _create_causal_mask(seq_len)
        out_masked = scaled_dot_product_attention(q, k, v, mask=mask)

        # Zero out the value at the last position
        v_zeroed = v.clone()
        v_zeroed[:, -1, :] = 0

        out_zeroed = scaled_dot_product_attention(q, k, v_zeroed, mask=mask)

        # First position unaffected (can't attend to last position under causal mask)
        assert torch.allclose(out_masked[:, 0, :], out_zeroed[:, 0, :], atol=1e-6), (
            "position 0 should not depend on position 3 under causal mask"
        )
        # The second-to-last position should also be unaffected
        assert torch.allclose(out_masked[:, 2, :], out_zeroed[:, 2, :], atol=1e-6), (
            "position 2 should not depend on position 3 under causal mask"
        )

    def test_different_d_k_and_d_v(self):
        """d_v may differ from d_k (e.g., when using multi-head attention)."""
        batch, seq_len, d_k, d_v = 2, 4, 8, 16
        q = torch.randn(batch, seq_len, d_k)
        k = torch.randn(batch, seq_len, d_k)
        v = torch.randn(batch, seq_len, d_v)
        out = scaled_dot_product_attention(q, k, v)
        assert out.shape[-1] == d_v, (
            f"output last dim {out.shape[-1]} != expected {d_v}"
        )
        assert out.shape[-2] == seq_len

    def test_batch_independence(self):
        """Outputs for different batch elements should only depend on own inputs."""
        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 4, 8)
        # Make the two batch elements clearly different
        v = torch.zeros(2, 4, 8)
        v[0, :, :] = 1.0
        v[1, :, :] = -1.0

        out = scaled_dot_product_attention(q, k, v)

        # Each batch element's output should be different (opposite values)
        assert not torch.allclose(out[0], out[1]), (
            "batch elements with different V should produce different outputs"
        )


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention module."""

    def test_construction(self):
        """MHA should initialise with the expected projections."""
        d_model, n_heads = 64, 8
        mha = MultiHeadAttention(d_model, n_heads)
        assert mha.d_model == d_model
        assert mha.n_heads == n_heads
        assert mha.d_k == d_model // n_heads  # = 8
        # Linear projections
        for name in ["w_q", "w_k", "w_v", "w_o"]:
            w = getattr(mha, name)
            assert w.in_features == d_model, f"{name}.in_features != d_model"
            assert w.out_features == d_model, f"{name}.out_features != d_model"
            assert isinstance(w.bias, torch.Tensor), (
                f"{name} should have bias by default"
            )

    def test_output_shape_and_type(self):
        """Output should match (batch, seq, d_model)."""
        batch, seq_len, d_model, n_heads = 2, 8, 64, 8
        mha = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch, seq_len, d_model)
        out = mha(x, x, x)
        assert out.shape == (batch, seq_len, d_model), f"shape mismatch: {out.shape}"

    def test_n_heads_must_divide_d_model(self):
        """d_model must be divisible by n_heads."""
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttention(d_model=64, n_heads=7)

    def test_multihead_not_identity(self):
        """MHA should transform the input, not pass it through unchanged."""
        batch, seq_len, d_model, n_heads = 1, 4, 16, 4
        mha = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch, seq_len, d_model)
        out = mha(x, x, x)
        # Output should NOT be identical to input (projections change things)
        assert not torch.allclose(out, x, atol=1e-4), (
            "MHA should transform the input, not pass it through"
        )

    def test_different_qkv(self):
        """MHA should work with different Q, K, V (cross-attention scenario)."""
        batch, q_len, kv_len, d_model, n_heads = 2, 6, 10, 32, 4
        mha = MultiHeadAttention(d_model, n_heads)
        q = torch.randn(batch, q_len, d_model)
        k = torch.randn(batch, kv_len, d_model)
        v = torch.randn(batch, kv_len, d_model)
        out = mha(q, k, v)
        # Output length should match query length (not key/value length)
        assert out.shape == (batch, q_len, d_model), (
            f"cross-attention shape mismatch: {out.shape}"
        )

    def test_causal_mask(self):
        """With causal mask, position i can't attend to future positions."""
        batch, seq_len, d_model, n_heads = 2, 6, 32, 4
        mha = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch, seq_len, d_model)
        mask = _create_causal_mask(seq_len)

        out_allowed = mha(x, x, x, mask=mask)

        # Zero out the last token's value — it should only affect the last position
        x_zeroed = x.clone()
        x_zeroed[:, -1, :] = 0.0
        out_zeroed = mha(x_zeroed, x_zeroed, x_zeroed, mask=mask)

        # Position 0 should be unaffected
        assert torch.allclose(out_allowed[:, 0, :], out_zeroed[:, 0, :], atol=1e-6), (
            "position 0 must not depend on position 5 under causal mask"
        )

    def test_no_causal_mask_cross_talk(self):
        """Without mask, all positions can attend to all others."""
        batch, seq_len, d_model, n_heads = 2, 4, 32, 4
        mha = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch, seq_len, d_model)

        out_normal = mha(x, x, x)

        # Zero out last position
        x_zeroed = x.clone()
        x_zeroed[:, -1, :] = 0.0
        out_zeroed = mha(x_zeroed, x_zeroed, x_zeroed)

        # Without mask, position 0 SHOULD be affected (it attends to all positions)
        # Note: difference may be small at init but should be nonzero
        assert not torch.allclose(
            out_normal[:, 0, :], out_zeroed[:, 0, :], atol=1e-6
        ), "position 0 should depend on position 3 when no causal mask"

    def test_multiple_heads_produce_distinct_outputs(self):
        """Different heads should learn different patterns (after training).

        Even at init, different random projections mean different outputs
        per head — test that the concatenation actually spans multi-head.
        We verify by checking that d_model > per-head dimension output
        preserves information from all heads.
        """
        batch, seq_len, d_model, n_heads = 1, 2, 8, 4
        mha = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch, seq_len, d_model)
        out = mha(x, x, x)
        # Should have non-trivial output (not all zeros)
        assert out.abs().sum().item() > 0, "output should not be all zeros"
        # Output should not be just a single repeated pattern across the d_model dim
        assert not torch.allclose(
            out[0, 0, : d_model // n_heads],
            out[0, 0, d_model // n_heads : 2 * d_model // n_heads],
            atol=1e-4,
        ), "different heads should not produce identical outputs at init"
