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


class TestGQA:
    """Tests for Grouped-Query Attention extension of MultiHeadAttention."""

    @pytest.mark.parametrize(
        "d_model, n_heads, n_kv_heads, expected_kv_out",
        [
            (64, 8, 4, 32),  # 8 heads, 4 KV heads → kv_dim = 8*4 = 32
            (64, 8, 2, 16),  # 8 heads, 2 KV heads → kv_dim = 8*2 = 16
            (64, 8, 8, 64),  # n_kv_heads = n_heads → kv_dim = d_model (MHA mode)
            (64, 8, None, 64),  # None defaults to n_heads → kv_dim = d_model
        ],
    )
    def test_kv_projection_dims(self, d_model, n_heads, n_kv_heads, expected_kv_out):
        """W_k and W_v should project to n_kv_heads * d_k, not d_model."""
        mha = MultiHeadAttention(d_model, n_heads, n_kv_heads=n_kv_heads)
        assert mha.w_k.out_features == expected_kv_out, (
            f"w_k.out_features = {mha.w_k.out_features}, expected {expected_kv_out}"
        )
        assert mha.w_v.out_features == expected_kv_out, (
            f"w_v.out_features = {mha.w_v.out_features}, expected {expected_kv_out}"
        )
        # n_groups should be correct
        expected_groups = n_heads // (n_kv_heads or n_heads)
        assert mha.n_groups == expected_groups, (
            f"n_groups = {mha.n_groups}, expected {expected_groups}"
        )

    @pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
    def test_gqa_output_shape(self, n_kv_heads):
        """GQA forward should produce the same output shape as MHA."""
        batch, seq_len, d_model, n_heads = 2, 6, 64, 8
        mha = MultiHeadAttention(d_model, n_heads, n_kv_heads=n_kv_heads)
        x = torch.randn(batch, seq_len, d_model)
        out, (k_new, v_new) = mha(x, x, x)
        assert out.shape == (batch, seq_len, d_model), (
            f"output shape mismatch: {out.shape}"
        )
        # KV cache should store n_kv_heads, not n_heads
        assert k_new.shape[1] == n_kv_heads, (
            f"k_new has {k_new.shape[1]} heads, expected {n_kv_heads}"
        )
        assert v_new.shape[1] == n_kv_heads, (
            f"v_new has {v_new.shape[1]} heads, expected {n_kv_heads}"
        )

    def test_gqa_with_kv_cache(self):
        """GQA should work correctly with KV cache (past_kv with n_kv_heads)."""
        batch, seq_len, d_model, n_heads, n_kv_heads = 2, 4, 32, 8, 2
        mha = MultiHeadAttention(d_model, n_heads, n_kv_heads=n_kv_heads)
        d_k = d_model // n_heads  # 4
        x = torch.randn(batch, seq_len, d_model)

        # First call: no cache
        out1, (k_new, v_new) = mha(x, x, x)
        assert k_new.shape == (batch, n_kv_heads, seq_len, d_k)

        # Second call: feed a single new token with past cache
        x_new = torch.randn(batch, 1, d_model)
        out2, (k_new2, v_new2) = mha(x_new, x_new, x_new, past_kv=(k_new, v_new))
        assert k_new2.shape == (batch, n_kv_heads, 1, d_k)
        # Output should still be valid
        assert out2.shape == (batch, 1, d_model)

    def test_gqa_fewer_parameters_than_mha(self):
        """GQA should save parameters in W_k and W_v."""
        d_model, n_heads, n_kv_heads = 64, 8, 4
        mha = MultiHeadAttention(d_model, n_heads, n_kv_heads=n_kv_heads)
        full_mha = MultiHeadAttention(d_model, n_heads, n_kv_heads=n_heads)

        kv_params_gqa = sum(p.numel() for p in [mha.w_k.weight, mha.w_v.weight])
        kv_params_full = sum(
            p.numel() for p in [full_mha.w_k.weight, full_mha.w_v.weight]
        )
        assert kv_params_gqa < kv_params_full, (
            f"GQA KV params ({kv_params_gqa}) should be less than MHA ({kv_params_full})"
        )
        # Q and O projections should be the same
        assert mha.w_q.weight.shape == full_mha.w_q.weight.shape

    def test_gqa_invalid_n_kv_heads(self):
        """n_heads must be divisible by n_kv_heads."""
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttention(d_model=64, n_heads=8, n_kv_heads=3)


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
        out, _ = mha(x, x, x)
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
        out, _ = mha(x, x, x)
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
        out = mha(q, k, v)[0]
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

        out_allowed, _ = mha(x, x, x, mask=mask)

        # Zero out the last token's value — it should only affect the last position
        x_zeroed = x.clone()
        x_zeroed[:, -1, :] = 0.0
        out_zeroed = mha(x_zeroed, x_zeroed, x_zeroed, mask=mask)[0]
        # Position 0 should be unaffected
        assert torch.allclose(out_allowed[:, 0, :], out_zeroed[:, 0, :], atol=1e-6), (
            "position 0 must not depend on position 5 under causal mask"
        )

    def test_no_causal_mask_cross_talk(self):
        """Without mask, all positions can attend to all others."""
        batch, seq_len, d_model, n_heads = 2, 4, 32, 4
        mha = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch, seq_len, d_model)

        out_normal, _ = mha(x, x, x)

        # Zero out last position
        x_zeroed = x.clone()
        x_zeroed[:, -1, :] = 0.0
        out_zeroed, _ = mha(x_zeroed, x_zeroed, x_zeroed)

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
        out, _ = mha(x, x, x)
        # Should have non-trivial output (not all zeros)
        assert out.abs().sum().item() > 0, "output should not be all zeros"
        # Output should not be just a single repeated pattern across the d_model dim
        assert not torch.allclose(
            out[0, 0, : d_model // n_heads],
            out[0, 0, d_model // n_heads : 2 * d_model // n_heads],
            atol=1e-4,
        ), "different heads should not produce identical outputs at init"
