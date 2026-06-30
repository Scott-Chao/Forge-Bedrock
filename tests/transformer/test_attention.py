"""
tests/transformer/test_attention.py — Tests for attention mechanisms.
"""

import pytest
import torch

from core.transformer.attention import (
    scaled_dot_product_attention,
    _create_causal_mask,
)


class TestCreateCausalMask:
    """Tests for the causal mask helper."""

    def test_shape(self):
        """Mask should be (seq_len, seq_len) for any seq_len."""
        for seq_len in [1, 2, 4, 16]:
            mask = _create_causal_mask(seq_len)
            assert mask.shape == (seq_len, seq_len), f"shape mismatch for seq_len={seq_len}"
            assert mask.dtype == torch.bool, f"mask should be boolean, got {mask.dtype}"

    def test_causal_property(self):
        """Position i should be able to attend to positions j <= i only."""
        mask = _create_causal_mask(4)
        # i >= j → True (allowed)
        assert mask[2, 0].item() is True   # row 3, col 1 — past, should be allowed
        assert mask[2, 1].item() is True   # row 3, col 2 — past
        assert mask[2, 2].item() is True   # row 3, col 3 — current
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
        """With V=ones, output should also be all ones (since attention weights sum to 1)."""
        q = torch.randn(1, 1, 4)
        k = torch.randn(1, 4, 4)
        v_ones = torch.ones(1, 4, 4)
        out = scaled_dot_product_attention(q, k, v_ones)
        assert torch.allclose(out, torch.ones_like(out), atol=1e-5), \
            "attention with V=ones should output ones"

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

        # The first position should be unaffected (it can't attend to last position anyway)
        assert torch.allclose(out_masked[:, 0, :], out_zeroed[:, 0, :], atol=1e-6), \
            "position 0 should not depend on position 3 under causal mask"
        # The second-to-last position should also be unaffected
        assert torch.allclose(out_masked[:, 2, :], out_zeroed[:, 2, :], atol=1e-6), \
            "position 2 should not depend on position 3 under causal mask"

    def test_different_d_k_and_d_v(self):
        """d_v may differ from d_k (e.g., when using multi-head attention)."""
        batch, seq_len, d_k, d_v = 2, 4, 8, 16
        q = torch.randn(batch, seq_len, d_k)
        k = torch.randn(batch, seq_len, d_k)
        v = torch.randn(batch, seq_len, d_v)
        out = scaled_dot_product_attention(q, k, v)
        assert out.shape[-1] == d_v, f"output last dim {out.shape[-1]} != expected {d_v}"
        assert out.shape[-2] == seq_len

    def test_batch_independence(self):
        """Outputs for different batch elements should only depend on their own inputs."""
        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 4, 8)
        # Make the two batch elements clearly different
        v = torch.zeros(2, 4, 8)
        v[0, :, :] = 1.0
        v[1, :, :] = -1.0

        out = scaled_dot_product_attention(q, k, v)

        # Each batch element's output should be different (opposite values)
        assert not torch.allclose(out[0], out[1]), \
            "batch elements with different V should produce different outputs"
