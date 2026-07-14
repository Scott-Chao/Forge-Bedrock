"""
tests/transformer/test_block.py — Tests for GPTBlock.
"""

import pytest
import torch
from core.transformer.transformer import GPTBlock, _create_causal_mask


class TestGPTBlock:
    """Tests for the GPT decoder block."""

    @pytest.fixture
    def small_block(self):
        """A tiny GPTBlock for quick tests."""
        return GPTBlock(
            d_model=32,
            n_heads=4,
            max_seq_len=16,
            d_ff=64,  # explicit small d_ff
        )

    @pytest.fixture
    def sample_input(self):
        batch, seq_len, d_model = 2, 8, 32
        return torch.randn(batch, seq_len, d_model)

    def test_construction(self, small_block):
        """Block should have all expected sub-modules."""
        block = small_block
        assert hasattr(block, "norm_1"), "missing norm_1"
        assert hasattr(block, "attn"), "missing attn"
        assert hasattr(block, "norm_2"), "missing norm_2"
        assert hasattr(block, "ff"), "missing ff"

    def test_norm_first_rmsnorm(self, small_block):
        """Pre-norm architecture: norm_1 and norm_2 should be RMSNorm."""
        from core.transformer.normalization import RMSNorm

        assert isinstance(small_block.norm_1, RMSNorm)
        assert isinstance(small_block.norm_2, RMSNorm)

    def test_attn_is_multihead(self, small_block):
        """Attention module should be MultiHeadAttention."""
        from core.transformer.transformer import MultiHeadAttention

        assert isinstance(small_block.attn, MultiHeadAttention)

    def test_ff_is_feedforward(self, small_block):
        """FeedForward module should be FeedForward."""
        from core.transformer.transformer import FeedForward

        assert isinstance(small_block.ff, FeedForward)

    def test_output_shape(self, small_block, sample_input):
        """Output should match (batch, seq_len, d_model)."""
        out, _ = small_block(sample_input)
        assert out.shape == sample_input.shape, (
            f"output shape {out.shape} != input shape {sample_input.shape}"
        )

    def test_output_not_identity(self, small_block, sample_input):
        """Block should transform the input."""
        out, _ = small_block(sample_input)
        assert not torch.allclose(out, sample_input, atol=1e-4), (
            "block should not pass through input unchanged"
        )

    def test_causal_mask_generated(self, small_block):
        """Block should work without explicit mask (auto-generates causal)."""
        x = torch.randn(1, 4, 32)
        out, _ = small_block(x)
        assert out.shape == x.shape

    def test_causal_mask_effect(self, small_block):
        """With causal mask, the block should be autoregressive.

        If we zero out the last token's value, earlier positions should
        be unaffected (since they can't attend to future tokens).
        """
        batch, seq_len, d_model = 1, 6, 32
        x = torch.randn(batch, seq_len, d_model)
        n_heads = small_block.n_heads

        # This test assumes the attention uses a causal mask.
        # Even with random init, the first position should not depend
        # on the last position.

        mask = _create_causal_mask(seq_len)
        out_full, _ = small_block(x, mask=mask)

        # Zero out the last token and recompute
        x_zeroed = x.clone()
        x_zeroed[:, -1, :] = 0.0
        out_zeroed, _ = small_block(x_zeroed, mask=mask)

        # Position 0 should be very similar (not affected by position 5)
        diff = (out_full[:, 0, :] - out_zeroed[:, 0, :]).abs().mean().item()
        assert diff < 1e-4, (
            f"position 0 should not depend on position 5 under causal mask, diff={diff:.6f}"
        )

    def test_two_blocks_stack(self):
        """Two GPTBlocks stacked should produce valid output (deeper)."""
        d_model, n_heads = 32, 4
        block1 = GPTBlock(d_model, n_heads, max_seq_len=16)
        block2 = GPTBlock(d_model, n_heads, max_seq_len=16)

        x = torch.randn(2, 6, d_model)
        x, _ = block1(x)
        assert x.shape == (2, 6, d_model), f"after block1: {x.shape}"
        x, _ = block2(x)
        assert x.shape == (2, 6, d_model), f"after block2: {x.shape}"
        # Two blocks should change the representation
        assert torch.all(torch.isfinite(x)), "output should be finite"

    def test_max_seq_len_respected(self):
        """Block should handle sequences shorter than max_seq_len."""
        block = GPTBlock(d_model=32, n_heads=4, max_seq_len=128)
        x = torch.randn(1, 3, 32)  # much shorter
        out, _ = block(x)
        assert out.shape == (1, 3, 32)

    def test_forward_with_explicit_mask(self, small_block, sample_input):
        """Should accept an explicit mask argument."""
        seq_len = sample_input.size(1)
        mask = _create_causal_mask(seq_len)
        out, _ = small_block(sample_input, mask=mask)
        assert out.shape == sample_input.shape

    def test_gradient_flows(self, small_block, sample_input):
        """Gradients should flow through all parameters."""
        out, _ = small_block(sample_input)
        loss = out.sum()
        loss.backward()

        # All parameters should have gradients
        for name, param in small_block.named_parameters():
            assert param.grad is not None, f"parameter {name} has no gradient"
            assert param.grad.abs().sum().item() > 0, (
                f"parameter {name} has zero gradient"
            )

    def test_causal_mask_factory(self, small_block):
        """_create_causal_mask should return a valid causal mask."""
        seq_len = 8
        mask = _create_causal_mask(seq_len)
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool
        # Lower triangle (including diagonal) should be True
        assert mask[0, 0].item() is True
        assert mask[3, 1].item() is True
        # Upper triangle (excluding diagonal) should be False
        assert mask[1, 3].item() is False
