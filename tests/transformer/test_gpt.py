"""
tests/transformer/test_gpt.py — Tests for the full GPT model.
"""

import pytest
import torch
from core.transformer.gpt import GPT


class TestGPT:
    """Tests for the full GPT language model."""

    @pytest.fixture
    def tiny_gpt(self):
        """A tiny GPT for fast tests (same param count as a toy)."""
        return GPT(
            vocab_size=70,
            d_model=32,
            n_layers=2,
            n_heads=4,
            max_seq_len=64,
        )

    @pytest.fixture
    def sample_tokens(self):
        return torch.randint(0, 70, (2, 8))  # (batch=2, seq=8)

    def test_construction(self, tiny_gpt):
        """GPT should have all expected sub-modules."""
        gpt = tiny_gpt
        assert hasattr(gpt, "token_embedding"), "missing token_embedding"
        assert hasattr(gpt, "blocks"), "missing blocks (ModuleList)"
        assert hasattr(gpt, "final_norm"), "missing final_norm"
        assert hasattr(gpt, "lm_head"), "missing lm_head"
        assert len(gpt.blocks) == gpt.n_layers, (
            f"expected {gpt.n_layers} blocks, got {len(gpt.blocks)}"
        )

    def test_output_shape(self, tiny_gpt, sample_tokens):
        """Logits should be (batch, seq_len, vocab_size)."""
        logits = tiny_gpt(sample_tokens)
        batch, seq_len = sample_tokens.shape
        assert logits.shape == (batch, seq_len, tiny_gpt.vocab_size), (
            f"logits shape {logits.shape} != {(batch, seq_len, tiny_gpt.vocab_size)}"
        )

    def test_logits_dtype(self, tiny_gpt, sample_tokens):
        """Logits should be float32."""
        logits = tiny_gpt(sample_tokens)
        assert logits.dtype == torch.float32

    def test_logits_not_zeros(self, tiny_gpt, sample_tokens):
        """Logits should not be all zeros (random init should break symmetry)."""
        logits = tiny_gpt(sample_tokens)
        assert logits.abs().sum().item() > 0, "logits should not be all zeros"

    def test_loss_decreases_with_training(self, tiny_gpt):
        """A single gradient step should decrease the loss."""
        tokens = torch.randint(0, 70, (4, 16))
        targets = tokens[:, 1:]  # predict next token
        inputs = tokens[:, :-1]

        # Initial loss
        logits = tiny_gpt(inputs)
        B, S, V = logits.shape
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, V), targets.reshape(-1)
        )
        loss_init = loss.item()

        # One gradient step
        loss.backward()
        with torch.no_grad():
            for p in tiny_gpt.parameters():
                p.data -= 0.01 * p.grad
                p.grad = None

        # Loss after step
        logits2 = tiny_gpt(inputs)
        loss2 = torch.nn.functional.cross_entropy(
            logits2.view(-1, V), targets.reshape(-1)
        )
        loss_final = loss2.item()

        assert loss_final < loss_init, (
            f"loss should decrease after 1 step: {loss_init:.4f} -> {loss_final:.4f}"
        )

    def test_loss_on_random_init(self, tiny_gpt, sample_tokens):
        """Cross-entropy loss at init should be roughly -ln(1/vocab_size).

        At random initialization, the model should be guessing uniformly,
        so cross-entropy should be close to ln(vocab_size).
        """
        logits = tiny_gpt(sample_tokens)
        # Shift: predict next token
        targets = sample_tokens[:, 1:]
        inputs = sample_tokens[:, :-1]

        logits = tiny_gpt(inputs)
        B, S, V = logits.shape
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, V), targets.reshape(-1)
        )
        expected_loss = torch.tensor(V).float().log()
        # At init, loss should be within reasonable range of uniform guess
        assert loss.item() < expected_loss.item() + 1.0, (
            f"init loss {loss.item():.4f} much higher than uniform {expected_loss.item():.4f}"
        )

    def test_blocks_are_gptblocks(self, tiny_gpt):
        """Each element in blocks should be a GPTBlock."""
        from core.transformer.block import GPTBlock

        for i, block in enumerate(tiny_gpt.blocks):
            assert isinstance(block, GPTBlock), (
                f"block {i} is not GPTBlock: {type(block)}"
            )

    def test_different_configs(self):
        """GPT should work with different model sizes."""
        configs = [
            dict(vocab_size=70, d_model=16, n_layers=2, n_heads=2, max_seq_len=32),
            dict(vocab_size=70, d_model=32, n_layers=4, n_heads=4, max_seq_len=64),
            dict(vocab_size=70, d_model=64, n_layers=3, n_heads=8, max_seq_len=128),
        ]
        for cfg in configs:
            gpt = GPT(**cfg)
            tokens = torch.randint(0, 70, (1, 8))
            logits = gpt(tokens)
            assert logits.shape == (1, 8, cfg["vocab_size"]), (
                f"shape mismatch for config d_model={cfg['d_model']}"
            )

    def test_num_parameters(self, tiny_gpt):
        """num_parameters should return a positive integer."""
        n = tiny_gpt.num_parameters
        assert isinstance(n, (int,))
        assert n > 0, f"num_parameters should be > 0, got {n}"

    def test_repr(self, tiny_gpt):
        """repr should include key config values."""
        r = repr(tiny_gpt)
        assert "GPT" in r
        assert str(tiny_gpt.vocab_size) in r
        assert str(tiny_gpt.d_model) in r
        assert str(tiny_gpt.n_layers) in r
        # num_parameters is formatted with comma: 29,792
        assert f"{tiny_gpt.num_parameters:,}" in r

    def test_causal_mask_passed_to_blocks(self, tiny_gpt):
        """GPT should pass mask through to blocks (autoregressive property).

        We test this by comparing outputs with and without causal mask.
        Without mask, the first token's output should depend on later tokens.
        With mask, it should NOT depend on later tokens.
        """
        tokens = torch.randint(0, 70, (1, 6))
        mask = None  # GPT should auto-create causal mask

        # Forward without explicit mask (GPT creates its own)
        logits_auto = tiny_gpt(tokens)

        # Now manually create a mask and compare
        from core.transformer.attention import _create_causal_mask

        seq_len = tokens.size(1)
        mask = _create_causal_mask(seq_len)

        logits_masked = tiny_gpt(tokens, mask=mask)

        assert torch.allclose(logits_auto, logits_masked, atol=1e-6), (
            "output with auto-mask should match explicit causal mask"
        )

    def test_forward_multiple_times(self, tiny_gpt, sample_tokens):
        """Multiple forward passes should give the same result (no state)."""
        out1 = tiny_gpt(sample_tokens)
        out2 = tiny_gpt(sample_tokens)
        assert torch.allclose(out1, out2, atol=1e-6), (
            "deterministic: same input should give same output"
        )

    def test_gradient_flows_to_embedding(self, tiny_gpt, sample_tokens):
        """Gradients should flow back to the token embedding."""
        logits = tiny_gpt(sample_tokens)
        loss = logits.sum()
        loss.backward()

        # Embedding should have gradients
        assert tiny_gpt.token_embedding.embedding.weight.grad is not None, (
            "token embedding should receive gradients"
        )
        assert tiny_gpt.token_embedding.embedding.weight.grad.abs().sum().item() > 0, (
            "token embedding gradients should be non-zero"
        )
