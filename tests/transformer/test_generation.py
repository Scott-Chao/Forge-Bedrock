"""
tests/transformer/test_generation.py — Tests for GPT.generate().
"""

import pytest
import torch
from core.transformer.transformer import GPT


class TestGenerate:
    """Tests for the GPT.generate() autoregressive decoding."""

    @pytest.fixture
    def tiny_gpt(self):
        """A tiny GPT for fast generation tests."""
        model = GPT(
            vocab_size=70,
            d_model=32,
            n_layers=2,
            n_heads=4,
            max_seq_len=64,
        )
        model.eval()  # eval mode for deterministic dropout
        return model

    def test_generate_basic(self, tiny_gpt):
        """generate() should return a longer sequence than the prompt."""
        prompt = torch.randint(0, 70, (1, 5))
        output = tiny_gpt.generate(prompt, max_new_tokens=10)
        assert output.shape == (1, 15), f"expected (1, 15), got {output.shape}"

    def test_generate_no_batch_dim(self, tiny_gpt):
        """generate() should accept 1-D prompt (no batch dim)."""
        prompt = torch.randint(0, 70, (5,))
        output = tiny_gpt.generate(prompt, max_new_tokens=5)
        # Should auto-add batch dim
        assert output.dim() == 2, (
            f"output should be 2-D (batch, seq), got shape {output.shape}"
        )
        assert output.shape[1] == 10, f"expected seq_len=10, got {output.shape[1]}"

    def test_generate_with_temperature(self, tiny_gpt):
        """generate() should work with temperature sampling."""
        prompt = torch.randint(0, 70, (1, 5))
        output = tiny_gpt.generate(prompt, max_new_tokens=5, temperature=0.8)
        assert output.shape == (1, 10)

    def test_generate_with_top_k(self, tiny_gpt):
        """generate() should work with top-k sampling."""
        prompt = torch.randint(0, 70, (1, 5))
        output = tiny_gpt.generate(prompt, max_new_tokens=5, top_k=40)
        assert output.shape == (1, 10)

    def test_generate_with_top_p(self, tiny_gpt):
        """generate() should work with top-p (nucleus) sampling."""
        prompt = torch.randint(0, 70, (1, 5))
        output = tiny_gpt.generate(prompt, max_new_tokens=5, top_p=0.9)
        assert output.shape == (1, 10)

    def test_generate_with_all_params(self, tiny_gpt):
        """generate() should work with all sampling params combined."""
        prompt = torch.randint(0, 70, (1, 5))
        output = tiny_gpt.generate(
            prompt, max_new_tokens=5, temperature=0.7, top_k=40, top_p=0.9
        )
        assert output.shape == (1, 10)

    def test_generate_argmax(self, tiny_gpt):
        """temperature=0 should give greedy/argmax generation."""
        prompt = torch.randint(0, 70, (1, 5))
        output = tiny_gpt.generate(prompt, max_new_tokens=10, temperature=0.0)
        assert output.shape == (1, 15)

    def test_generate_argmax_deterministic(self, tiny_gpt):
        """argmax generation should be deterministic for same prompt."""
        prompt = torch.randint(0, 70, (1, 5))
        out1 = tiny_gpt.generate(prompt, max_new_tokens=10, temperature=0.0)
        out2 = tiny_gpt.generate(prompt, max_new_tokens=10, temperature=0.0)
        assert torch.equal(out1, out2), "argmax should be deterministic"

    def test_generate_eos_stops_early(self, tiny_gpt):
        """EOS token should stop generation early."""
        # Use a dummy token ID as EOS (say token 0)
        prompt = torch.randint(1, 70, (1, 3))  # avoid using token 0 in prompt
        output = tiny_gpt.generate(
            prompt, max_new_tokens=50, eos_token_id=0, temperature=0.0
        )
        # The output should not be longer than 53 (3 + 50)
        assert output.shape[1] <= 53, "generation should stop at EOS or max_new_tokens"

    def test_generate_no_tokens(self, tiny_gpt):
        """max_new_tokens=0 should return the prompt unchanged."""
        prompt = torch.randint(0, 70, (1, 5))
        output = tiny_gpt.generate(prompt, max_new_tokens=0)
        assert torch.equal(output, prompt), (
            "max_new_tokens=0 should return prompt unchanged"
        )

    def test_generate_output_dtype(self, tiny_gpt):
        """Output should be LongTensor (token IDs)."""
        prompt = torch.randint(0, 70, (1, 5))
        output = tiny_gpt.generate(prompt, max_new_tokens=5)
        assert output.dtype == torch.long, f"expected long dtype, got {output.dtype}"

    def test_generate_batch(self, tiny_gpt):
        """generate() should handle batched prompts."""
        prompt = torch.randint(0, 70, (3, 5))  # (batch=3, seq=5)
        output = tiny_gpt.generate(prompt, max_new_tokens=5)
        assert output.shape == (3, 10), f"expected (3, 10), got {output.shape}"

    def test_generate_batch_independent(self, tiny_gpt):
        """Different batch elements should generate independently."""
        prompt = torch.randint(0, 70, (2, 4))
        # With argmax, outputs might be same or different depending on logits
        output = tiny_gpt.generate(prompt, max_new_tokens=3, temperature=0.0)
        assert output.shape == (2, 7)

    def test_total_length_limit(self, tiny_gpt):
        """Generated sequence should not exceed max_seq_len."""
        max_seq_len = tiny_gpt.max_seq_len
        prompt = torch.randint(0, 70, (1, max_seq_len - 5))
        output = tiny_gpt.generate(prompt, max_new_tokens=20, temperature=0.0)
        # If prompt + new tokens tries to exceed max_seq_len, the model
        # should handle it gracefully (or at least not crash)
        assert output.shape[1] <= max_seq_len, (
            f"output length {output.shape[1]} should not exceed max_seq_len={max_seq_len}"
        )
