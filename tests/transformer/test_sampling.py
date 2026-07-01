"""
tests/transformer/test_sampling.py — Tests for token sampling strategies.
"""

import torch
from core.transformer.sampling import (
    sample,
    sample_argmax,
    sample_top_k,
    sample_top_p,
    sample_with_temperature,
)


class TestSampleArgmax:
    """Tests for greedy argmax sampling."""

    def test_returns_highest_logit(self):
        """Argmax should return the index of the highest logit."""
        logits = torch.tensor([-5.0, -3.0, 10.0, -1.0, 2.0])
        token = sample_argmax(logits)
        assert token.item() == 2, "argmax should return index 2"

    def test_batched(self):
        """Argmax should work with batched inputs."""
        logits = torch.tensor(
            [
                [-5.0, 10.0, -1.0],
                [10.0, -5.0, -1.0],
            ]
        )
        tokens = sample_argmax(logits)
        assert tokens.shape == (2,), "batched argmax should return (batch,)"

    def test_deterministic(self):
        """Argmax should always return the same result."""
        logits = torch.randn(10)
        t1 = sample_argmax(logits)
        t2 = sample_argmax(logits)
        assert t1.item() == t2.item(), "argmax should be deterministic"

    def test_shape_preservation(self):
        """1-D input returns scalar; batched returns 1-D."""
        assert sample_argmax(torch.randn(10)).dim() == 0, "1D should give 0D"
        assert sample_argmax(torch.randn(2, 10)).shape == (2,), (
            "2D should give (batch,)"
        )


class TestSampleWithTemperature:
    """Tests for temperature sampling."""

    def test_low_temp_approximates_argmax(self):
        """Very low temperature should usually pick the highest logit."""
        logits = torch.tensor([-100.0, -100.0, 50.0, -100.0, -100.0])
        # temperature=0.01 makes the distribution very sharp
        tokens = [
            sample_with_temperature(logits, temperature=0.01).item() for _ in range(20)
        ]
        assert all(t == 2 for t in tokens), "low temp should almost always pick argmax"

    def test_high_temp_increases_randomness(self):
        """High temperature should make distribution more uniform."""
        logits = torch.tensor([0.0, 0.0, 0.0])
        # At high temperature, distribution is nearly uniform
        token = sample_with_temperature(logits, temperature=100.0).item()
        assert 0 <= token <= 2, "token should be in valid range"

    def test_temperature_one_is_softmax(self):
        """Temperature=1.0 is standard softmax sampling."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        token = sample_with_temperature(logits, temperature=1.0)
        assert token.item() in [0, 1, 2], "token should be in vocabulary range"

    def test_output_shape(self):
        """Output shapes should match argmax convention."""
        assert sample_with_temperature(torch.randn(10), 1.0).dim() == 0, "1D"
        assert sample_with_temperature(torch.randn(2, 10), 1.0).shape == (2,), "batched"


class TestSampleTopK:
    """Tests for top-k sampling."""

    def test_only_top_k_survive(self):
        """Only the top-k tokens should ever be sampled."""
        logits = torch.tensor([-100.0, -100.0, 0.0, -100.0, 100.0])
        # Top-2: indices 2 and 4
        tokens = [sample_top_k(logits, k=2).item() for _ in range(100)]
        assert all(t in [2, 4] for t in tokens), (
            f"only indices 2,4 should appear, got {set(tokens)}"
        )

    def test_k_equals_vocab_size_is_standard(self):
        """k=vocab_size is equivalent to no filtering."""
        logits = torch.randn(10)
        t1 = sample_top_k(logits, k=10)
        t2 = sample_with_temperature(logits, temperature=1.0)
        assert t1.item() in range(10)
        assert t2.item() in range(10)

    def test_k_one_is_argmax(self):
        """k=1 should be equivalent to argmax."""
        logits = torch.tensor([-5.0, 10.0, 3.0, -2.0])
        token = sample_top_k(logits, k=1)
        assert token.item() == 1, "k=1 should give argmax"


class TestSampleTopP:
    """Tests for nucleus (top-p) sampling."""

    def test_p_one_is_no_filtering(self):
        """p=1.0 includes all tokens, equivalent to standard sampling."""
        logits = torch.randn(10)
        token = sample_top_p(logits, p=1.0)
        assert token.item() in range(10)

    def test_p_zero_is_argmax(self):
        """p=0.0 should be equivalent to argmax."""
        logits = torch.tensor([-10.0, 5.0, -10.0])
        token = sample_top_p(logits, p=0.0)
        assert token.item() == 1, "p=0 should give argmax"

    def test_small_p_limits_tokens(self):
        """Very small p should restrict to few tokens."""
        logits = torch.tensor([-100.0, -100.0, 50.0, -100.0, 0.0])
        # With p=0.5, only index 2 (very high prob after softmax) should be picked
        tokens = [sample_top_p(logits, p=0.5).item() for _ in range(100)]
        assert all(t == 2 for t in tokens), (
            f"p=0.5 should only pick token 2 when it dominates, got {set(tokens)}"
        )


class TestSample:
    """Tests for the combined sample() function."""

    def test_sample_defaults_to_temperature_1(self):
        """Default params should work (temperature=1, no top-k/p)."""
        logits = torch.randn(10)
        token = sample(logits)
        assert token.item() in range(10)

    def test_temperature_zero_is_argmax(self):
        """temperature=0 should behave like argmax."""
        logits = torch.tensor([-50.0, 100.0, -50.0])
        token = sample(logits, temperature=0.0)
        assert token.item() == 1, "temp=0 should give argmax"

    def test_top_k_and_top_p(self):
        """Combined top-k and top-p should work together."""
        logits = torch.randn(100)
        token = sample(logits, temperature=1.0, top_k=50, top_p=0.9)
        assert token.item() in range(100)

    def test_top_k_filtering(self):
        """Top-k should limit tokens."""
        logits = torch.tensor([-100.0, -100.0, 0.0, -100.0, 100.0])
        tokens = [sample(logits, temperature=1.0, top_k=2).item() for _ in range(100)]
        assert all(t in [2, 4] for t in tokens), (
            f"top-k=2 should only allow indices 2 and 4, got {set(tokens)}"
        )

    def test_batched(self):
        """Combined sample should work with batched logits."""
        logits = torch.randn(4, 100)
        tokens = sample(logits, temperature=0.8, top_k=50)
        assert tokens.shape == (4,), "batched sample should return (batch,)"

    def test_reproducible_with_seed(self):
        """Setting torch seed should make sample reproducible."""
        torch.manual_seed(42)
        t1 = sample(torch.randn(10), temperature=0.8, top_k=5)

        torch.manual_seed(42)
        t2 = sample(torch.randn(10), temperature=0.8, top_k=5)

        assert t1.item() == t2.item(), "same seed should give same result"
