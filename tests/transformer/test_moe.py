"""
tests/transformer/test_moe.py — Tests for Mixture of Experts components.
"""

import pytest
import torch
from core.transformer.moe import MoERouter


class TestMoERouter:
    """Tests for the top-k softmax router."""

    @pytest.mark.parametrize(
        "batch,seq_len,k,n_experts",
        [
            (2, 5, 2, 8),  # typical
            (1, 1, 1, 4),  # k=1, minimal shape
            (4, 3, 4, 4),  # k=n_experts
            (2, 128, 2, 16),  # longer seq, more experts
        ],
    )
    def test_routing_properties(self, batch, seq_len, k, n_experts):
        """Core contract: correct shapes, weights sum to 1, indices in range."""
        router = MoERouter(d_model=8, n_experts=n_experts, k=k)
        x = torch.randn(batch, seq_len, 8)
        weights, indices = router(x)

        assert weights.shape == (batch, seq_len, k)
        assert weights.dtype == torch.float32
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, seq_len))

        assert indices.shape == (batch, seq_len, k)
        assert indices.dtype == torch.long
        assert indices.min() >= 0
        assert indices.max() < n_experts

        assert router.gate.bias is None

    def test_deterministic(self):
        """Same input always produces the same routing."""
        router = MoERouter(d_model=8, n_experts=4, k=2)
        x = torch.randn(2, 5, 8)
        w1, i1 = router(x)
        w2, i2 = router(x)

        assert torch.equal(i1, i2)
        assert torch.equal(w1, w2)
