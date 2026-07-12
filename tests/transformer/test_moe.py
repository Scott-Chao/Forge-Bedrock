"""
tests/transformer/test_moe.py — Tests for Mixture of Experts components.
"""

import pytest
import torch
from core.transformer.moe import MoEFFN, MoERouter


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


class TestMoEFFN:
    """Tests for the sparse MoE feedforward network."""

    def test_construction(self):
        """Should create correct number of experts with a router."""
        d_model, d_ff, n_experts, k = 32, 128, 8, 2
        moe = MoEFFN(d_model, d_ff, n_experts, k)

        assert len(moe.experts) == n_experts
        assert isinstance(moe.router, MoERouter)
        assert moe.router.n_experts == n_experts
        assert moe.router.k == k

    def test_output_shape(self):
        """Output should match input shape."""
        batch, seq_len, d_model = 2, 8, 32
        moe = MoEFFN(d_model, d_ff=128, n_experts=4, k=2)
        x = torch.randn(batch, seq_len, d_model)
        out, aux_loss = moe(x)
        assert out.shape == (batch, seq_len, d_model)
        assert aux_loss.shape == torch.Size([])
        assert aux_loss.item() > 0

    def test_sparse_activation(self):
        """Each token should activate exactly k experts.

        The output from MoEFFN should be a weighted sum of exactly k
        expert outputs, which means the router should assign each
        token to exactly k distinct experts.
        """
        batch, seq_len, d_model = 2, 8, 16
        n_experts, k = 8, 2
        moe = MoEFFN(d_model, d_ff=64, n_experts=n_experts, k=k)
        x = torch.randn(batch, seq_len, d_model)

        weights, indices = moe.router(x)

        # Each token has exactly k non-zero weights summing to 1
        assert weights.shape == (batch, seq_len, k)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, seq_len))

        # Each token's k expert indices are distinct
        for b in range(batch):
            for t in range(seq_len):
                assert len(set(indices[b, t].tolist())) == k, (
                    f"token ({b},{t}) should have {k} distinct expert indices, "
                    f"got {indices[b, t].tolist()}"
                )

    def test_gradient_flows(self):
        """Gradients should flow through MoEFFN, including the router."""
        d_model, d_ff = 16, 64
        moe = MoEFFN(d_model, d_ff, n_experts=4, k=2)
        # Use enough tokens so every expert gets at least one
        x = torch.randn(2, 16, d_model, requires_grad=True)
        out, aux_loss = moe(x)
        loss = out.sum() + aux_loss
        loss.backward()

        assert x.grad is not None, "gradients should flow to input"
        assert x.grad.abs().sum().item() > 0

        # Router gate should have gradients (routing decisions are differentiable)
        assert moe.router.gate.weight.grad is not None, (
            "router gate should receive gradients"
        )
        assert moe.router.gate.weight.grad.abs().sum().item() > 0

        # All experts should have gradients
        for i, expert in enumerate(moe.experts):
            for name, param in expert.named_parameters():
                assert param.grad is not None, (
                    f"gradient missing for expert {i}, param {name}"
                )

    def test_total_params_greater_than_active(self):
        """Total params should exceed per-token active params (k << n_experts).

        With 8 experts and k=2, total params is ~4x more than what
        gets activated for any single token.
        """
        d_model, d_ff = 32, 128
        n_experts, k = 8, 2
        moe = MoEFFN(d_model, d_ff, n_experts=n_experts, k=k)

        total = sum(p.numel() for p in moe.parameters())

        # Each expert: w1 (d_model*d_ff) + bias1 (d_ff) + w2 (d_ff*d_model) + bias2 (d_model)
        expert_params = d_model * d_ff + d_ff + d_ff * d_model + d_model
        active = k * expert_params  # k=2 experts active per token

        assert total > active, (
            f"Total params ({total}) should exceed active ({active}) for sparsity"
        )
        # n_experts experts + router gate (d_model * n_experts, no bias)
        expected_total = n_experts * expert_params + d_model * n_experts
        assert total == expected_total, (
            f"Total params mismatch: {total} vs {expected_total}"
        )

    def test_aux_loss_uniform_routing(self):
        """Uniform routing should give aux_loss ≈ 1.0.

        When every expert gets exactly the same fraction of tokens and the
        router assigns equal probability to all, f_i * P_i = 1/n² each,
        so n_experts * Σ(f_i * P_i) = n * n * (1/n²) = 1.0.
        """
        d_model, d_ff, n_experts, k = 16, 64, 4, 1
        moe = MoEFFN(d_model, d_ff, n_experts=n_experts, k=k)

        # Set gate weights to zero so all experts get equal routing prob
        with torch.no_grad():
            moe.router.gate.weight.zero_()

        x = torch.randn(4, 8, d_model)
        _, aux_loss = moe(x)

        # With uniform probs and k=1, each expert gets 1/n of tokens
        # aux_loss should be close to n * Σ((1/n) * (1/n)) = 1.0
        assert torch.isclose(aux_loss, torch.tensor(1.0), atol=1e-5), (
            f"uniform routing aux_loss = {aux_loss.item():.6f}, expected ~1.0"
        )

    def test_aux_loss_higher_with_imbalanced_routing(self):
        """Imbalanced routing should produce higher aux_loss than uniform.

        With deterministic input (all ones) and k=1, biasing the gate toward
        expert 0 makes almost every token route to expert 0, increasing loss.
        """
        d_model, d_ff, n_experts, k = 16, 64, 4, 1
        moe = MoEFFN(d_model, d_ff, n_experts=n_experts, k=k)

        x = torch.ones(4, 8, d_model)  # deterministic input

        # Uniform routing: all gate weights = 0
        with torch.no_grad():
            moe.router.gate.weight.zero_()
        _, aux_base = moe(x)

        # Imbalanced: only expert 0 has non-zero weights
        with torch.no_grad():
            moe.router.gate.weight.zero_()
            moe.router.gate.weight[0, :] = 1.0
        _, aux_biased = moe(x)

        assert aux_biased > aux_base, (
            f"biased loss ({aux_biased.item():.6f}) should be > "
            f"uniform loss ({aux_base.item():.6f})"
        )

    def test_per_position_independence(self):
        """Each position should be processed independently by MoEFFN.

        Since experts are position-wise, modifying one token's input
        should only affect that token's output.
        """
        batch, seq_len, d_model = 1, 4, 16
        moe = MoEFFN(d_model, d_ff=64, n_experts=4, k=2)
        x = torch.randn(batch, seq_len, d_model)
        out_original, _ = moe(x)

        # Change only position 2
        x_mod = x.clone()
        x_mod[:, 2, :] = torch.randn(d_model)
        out_mod, _ = moe(x_mod)

        for pos in [0, 1, 3]:
            assert torch.allclose(
                out_original[:, pos, :], out_mod[:, pos, :], atol=1e-6
            ), f"position {pos} changed after modifying position 2"

        assert not torch.allclose(out_original[:, 2, :], out_mod[:, 2, :], atol=1e-6)
