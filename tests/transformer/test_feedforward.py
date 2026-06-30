"""
tests/transformer/test_feedforward.py — Tests for FeedForward.
"""

import torch
from core.transformer.feedforward import FeedForward


class TestFeedForward:
    """Tests for the position-wise FeedForward network."""

    def test_construction_default(self):
        """Default d_ff should be 4 * d_model."""
        d_model = 64
        ff = FeedForward(d_model)
        assert ff.d_ff == 4 * d_model, (
            f"default d_ff should be 4*d_model, got {ff.d_ff}"
        )
        assert ff.w_1.in_features == d_model
        assert ff.w_1.out_features == 4 * d_model
        assert ff.w_2.in_features == 4 * d_model
        assert ff.w_2.out_features == d_model

    def test_construction_custom_d_ff(self):
        """Should accept an explicit d_ff."""
        d_model, d_ff = 32, 128
        ff = FeedForward(d_model, d_ff=d_ff)
        assert ff.d_ff == d_ff
        assert ff.w_1.out_features == d_ff
        assert ff.w_2.in_features == d_ff

    def test_output_shape(self):
        """Output should match (batch, seq, d_model)."""
        batch, seq_len, d_model = 2, 8, 64
        ff = FeedForward(d_model)
        x = torch.randn(batch, seq_len, d_model)
        out = ff(x)
        assert out.shape == (batch, seq_len, d_model), (
            f"output shape {out.shape} != expected {(batch, seq_len, d_model)}"
        )

    def test_output_not_identity(self):
        """FFN should transform the input, not pass it through unchanged."""
        batch, seq_len, d_model = 1, 4, 16
        ff = FeedForward(d_model)
        x = torch.randn(batch, seq_len, d_model)
        out = ff(x)
        assert not torch.allclose(out, x, atol=1e-4), (
            "FFN should transform input, not pass through"
        )

    def test_relu_introduces_nonlinearity(self):
        """ReLU breaks sign antisymmetry: f(-x) != -f(x).

        For a purely linear network (no activation), f(-x) = -f(x) exactly.
        ReLU breaks this by zeroing negative hidden activations, so the
        output pattern changes beyond just a sign flip.
        """
        d_model = 16
        ff = FeedForward(d_model, bias=False)
        x = torch.randn(1, 1, d_model)
        out_pos = ff(x)
        out_neg = ff(-x)
        # For a linear network: f(-x) = -f(x)
        # With ReLU this should NOT hold
        assert not torch.allclose(out_neg, -out_pos, atol=1e-4), (
            "FFN with ReLU should be non-linear: f(-x) != -f(x)"
        )

    def test_positive_input_flows_through(self):
        """Positive input should produce non-zero output after ReLU."""
        d_model = 16
        ff = FeedForward(d_model, bias=False)
        x = torch.ones(1, 1, d_model) * 2.0
        out = ff(x)
        # Output should be non-zero (positive activations survive ReLU)
        assert out.abs().sum().item() > 0, (
            "positive input should produce non-zero output"
        )

    def test_per_position_independence(self):
        """Each position should be processed independently by the FFN.

        If we modify one position's input, only that position's output
        should change.
        """
        batch, seq_len, d_model = 1, 4, 16
        ff = FeedForward(d_model)
        x = torch.randn(batch, seq_len, d_model)

        out_original = ff(x)

        # Change only position 2
        x_mod = x.clone()
        x_mod[:, 2, :] = torch.randn(d_model)
        out_mod = ff(x_mod)

        # Positions 0, 1, 3 should be unchanged
        for pos in [0, 1, 3]:
            assert torch.allclose(
                out_original[:, pos, :], out_mod[:, pos, :], atol=1e-6
            ), f"position {pos} changed after modifying position 2"

        # Position 2 should be different
        assert not torch.allclose(out_original[:, 2, :], out_mod[:, 2, :], atol=1e-6), (
            "position 2 should change after its input changed"
        )

    def test_bias_enabled_by_default(self):
        """Biases should be present by default."""
        d_model = 32
        ff = FeedForward(d_model)
        assert ff.w_1.bias is not None and isinstance(ff.w_1.bias, torch.Tensor), (
            "w_1 should have bias by default"
        )
        assert ff.w_2.bias is not None and isinstance(ff.w_2.bias, torch.Tensor), (
            "w_2 should have bias by default"
        )

    def test_bias_disabled(self):
        """Setting bias=False should remove bias terms."""
        d_model = 32
        ff = FeedForward(d_model, bias=False)
        assert ff.w_1.bias is None or not ff.w_1.bias.requires_grad, (
            "w_1 should have no bias when bias=False" if hasattr(ff.w_1, "bias") else ""
        )
        # nn.Linear with bias=False still has the attribute but it's None
        assert ff.w_1.bias is None, "w_1 bias should be None when bias=False"
        assert ff.w_2.bias is None, "w_2 bias should be None when bias=False"

    def test_relu_module_type(self):
        """The activation should be a ReLU instance."""
        d_model = 16
        ff = FeedForward(d_model)
        assert isinstance(ff.relu, torch.nn.ReLU), "activation should be torch.nn.ReLU"

    def test_gradient_flows(self):
        """Gradients should flow through the FFN."""
        d_model = 16
        ff = FeedForward(d_model)
        x = torch.randn(1, 1, d_model, requires_grad=True)
        out = ff(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "gradients should flow to input"
        assert x.grad.abs().sum().item() > 0, "gradients should be non-zero"
        # Parameters should also have gradients
        for name, param in ff.named_parameters():
            assert param.grad is not None, f"gradient missing for {name}"
            assert param.grad.abs().sum().item() > 0, f"zero gradient for {name}"
