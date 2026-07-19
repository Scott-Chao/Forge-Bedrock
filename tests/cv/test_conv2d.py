"""
tests/cv/test_conv2d.py — Tests for Conv2d (im2col + GEMM).

Compares against torch.nn.Conv2d as reference and checks gradient flow.
"""

import pytest
import torch
import torch.nn as nn
from core.cv import Conv2d as MyConv2d

# ── helpers ────────────────────────────────────────────────────────────────


def clone_params(src: nn.Module, dst: nn.Module) -> None:
    """Copy weights (and bias if present) from our Conv2d *to* a reference
    ``torch.nn.Conv2d`` so both start from identical parameters."""
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if src.bias is not None and dst.bias is not None:
            dst.bias.copy_(src.bias)


# ============================================================================
# 1. Initialisation & Properties
# ============================================================================


class TestInit:
    def test_weight_shape(self):
        """Weight shape should be (out_channels, in_channels, k_h, k_w)."""
        layer = MyConv2d(3, 16, 3, padding=1)
        assert layer.weight.shape == (16, 3, 3, 3), (
            f"expected (16, 3, 3, 3), got {layer.weight.shape}"
        )

    @pytest.mark.parametrize(
        "kernel_size, expected",
        [
            (3, (3, 3)),
            ((5, 3), (5, 3)),
        ],
    )
    def test_kernel_size_normalisation(self, kernel_size, expected):
        layer = MyConv2d(3, 8, kernel_size)
        assert layer.kernel_size == expected

    @pytest.mark.parametrize(
        "stride, expected",
        [
            (1, (1, 1)),
            ((2, 1), (2, 1)),
        ],
    )
    def test_stride_normalisation(self, stride, expected):
        layer = MyConv2d(3, 8, 3, stride=stride)
        assert layer.stride == expected

    @pytest.mark.parametrize(
        "padding, expected",
        [
            (0, (0, 0)),
            ((1, 2), (1, 2)),
        ],
    )
    def test_padding_normalisation(self, padding, expected):
        layer = MyConv2d(3, 8, 3, padding=padding)
        assert layer.padding == expected

    @pytest.mark.parametrize(
        "dilation, expected",
        [
            (1, (1, 1)),
            ((2, 3), (2, 3)),
        ],
    )
    def test_dilation_normalisation(self, dilation, expected):
        layer = MyConv2d(3, 8, 3, dilation=dilation)
        assert layer.dilation == expected

    @pytest.mark.parametrize("bias", [True, False])
    def test_bias_parameter(self, bias):
        layer = MyConv2d(3, 16, 3, padding=1, bias=bias)
        if bias:
            assert layer.bias is not None, "bias should exist"
            assert layer.bias.shape == (16,), (
                f"bias shape expected (16,), got {layer.bias.shape}"
            )
        else:
            assert layer.bias is None, "bias should be None"

    def test_weight_is_parameter(self):
        layer = MyConv2d(3, 8, 3)
        assert isinstance(layer.weight, torch.nn.Parameter)

    def test_parameters_count(self):
        layer = MyConv2d(3, 8, 3, bias=True)
        params = list(layer.parameters())
        assert len(params) == 2  # weight + bias

        layer_nb = MyConv2d(3, 8, 3, bias=False)
        params_nb = list(layer_nb.parameters())
        assert len(params_nb) == 1  # weight only

    def test_repr(self):
        layer = MyConv2d(3, 16, kernel_size=3, stride=2, padding=1, dilation=1)
        r = repr(layer)
        assert "in_channels=3" in r
        assert "out_channels=16" in r
        assert "kernel_size=(3, 3)" in r
        assert "stride=(2, 2)" in r
        assert "bias=True" in r

        layer_nb = MyConv2d(3, 8, 3, bias=False)
        assert "bias=False" in repr(layer_nb)


# ============================================================================
# 2. Forward Pass — Numerical Correctness
# ============================================================================


class TestForward:
    """Compare output of MyConv2d with torch.nn.Conv2d on identical params."""

    N, C_in, H, W = 2, 3, 32, 32

    @pytest.fixture
    def x(self):
        torch.manual_seed(0)
        return torch.randn(self.N, self.C_in, self.H, self.W)

    @pytest.mark.parametrize(
        "kernel_size,stride,padding,dilation",
        [
            (3, 1, 1, 1),  # standard 3×3
            (5, 1, 2, 1),  # 5×5
            (3, 2, 1, 1),  # stride=2 (downsample)
            (3, 1, 0, 1),  # no padding
            (3, 1, 2, 2),  # dilation=2
            (3, 1, 1, 1),  # (will be combined with no-bias below)
        ],
    )
    def test_output_matches_ref(self, x, kernel_size, stride, padding, dilation):
        C_out = 16
        my_conv = MyConv2d(
            self.C_in,
            C_out,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        ref_conv = nn.Conv2d(
            self.C_in,
            C_out,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        clone_params(my_conv, ref_conv)

        my_out = my_conv(x)
        ref_out = ref_conv(x)

        assert my_out.shape == ref_out.shape, (
            f"shape mismatch: {my_out.shape} vs {ref_out.shape}"
        )
        assert torch.allclose(my_out, ref_out, atol=1e-6), (
            f"max diff = {(my_out - ref_out).abs().max().item():.2e}"
        )

    def test_no_bias_matches_ref(self, x):
        my_conv = MyConv2d(self.C_in, 16, 3, padding=1, bias=False)
        ref_conv = nn.Conv2d(self.C_in, 16, 3, padding=1, bias=False)
        clone_params(my_conv, ref_conv)

        my_out = my_conv(x)
        ref_out = ref_conv(x)
        assert torch.allclose(my_out, ref_out, atol=1e-6)

    def test_asymmetric_kernel_padding_stride(self, x):
        my_conv = MyConv2d(
            self.C_in, 8, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)
        )
        ref_conv = nn.Conv2d(
            self.C_in, 8, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)
        )
        clone_params(my_conv, ref_conv)

        my_out = my_conv(x)
        ref_out = ref_conv(x)
        assert my_out.shape == ref_out.shape
        assert torch.allclose(my_out, ref_out, atol=1e-6)

    def test_output_shape_formula(self):
        """Verify spatial output size matches the documented formula."""
        C_in, C_out, k, s, p, d = 3, 8, 3, 2, 1, 1
        H_in, W_in = 32, 32
        H_out_exp = (H_in + 2 * p - d * (k - 1) - 1) // s + 1
        W_out_exp = (W_in + 2 * p - d * (k - 1) - 1) // s + 1

        layer = MyConv2d(C_in, C_out, k, stride=s, padding=p, dilation=d)
        x = torch.randn(1, C_in, H_in, W_in)
        out = layer(x)
        assert out.shape == (1, C_out, H_out_exp, W_out_exp), (
            f"expected (1, {C_out}, {H_out_exp}, {W_out_exp}), got {out.shape}"
        )


# ============================================================================
# 3. Backward Pass — Gradient Flow
# ============================================================================


class TestBackward:
    N, C_in, H, W = 2, 3, 16, 16

    @pytest.fixture
    def x(self):
        torch.manual_seed(42)
        return torch.randn(self.N, self.C_in, self.H, self.W, requires_grad=True)

    def test_weight_grad_shape(self, x):
        layer = MyConv2d(self.C_in, 8, 3, padding=1)
        out = layer(x).sum()
        out.backward()
        assert layer.weight.grad.shape == layer.weight.shape, (
            f"expected {layer.weight.shape}, got {layer.weight.grad.shape}"
        )

    def test_bias_grad_shape(self, x):
        layer = MyConv2d(self.C_in, 8, 3, padding=1, bias=True)
        out = layer(x).sum()
        out.backward()
        assert layer.bias.grad is not None
        assert layer.bias.grad.shape == layer.bias.shape, (
            f"expected {layer.bias.shape}, got {layer.bias.grad.shape}"
        )

    def test_input_grad_shape(self, x):
        layer = MyConv2d(self.C_in, 8, 3, padding=1)
        out = layer(x).sum()
        out.backward()
        assert x.grad.shape == x.shape, f"expected {x.shape}, got {x.grad.shape}"

    def test_grad_equals_ref(self, x):
        """Compare our grad against torch.nn.Conv2d's grad."""
        layer = MyConv2d(self.C_in, 8, 3, padding=1)
        ref_layer = nn.Conv2d(self.C_in, 8, 3, padding=1)
        clone_params(layer, ref_layer)

        # forward + backward on our layer
        out = layer(x).sum()
        out.backward()
        our_w_grad = layer.weight.grad.clone()
        our_b_grad = layer.bias.grad.clone()

        # forward + backward on ref
        with torch.no_grad():
            ref_layer.weight.grad = None
            ref_layer.bias.grad = None
        ref_out = ref_layer(x).sum()
        ref_out.backward()
        ref_w_grad = ref_layer.weight.grad
        ref_b_grad = ref_layer.bias.grad

        assert torch.allclose(our_w_grad, ref_w_grad, atol=1e-6), (
            f"weight grad max diff = {(our_w_grad - ref_w_grad).abs().max().item():.2e}"
        )
        assert torch.allclose(our_b_grad, ref_b_grad, atol=1e-6), (
            f"bias grad max diff = {(our_b_grad - ref_b_grad).abs().max().item():.2e}"
        )

    def test_no_bias_grad(self, x):
        layer = MyConv2d(self.C_in, 8, 3, padding=1, bias=False)
        out = layer(x).sum()
        out.backward()
        assert layer.bias is None
        # Should not error — bias just doesn't exist
        assert layer.weight.grad is not None


# ============================================================================
# 4. Edge Cases & Oddities
# ============================================================================


class TestEdgeCases:
    def test_single_channel(self):
        layer = MyConv2d(1, 4, 3, padding=1)
        x = torch.randn(1, 1, 8, 8)
        out = layer(x)
        assert out.shape == (1, 4, 8, 8)

    def test_single_pixel_input(self):
        """A single-pixel input with k=1 should reduce to a linear layer."""
        layer = MyConv2d(3, 8, kernel_size=1)
        x = torch.randn(4, 3, 1, 1)
        out = layer(x)
        assert out.shape == (4, 8, 1, 1)

    def test_kernel_larger_than_input(self):
        """When kernel > input (no padding), unfold raises RuntimeError."""
        layer = MyConv2d(3, 8, kernel_size=5)
        x = torch.randn(1, 3, 4, 4)
        with pytest.raises(
            RuntimeError, match="calculated shape.*must be at least one"
        ):
            _ = layer(x)

    def test_zero_padding_same_output_size(self):
        """With k=3, p=1, s=1, output spatial dims should equal input."""
        layer = MyConv2d(3, 16, 3, padding=1)
        x = torch.randn(2, 3, 32, 32)
        out = layer(x)
        assert out.shape == (2, 16, 32, 32)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_various_batch_sizes(self, batch_size):
        layer = MyConv2d(3, 8, 3, padding=1)
        x = torch.randn(batch_size, 3, 8, 8)
        out = layer(x)
        assert out.shape == (batch_size, 8, 8, 8)

    def test_device_and_dtype(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        layer = MyConv2d(3, 16, 3, padding=1, device="cuda", dtype=torch.float32)
        x = torch.randn(2, 3, 16, 16, device="cuda", dtype=torch.float32)
        out = layer(x)
        assert out.device.type == "cuda"
        assert out.dtype == torch.float32
