"""
tests/cv/test_conv_transpose.py — Tests for ConvTranspose2d (matmul + fold).
"""

import pytest
import torch
import torch.nn as nn
from core.cv import ConvTranspose2d as MyConvTranspose2d


def clone_params(src, dst):
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if src.bias is not None and dst.bias is not None:
            dst.bias.copy_(src.bias)


# ============================================================================
# 1. Initialisation
# ============================================================================


class TestInit:
    def test_weight_shape(self):
        layer = MyConvTranspose2d(3, 16, 3)
        assert layer.weight.shape == (3, 16, 3, 3)

    @pytest.mark.parametrize("bias", [True, False])
    def test_bias_parameter(self, bias):
        layer = MyConvTranspose2d(3, 8, 3, bias=bias)
        if bias:
            assert layer.bias is not None and layer.bias.shape == (8,)
        else:
            assert layer.bias is None

    def test_repr(self):
        layer = MyConvTranspose2d(3, 16, 3, stride=2, padding=1, dilation=2)
        r = repr(layer)
        assert "in_channels=3" in r and "out_channels=16" in r
        assert "kernel_size=(3, 3)" in r


# ============================================================================
# 2. Forward
# ============================================================================


class TestForward:
    @pytest.fixture
    def x(self):
        return torch.randn(2, 3, 8, 8)

    @pytest.mark.parametrize(
        "kw,s,p,op,d",
        [
            (3, 2, 1, 0, 1),
            (3, 2, 1, 1, 1),
            (4, 2, 1, 0, 1),
            (3, 1, 0, 0, 1),
            (3, 2, 1, 0, 2),
        ],
    )
    def test_output_matches_ref(self, x, kw, s, p, op, d):
        my_conv = MyConvTranspose2d(
            3, 8, kw, stride=s, padding=p, output_padding=op, dilation=d
        )
        ref_conv = nn.ConvTranspose2d(
            3, 8, kw, stride=s, padding=p, output_padding=op, dilation=d
        )
        clone_params(my_conv, ref_conv)
        assert torch.allclose(my_conv(x), ref_conv(x), atol=1e-6)

    def test_output_shape_formula(self):
        H_out = (8 - 1) * 2 - 2 * 1 + (3 - 1) + 1
        layer = MyConvTranspose2d(3, 8, 3, stride=2, padding=1)
        out = layer(torch.randn(1, 3, 8, 8))
        assert out.shape == (1, 8, H_out, H_out)


# ============================================================================
# 3. Backward
# ============================================================================


class TestBackward:
    def test_grad_shapes(self):
        x = torch.randn(2, 3, 8, 8, requires_grad=True)
        layer = MyConvTranspose2d(3, 8, 3, stride=2, padding=1)
        layer(x).sum().backward()
        assert x.grad.shape == x.shape
        assert layer.weight.grad.shape == layer.weight.shape
        assert layer.bias.grad.shape == layer.bias.shape

    def test_grad_equals_ref(self):
        x = torch.randn(2, 3, 8, 8)
        layer = MyConvTranspose2d(3, 8, 3, stride=2, padding=1)
        ref = nn.ConvTranspose2d(3, 8, 3, stride=2, padding=1)
        clone_params(layer, ref)

        x1 = x.clone().detach().requires_grad_(True)
        x2 = x.clone().detach().requires_grad_(True)
        layer(x1).sum().backward()
        ref(x2).sum().backward()

        assert torch.allclose(layer.weight.grad, ref.weight.grad, atol=5e-5)
        assert torch.allclose(layer.bias.grad, ref.bias.grad, atol=5e-5)
        assert torch.allclose(x1.grad, x2.grad, atol=5e-5)


# ============================================================================
# 4. Edge cases
# ============================================================================


class TestEdgeCases:
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_various_batch_sizes(self, batch_size):
        layer = MyConvTranspose2d(3, 8, 3, stride=2, padding=1)
        out = layer(torch.randn(batch_size, 3, 8, 8))
        assert out.shape == (batch_size, 8, 15, 15)

    def test_zero_bias_grad(self):
        """bias=False should not crash on backward."""
        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        layer = MyConvTranspose2d(3, 8, 3, stride=2, padding=1, bias=False)
        layer(x).sum().backward()
        assert layer.bias is None
        assert layer.weight.grad is not None
