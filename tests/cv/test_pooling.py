"""
tests/cv/test_pooling.py — Tests for MaxPool2d & AvgPool2d.

Compares output against ``torch.nn.MaxPool2d`` / ``torch.nn.AvgPool2d``
as reference and checks gradient flow through the pooling operation.
"""

import pytest
import torch
import torch.nn as nn
from core.cv import AvgPool2d as MyAvgPool2d
from core.cv import MaxPool2d as MyMaxPool2d

# ── common configs for parametrized tests ──────────────────────────────

POOL_CONFIGS = [
    pytest.param((2, 2), None, 0, id="k2_sNone_p0"),
    pytest.param((2, 2), 2, 0, id="k2_s2_p0"),
    pytest.param((3, 3), 1, 1, id="k3_s1_p1"),
    pytest.param((4, 4), 2, 0, id="k4_s2_p0"),
    pytest.param((3, 3), 2, 0, id="k3_s2_p0"),
    pytest.param((2, 2), 2, 1, id="k2_s2_p1"),
]

# ============================================================================
# 1. Initialisation & Properties
# ============================================================================


class TestInit:
    def test_kernel_size_normalisation(self):
        pool = MyMaxPool2d(3)
        assert pool.kernel_size == (3, 3)

    def test_stride_defaults_to_kernel_size(self):
        """PyTorch pooling convention: stride=None → stride = kernel_size."""
        pool = MyAvgPool2d(kernel_size=2)
        assert pool.stride == (2, 2)

    @pytest.mark.parametrize(
        "kw, expected",
        [
            ((2, 3), (2, 3)),
            (4, (4, 4)),
        ],
    )
    def test_asymmetric_kernel(self, kw, expected):
        pool = MyMaxPool2d(kw)
        assert pool.kernel_size == expected

    @pytest.mark.parametrize("stride", [None, 1, (2, 1)])
    def test_stride_normalisation(self, stride):
        pool = MyAvgPool2d(3, stride=stride)
        if stride is None:
            assert pool.stride == (3, 3)
        elif isinstance(stride, int):
            assert pool.stride == (stride, stride)
        else:
            assert pool.stride == stride

    def test_repr(self):
        pool = MyMaxPool2d(3, stride=2, padding=1)
        r = repr(pool)
        assert "kernel_size=(3, 3)" in r
        assert "stride=(2, 2)" in r
        assert "padding=(1, 1)" in r


# ============================================================================
# 2. Forward Pass — Numerical Correctness
# ============================================================================


class TestForward:
    N, C, H, W = 2, 4, 16, 16

    @pytest.fixture
    def x(self):
        torch.manual_seed(0)
        return torch.randn(self.N, self.C, self.H, self.W)

    @pytest.mark.parametrize("kernel_size,stride,padding", POOL_CONFIGS)
    def test_max_pool_matches_ref(self, x, kernel_size, stride, padding):
        my_pool = MyMaxPool2d(kernel_size, stride=stride, padding=padding)
        ref_pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

        my_out = my_pool(x)
        ref_out = ref_pool(x)

        assert my_out.shape == ref_out.shape, (
            f"shape: {my_out.shape} vs {ref_out.shape}"
        )
        assert torch.allclose(my_out, ref_out, atol=1e-6), (
            f"max diff = {(my_out - ref_out).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize("kernel_size,stride,padding", POOL_CONFIGS)
    def test_avg_pool_matches_ref(self, x, kernel_size, stride, padding):
        my_pool = MyAvgPool2d(kernel_size, stride=stride, padding=padding)
        ref_pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

        my_out = my_pool(x)
        ref_out = ref_pool(x)

        assert my_out.shape == ref_out.shape, (
            f"shape: {my_out.shape} vs {ref_out.shape}"
        )
        assert torch.allclose(my_out, ref_out, atol=1e-6), (
            f"max diff = {(my_out - ref_out).abs().max().item():.2e}"
        )

    def test_asymmetric_kernel_stride_padding(self, x):
        """Verify asymmetric (non-square) settings work correctly."""
        my_pool = MyMaxPool2d(kernel_size=(3, 5), stride=(2, 3), padding=(1, 2))
        ref_pool = nn.MaxPool2d(kernel_size=(3, 5), stride=(2, 3), padding=(1, 2))

        my_out = my_pool(x)
        ref_out = ref_pool(x)

        assert my_out.shape == ref_out.shape
        assert torch.allclose(my_out, ref_out, atol=1e-6)

    def test_output_shape_formula(self):
        """Verify spatial output matches the documented formula."""
        k, s, p = 3, 2, 1
        N, C, H_in, W_in = 2, 4, 32, 32
        H_out = (H_in + 2 * p - k) // s + 1
        W_out = (W_in + 2 * p - k) // s + 1

        pool = MyMaxPool2d(k, stride=s, padding=p)
        x = torch.randn(N, C, H_in, W_in)
        out = pool(x)

        assert out.shape == (N, C, H_out, W_out), (
            f"expected {(N, C, H_out, W_out)}, got {out.shape}"
        )


# ============================================================================
# 3. Backward Pass — Gradient Flow
# ============================================================================


class TestBackward:
    N, C, H, W = 2, 3, 8, 8

    @pytest.fixture
    def x(self):
        torch.manual_seed(42)
        return torch.randn(self.N, self.C, self.H, self.W, requires_grad=True)

    def test_max_pool_grad_shape(self, x):
        pool = MyMaxPool2d(2, stride=2)
        out = pool(x).sum()
        out.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape, f"expected {x.shape}, got {x.grad.shape}"

    def test_avg_pool_grad_shape(self, x):
        pool = MyAvgPool2d(2, stride=2)
        out = pool(x).sum()
        out.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape, f"expected {x.shape}, got {x.grad.shape}"

    def test_grad_matches_ref(self, x):
        """Compare our pooling gradient against torch.nn's gradient."""
        # MaxPool
        my_max = MyMaxPool2d(2, stride=2)
        ref_max = nn.MaxPool2d(2, stride=2)

        x_my = x.clone().detach().requires_grad_(True)
        x_ref = x.clone().detach().requires_grad_(True)

        my_max(x_my).sum().backward()
        ref_max(x_ref).sum().backward()

        assert torch.allclose(x_my.grad, x_ref.grad, atol=1e-6), (
            f"MaxPool grad diff: {(x_my.grad - x_ref.grad).abs().max().item():.2e}"
        )

        # AvgPool
        torch.manual_seed(43)
        x2 = torch.randn(self.N, self.C, self.H, self.W, requires_grad=True)

        my_avg = MyAvgPool2d(2, stride=2)
        ref_avg = nn.AvgPool2d(2, stride=2)

        x_my2 = x2.clone().detach().requires_grad_(True)
        x_ref2 = x2.clone().detach().requires_grad_(True)

        my_avg(x_my2).sum().backward()
        ref_avg(x_ref2).sum().backward()

        assert torch.allclose(x_my2.grad, x_ref2.grad, atol=1e-6), (
            f"AvgPool grad diff: {(x_my2.grad - x_ref2.grad).abs().max().item():.2e}"
        )


# ============================================================================
# 4. Edge Cases
# ============================================================================


class TestEdgeCases:
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_various_batch_sizes(self, batch_size):
        pool = MyMaxPool2d(2, stride=2)
        x = torch.randn(batch_size, 3, 8, 8)
        out = pool(x)
        assert out.shape == (batch_size, 3, 4, 4)

    def test_kernel_equals_input(self):
        """Kernel covering entire spatial region reduces to 1×1."""
        pool = MyMaxPool2d(kernel_size=8)
        x = torch.randn(2, 3, 8, 8)
        out = pool(x)
        assert out.shape == (2, 3, 1, 1)

    def test_non_overlapping_windows(self):
        """Stride equal to kernel_size → no overlap, exact tiles."""
        pool = MyAvgPool2d(kernel_size=4, stride=4)
        x = torch.randn(1, 2, 8, 8)
        out = pool(x)
        assert out.shape == (1, 2, 2, 2)
