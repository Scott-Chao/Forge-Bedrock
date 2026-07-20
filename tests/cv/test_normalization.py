"""
tests/cv/test_normalization.py — Tests for BatchNorm2d.
"""

import pytest
import torch
import torch.nn as nn
from core.cv import BatchNorm2d as MyBatchNorm2d


def clone_params(src: nn.Module, dst: nn.Module) -> None:
    with torch.no_grad():
        if src.weight is not None:
            dst.weight.copy_(src.weight)
        if src.bias is not None:
            dst.bias.copy_(src.bias)


# ============================================================================
# 1. Core correctness
# ============================================================================


class TestInit:
    @pytest.mark.parametrize(
        "num_features,affine,bias_shape",
        [(4, True, (4,)), (8, False, None)],
    )
    def test_construction(self, num_features, affine, bias_shape):
        bn = MyBatchNorm2d(num_features, affine=affine)
        assert bn.weight.shape == (num_features,) if affine else bn.weight is None
        assert bn.bias.shape == bias_shape if affine else bn.bias is None
        assert bn.running_mean.shape == (num_features,)
        assert bn.running_var.shape == (num_features,)
        assert bn.running_mean.allclose(torch.zeros(num_features))
        assert bn.running_var.allclose(torch.ones(num_features))


class TestForward:
    N, C, H, W = 4, 8, 16, 16

    @pytest.fixture
    def x(self):
        torch.manual_seed(0)
        return torch.randn(self.N, self.C, self.H, self.W)

    def test_train_mode(self, x):
        """Matches torch.nn.BatchNorm2d in train mode with no running stats."""
        my_bn = MyBatchNorm2d(self.C)
        ref_bn = nn.BatchNorm2d(self.C)
        clone_params(my_bn, ref_bn)
        my_bn.track_running_stats = False
        ref_bn.track_running_stats = False
        my_bn.reset_parameters()
        ref_bn.reset_parameters()

        my_out = my_bn(x)
        ref_out = ref_bn(x)
        assert my_out.shape == ref_out.shape
        assert torch.allclose(my_out, ref_out, atol=1e-6)

    def test_eval_mode(self, x):
        """In eval mode, normalises with injected running stats."""
        my_bn = MyBatchNorm2d(self.C)
        ref_bn = nn.BatchNorm2d(self.C)
        clone_params(my_bn, ref_bn)
        with torch.no_grad():
            my_bn.running_mean.copy_(torch.randn(self.C))
            my_bn.running_var.copy_(torch.rand(self.C).clamp_min(0.1))
            ref_bn.running_mean.copy_(my_bn.running_mean)
            ref_bn.running_var.copy_(my_bn.running_var)

        my_bn.eval()
        ref_bn.eval()
        my_out = my_bn(x)
        ref_out = ref_bn(x)
        assert torch.allclose(my_out, ref_out, atol=1e-6)

    def test_running_stats_update(self):
        """EMA update direction: running = lerp(running, batch, momentum)."""
        bn = MyBatchNorm2d(4, momentum=0.5)
        x = torch.randn(8, 4, 4, 4)
        batch_mean = x.mean(dim=(0, 2, 3))
        batch_var = x.var(dim=(0, 2, 3), correction=0)

        bn.train()
        bn(x)

        assert bn.running_mean.allclose(0.5 * batch_mean, atol=1e-6)
        assert bn.running_var.allclose(0.5 * torch.ones(4) + 0.5 * batch_var, atol=1e-6)

    @pytest.mark.parametrize("C", [1, 3, 32])
    def test_various_channels(self, C):
        """Output shape preserved for different channel counts."""
        bn = MyBatchNorm2d(C)
        x = torch.randn(2, C, 8, 8)
        assert bn(x).shape == (2, C, 8, 8)


# ============================================================================
# 2. Backward pass
# ============================================================================


class TestBackward:
    N, C, H, W = 2, 4, 8, 8

    @pytest.mark.parametrize("affine", [True, False])
    def test_gradient_flows(self, affine):
        x = torch.randn(self.N, self.C, self.H, self.W, requires_grad=True)
        bn = MyBatchNorm2d(self.C, affine=affine)
        bn(x).sum().backward()
        assert x.grad is not None and x.grad.shape == x.shape
        if affine:
            assert bn.weight.grad is not None and bn.weight.grad.shape == (self.C,)

    def test_grad_matches_ref(self):
        """Gradients match PyTorch's (unfused graph → fused kernel ≈ FP32)."""
        x = torch.randn(self.N, self.C, self.H, self.W, requires_grad=True)
        my_bn = MyBatchNorm2d(self.C)
        ref_bn = nn.BatchNorm2d(self.C)
        clone_params(my_bn, ref_bn)

        x1 = x.clone().detach().requires_grad_(True)
        x2 = x.clone().detach().requires_grad_(True)

        my_bn.train()
        ref_bn.train()
        my_bn.track_running_stats = False
        ref_bn.track_running_stats = False

        my_bn(x1).sum().backward()
        ref_bn(x2).sum().backward()

        assert x1.grad is not None and x2.grad is not None
        assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
        w_grad, ref_w_grad = my_bn.weight.grad, ref_bn.weight.grad
        assert w_grad is not None and ref_w_grad is not None
        assert torch.allclose(w_grad, ref_w_grad, atol=1e-5)
