"""
tests/cv/test_unet.py — Tests for U-Net building blocks (DownBlock, UpBlock).

Verifies:
  - DownBlock: output shapes, skip semantics, gradient flow.
  - UpBlock: output shapes, channel/spatial transform, skip usage, gradient flow.
"""

import pytest
import torch
from core.cv import DownBlock, UpBlock

# ============================================================================
# 1. Forward — Shape Correctness
# ============================================================================


class TestShape:
    N, H, W = 4, 32, 32

    @pytest.mark.parametrize(
        "in_c,out_c,residual,note",
        [
            (64, 64, True, "same channels, residual"),
            (64, 128, True, "channel expand, residual"),
            (64, 64, False, "same channels, no residual"),
            (64, 128, False, "channel expand, no residual"),
            (3, 64, True, "RGB input, residual"),
        ],
    )
    def test_output_shapes(self, in_c, out_c, residual, note):
        """Both return tensors have correct shapes."""
        block = DownBlock(in_c, out_c, use_residual=residual)
        x = torch.randn(self.N, in_c, self.H, self.W)
        out, skip = block(x)

        expect_out = (self.N, out_c, self.H // 2, self.W // 2)
        expect_skip = (self.N, out_c, self.H, self.W)
        assert out.shape == expect_out, (
            f"[{note}] out: expected {expect_out}, got {list(out.shape)}"
        )
        assert skip.shape == expect_skip, (
            f"[{note}] skip: expected {expect_skip}, got {list(skip.shape)}"
        )


# ============================================================================
# 2. Skip Semantics
# ============================================================================


class TestSkipSemantics:
    """skip is the pre-pool feature map, out is the post-pool version."""

    def test_skip_is_pre_pool(self):
        """If out is the pooled version of skip, then downsampling by 2×
        with MaxPool feels the same as skipping every other element of skip
        *only* when pool lands on a local max.  Instead, verify that skip
        is spatially larger and has the same channel count."""
        block = DownBlock(64, 128)
        x = torch.randn(2, 64, 16, 16)
        out, skip = block(x)

        assert skip.shape[2:] == (16, 16), "skip should be pre-pool (H,W)"
        assert out.shape[2:] == (8, 8), "out should be post-pool (H/2,W/2)"
        assert skip.shape[1] == out.shape[1] == 128, "same channels"

    def test_non_residual_skip_has_relu(self):
        """Without residual, the skip should still be ReLU-activated
        (i.e., non-negative)."""
        block = DownBlock(64, 64, use_residual=False)
        x = torch.randn(2, 64, 8, 8)
        _, skip = block(x)
        assert (skip >= 0).all(), "skip should be non-negative (ReLU applied)"


# ============================================================================
# 3. Backward — Gradient Flow
# ============================================================================


class TestBackward:
    N, C, H, W = 2, 64, 16, 16

    @pytest.mark.parametrize("residual", [True, False])
    def test_grad_flows(self, residual):
        """Gradient flows through both (out, skip) paths."""
        block = DownBlock(self.C, self.C, use_residual=residual)
        x = torch.randn(self.N, self.C, self.H, self.W, requires_grad=True)

        out, skip = block(x)
        loss = out.sum() + skip.sum()
        loss.backward()

        assert x.grad is not None, "input grad is None"
        assert x.grad.shape == x.shape, f"expected {x.shape}, got {x.grad.shape}"
        for name, p in block.named_parameters():
            assert p.grad is not None, f"{name} grad is None"
            assert p.grad.shape == p.shape, f"{name} grad shape mismatch"

    @pytest.mark.parametrize("residual", [True, False])
    def test_out_only_grad(self, residual):
        """Loss only from the downsampled output (decoder path)."""
        block = DownBlock(self.C, self.C * 2, use_residual=residual)
        x = torch.randn(self.N, self.C, self.H, self.W, requires_grad=True)

        out, _ = block(x)
        out.sum().backward()

        assert x.grad is not None
        for name, p in block.named_parameters():
            assert p.grad is not None, f"{name} grad is None"

    @pytest.mark.parametrize("residual", [True, False])
    def test_skip_only_grad(self, residual):
        """Loss only from the skip connection (encoder→decoder direct)."""
        block = DownBlock(self.C, self.C, use_residual=residual)
        x = torch.randn(self.N, self.C, self.H, self.W, requires_grad=True)

        _, skip = block(x)
        skip.sum().backward()

        assert x.grad is not None
        for name, p in block.named_parameters():
            assert p.grad is not None, f"{name} grad is None"

    def test_skip_projection_grad(self):
        """When in_c != out_c, the 1×1 skip projection also gets gradient."""
        block = DownBlock(64, 128)
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        out, _ = block(x)
        out.sum().backward()

        # The skip_proj weight must have non-zero gradient
        assert block.skip_proj.weight is not None
        assert block.skip_proj.weight.grad is not None
        assert block.skip_proj.weight.grad.abs().sum().item() > 0, (
            "skip projection grad is all-zero — maybe not connected?"
        )


# ============================================================================
# UpBlock Tests
# ============================================================================


class TestUpBlockShape:
    N, H, W = 2, 16, 16

    @pytest.mark.parametrize(
        "in_c,out_c,note",
        [
            (1024, 512, "bottleneck → decoder"),
            (512, 256, "mid decoder"),
            (64, 64, "same channels"),
            (128, 3, "final decoder → output"),
        ],
    )
    def test_output_shapes(self, in_c, out_c, note):
        """Output has correct channels and 2× spatial size."""
        up = UpBlock(in_c, out_c)
        x = torch.randn(self.N, in_c, self.H, self.W)
        skip = torch.randn(self.N, out_c, self.H * 2, self.W * 2)
        y = up(x, skip)

        expected = (self.N, out_c, self.H * 2, self.W * 2)
        assert y.shape == expected, f"[{note}] expected {expected}, got {list(y.shape)}"

    def test_without_skip_returns_different(self):
        """Changing skip content changes output (skip is actually used)."""
        up = UpBlock(64, 32)
        x = torch.randn(2, 64, 8, 8)
        skip_a = torch.randn(2, 32, 16, 16)
        skip_b = torch.randn(2, 32, 16, 16) + 100.0  # very different

        out_a = up(x, skip_a)
        out_b = up(x, skip_b)
        # Outputs should differ by more than numerical noise
        diff = (out_a - out_b).abs().mean().item()
        assert diff > 0.1, f"skip should influence output, diff={diff:.6f}"

    def test_identity_when_skip_is_zero(self):
        """When skip is zero and x is controlled, the block still processes."""
        up = UpBlock(64, 32)
        x = torch.randn(2, 64, 8, 8)
        skip = torch.zeros(2, 32, 16, 16)
        y = up(x, skip)
        assert y.shape == (2, 32, 16, 16)
        assert not torch.isnan(y).any(), "output has NaN on zero skip"


# ============================================================================
# UpBlock — Gradient Flow
# ============================================================================


class TestUpBlockGrad:
    N, H, W = 2, 8, 8

    @pytest.mark.parametrize("in_c,out_c", [(1024, 512), (64, 64)])
    def test_grad_flows_from_output(self, in_c, out_c):
        """Gradient flows through up_conv and conv, into both inputs."""
        up = UpBlock(in_c, out_c)
        x = torch.randn(self.N, in_c, self.H, self.W, requires_grad=True)
        skip = torch.randn(self.N, out_c, self.H * 2, self.W * 2, requires_grad=True)

        y = up(x, skip)
        y.sum().backward()

        assert x.grad is not None, "decoder input x has no grad"
        assert skip.grad is not None, "skip input has no grad"
        assert x.grad.shape == x.shape
        assert skip.grad.shape == skip.shape
        for name, p in up.named_parameters():
            assert p.grad is not None, f"{name} grad is None"
            assert p.grad.shape == p.shape, f"{name} grad shape mismatch"
