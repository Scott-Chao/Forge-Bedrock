"""
tests/cv/test_resnet.py — Tests for BasicBlock & BottleneckBlock.

Verifies:
  - Output shape for same-dims and downsampling configurations.
  - Forward numerical correctness against a reference PyTorch ResNet.
  - Backward pass (gradient flow) through both blocks.
  - Edge cases (no stride, identity skip, 1×1 projection skip).
"""

import pytest
import torch
from core.cv import BasicBlock, BottleneckBlock, ResNet

# ============================================================================
# 1. Forward — Shape Correctness
# ============================================================================


class TestShape:
    """Verify spatial and channel dimensions for all block configs."""

    N, H, W = 2, 32, 32

    @pytest.mark.parametrize(
        "block_cls,cin,cout,stride,expected_spatial",
        [
            # same-dims: spatial unchanged
            (BasicBlock, 64, 64, 1, (32, 32)),
            (BottleneckBlock, 256, 256, 1, (32, 32)),
            # downsampling: spatial halved (stride=2)
            (BasicBlock, 64, 128, 2, (16, 16)),
            (BottleneckBlock, 256, 512, 2, (16, 16)),
            # non-square stride
            (BasicBlock, 64, 128, (2, 1), (16, 32)),
        ],
    )
    def test_output_shape(self, block_cls, cin, cout, stride, expected_spatial):
        block = block_cls(cin, cout, stride=stride)
        x = torch.randn(self.N, cin, self.H, self.W)
        out = block(x)
        H_exp, W_exp = expected_spatial
        assert out.shape == (self.N, cout, H_exp, W_exp), (
            f"expected (N={self.N}, C={cout}, H={H_exp}, W={W_exp}), got {out.shape}"
        )

    def test_expansion_attribute(self):
        assert BasicBlock.expansion == 1
        assert BottleneckBlock.expansion == 4

    def test_skip_is_identity_when_dims_match(self):
        """When no projection is needed, self.skip should be Identity."""
        block = BasicBlock(64, 64, stride=1)
        assert isinstance(block.skip, torch.nn.Identity), (
            "expected Identity skip for same-dims block"
        )

    def test_skip_is_conv_when_dims_differ(self):
        block = BasicBlock(64, 128, stride=2)
        from core.cv import Conv2d

        assert isinstance(block.skip, Conv2d), "expected Conv2d projection skip"
        assert block.skip.kernel_size == (1, 1)


# ============================================================================
# 2. Forward — Numerical Correctness (vs PyTorch ref)
# ============================================================================


class TestNumericalCorrectness:
    """Check that a Block built from our custom layers matches the same
    architecture built from torch.nn layers."""

    @pytest.fixture
    def x(self):
        torch.manual_seed(42)
        return torch.randn(2, 64, 16, 16)

    @staticmethod
    def _ref_basic_block(cin, cout, stride):
        """Reference BasicBlock using pure torch.nn modules."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(cout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(cout),
        )

    def test_basic_block_numerical(self, x):
        """Verify that the residual branch of BasicBlock matches a pure
        torch.nn stack (the skip connection itself is trivial to verify)."""
        cin, cout, stride = 64, 64, 1
        block = BasicBlock(cin, cout, stride=stride)

        # Build reference trunk with same weights
        ref_trunk = self._ref_basic_block(cin, cout, stride)
        with torch.no_grad():
            ref_trunk[0].weight.copy_(block.trunk[0].weight)
            ref_trunk[1].weight.copy_(block.trunk[1].weight)
            ref_trunk[1].bias.copy_(block.trunk[1].bias)
            ref_trunk[3].weight.copy_(block.trunk[3].weight)
            ref_trunk[4].weight.copy_(block.trunk[4].weight)
            ref_trunk[4].bias.copy_(block.trunk[4].bias)

        out = block(x)
        ref_out = ref_trunk(x) + x  # manual skip addition + ReLU
        ref_out = torch.relu(ref_out)

        assert torch.allclose(out, ref_out, atol=1e-5), (
            f"max diff = {(out - ref_out).abs().max().item():.2e}"
        )


# ============================================================================
# 3. Backward — Gradient Flow
# ============================================================================


class TestBackward:
    N, C, H, W = 2, 64, 8, 8

    def test_basic_block_grad_flows(self):
        x = torch.randn(self.N, self.C, self.H, self.W, requires_grad=True)
        block = BasicBlock(self.C, self.C)
        out = block(x).sum()
        out.backward()
        assert x.grad is not None, "input grad is None"
        assert x.grad.shape == x.shape, f"expected {x.shape}, got {x.grad.shape}"
        # Every parameter should have a non-None grad
        for name, p in block.named_parameters():
            assert p.grad is not None, f"{name} grad is None"
            assert p.grad.shape == p.shape, (
                f"{name} grad shape mismatch: {p.grad.shape}"
            )

    def test_bottleneck_grad_flows(self):
        x = torch.randn(self.N, self.C * 4, self.H, self.W, requires_grad=True)
        block = BottleneckBlock(self.C * 4, self.C * 4)
        out = block(x).sum()
        out.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        for name, p in block.named_parameters():
            assert p.grad is not None, f"{name} grad is None"

    def test_downsample_grad_flows(self):
        """stride=2 + channel change: 1×1 skip should also get gradient."""
        x = torch.randn(self.N, self.C, self.H, self.W, requires_grad=True)
        block = BasicBlock(self.C, self.C * 2, stride=2)
        out = block(x).sum()
        out.backward()
        # Both trunk and skip parameters should have gradients
        for name, p in block.named_parameters():
            assert p.grad is not None, f"{name} grad is None"
        assert x.grad.shape == x.shape


# ============================================================================
# 4. Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_minimal_spatial(self):
        """With stride=1, k=3, padding=1, spatial size is preserved
        even at small resolutions."""
        x = torch.randn(1, 64, 2, 2)
        block = BasicBlock(64, 64, stride=1)
        out = block(x)
        assert out.shape == (1, 64, 2, 2)

    def test_bottleneck_different_in_out_channels(self):
        """Bottleneck with different in/out channels and stride=2."""
        block = BottleneckBlock(128, 512, stride=2)
        x = torch.randn(1, 128, 16, 16)
        out = block(x)
        assert out.shape == (1, 512, 8, 8), f"expected (1, 512, 8, 8), got {out.shape}"


# ============================================================================
# 5. ResNet — Full Network
# ============================================================================


class TestResNetInit:
    """Construction and parameter correctness."""

    @pytest.mark.parametrize("depth", [18, 34, 50, 101, 152])
    def test_forward_shape(self, depth):
        """Output shape is (N, num_classes) for all variants."""
        rn = ResNet(depth=depth, num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        out = rn(x)
        assert out.shape == (2, 10), f"ResNet-{depth}: {out.shape}"

    # Known parameter counts from the ResNet paper.
    _EXPECTED_PARAMS = {
        18: 11.7,
        34: 21.8,
        50: 25.5,
        101: 44.5,
        152: 60.2,
    }

    @pytest.mark.parametrize("depth,expected_m", _EXPECTED_PARAMS.items())
    def test_parameter_count(self, depth, expected_m):
        rn = ResNet(depth=depth)
        total_m = sum(p.numel() for p in rn.parameters()) / 1e6
        assert abs(total_m - expected_m) < 0.2, (
            f"ResNet-{depth}: {total_m:.1f}M params, expected ~{expected_m}M"
        )

    def test_invalid_depth_raises(self):
        with pytest.raises(ValueError, match="Unsupported depth"):
            ResNet(depth=13)

    def test_repr(self):
        rn = ResNet(depth=50, num_classes=1000)
        r = repr(rn)
        assert "ResNet-50" in r
        assert "25.5M" in r or "25.6M" in r or "25.4M" in r, f"Unexpected repr: {r}"

    def test_custom_num_classes(self):
        rn = ResNet(depth=18, num_classes=7)
        x = torch.randn(1, 3, 224, 224)
        out = rn(x)
        assert out.shape == (1, 7)

    @pytest.mark.parametrize("depth", [18, 50])
    def test_zero_init_residual(self, depth):
        """All residual blocks' last BN weight is zero."""
        rn = ResNet(depth=depth, zero_init_residual=True)
        for m in rn.modules():
            if isinstance(m, BasicBlock):
                assert m.trunk[4].weight.abs().sum().item() == 0
            elif isinstance(m, BottleneckBlock):
                assert m.trunk[7].weight.abs().sum().item() == 0


class TestResNetForward:
    """Stage-by-stage spatial sizes."""

    _STAGE_SIZES = {
        18: [(64, 56, 56), (64, 56, 56), (128, 28, 28), (256, 14, 14), (512, 7, 7)],
        50: [(64, 56, 56), (256, 56, 56), (512, 28, 28), (1024, 14, 14), (2048, 7, 7)],
    }

    @pytest.mark.parametrize("depth,expected", _STAGE_SIZES.items())
    def test_stage_output_shapes(self, depth, expected):
        """Stem + 4 layers produce expected (C, H, W) at each stage."""
        model = ResNet(depth=depth, zero_init_residual=True)
        x = torch.randn(1, 3, 224, 224)

        # Run up to each named child & capture shape
        names = ["stem", "layer1", "layer2", "layer3", "layer4"]
        h = x
        for name, (c_exp, h_exp, w_exp) in zip(names, expected):
            h = getattr(model, name)(h)
            assert h.shape[1:] == (c_exp, h_exp, w_exp), (
                f"ResNet-{depth} {name}: expected ({c_exp},{h_exp},{w_exp}), "
                f"got {tuple(h.shape[1:])}"
            )


class TestResNetBackward:
    """Gradient flow through the full network."""

    @pytest.mark.parametrize("depth", [18, 50])
    def test_grad_flows(self, depth):
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        rn = ResNet(depth=depth, num_classes=10)
        out = rn(x).sum()
        out.backward()
        assert x.grad is not None, "input grad is None"
        assert x.grad.shape == x.shape
        # Every param in the network should receive a gradient
        for name, p in rn.named_parameters():
            assert p.grad is not None, f"{name} grad is None"
            assert p.grad.shape == p.shape, f"{name} shape mismatch"
