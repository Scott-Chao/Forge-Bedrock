import numpy as np
import pytest
from core.nn import Parameter
from core.nn.clip import _total_grad_norm, clip_grad_norm_, clip_grad_value_

# =========================================================
# Fixtures
# =========================================================


@pytest.fixture
def multi_params():
    """Two active parameters + one frozen parameter (grad=None)."""
    p1 = Parameter(np.array([4.0, 5.0]))
    p1.grad = np.array([0.5, -1.5])
    p2 = Parameter(np.array([-1.0]))
    p2.grad = np.array([3.0])
    p3 = Parameter(np.array([10.0, 20.0]))
    p3.grad = None
    return [p1, p2, p3]


# =========================================================
# 1. clip_grad_value_
# =========================================================


class TestClipGradValue:
    def test_clamps_large_values(self):
        """Components exceeding threshold are clamped; in-range stay unchanged."""
        p = Parameter(np.array([0.0]))
        p.grad = np.array([100.0, 0.01, -200.0, 3.0])
        clip_grad_value_([p], threshold=10.0)

        expected = np.array([10.0, 0.01, -10.0, 3.0])
        np.testing.assert_array_equal(p.grad, expected)

    def test_invalid_threshold_raises(self):
        """threshold <= 0 raises ValueError."""
        p = Parameter(np.array([1.0]))
        p.grad = np.array([5.0])
        with pytest.raises(ValueError, match="threshold must be > 0"):
            clip_grad_value_([p], threshold=0.0)
        with pytest.raises(ValueError, match="threshold must be > 0"):
            clip_grad_value_([], threshold=-1.0)

    def test_skip_none_grad(self, multi_params):
        """Parameters with grad=None are left untouched."""
        clip_grad_value_(multi_params, threshold=1.0)
        assert multi_params[2].grad is None


# =========================================================
# 2. clip_grad_norm_
# =========================================================


class TestClipGradNorm:
    def test_scales_and_preserves_direction(self):
        """When norm exceeds max_norm, gradients are scaled down uniformly,
        keeping the original vector direction."""
        p = Parameter(np.array([0.0, 0.0]))
        p.grad = np.array([3.0, 4.0])  # L2 norm = 5.0

        returned = clip_grad_norm_([p], max_norm=2.0)

        # Norm clipped to max_norm
        new_norm = float(np.sqrt(np.sum(p.grad**2)))
        assert abs(new_norm - 2.0) < 1e-10

        # Direction preserved (unit vector unchanged)
        np.testing.assert_allclose(
            p.grad / new_norm, np.array([3.0, 4.0]) / 5.0, atol=1e-10
        )

        # Returns the original norm before clipping
        assert abs(returned - 5.0) < 1e-10

    def test_does_not_scale_when_within_limit(self):
        """When norm <= max_norm, gradients are unchanged."""
        p = Parameter(np.array([0.0, 0.0]))
        p.grad = np.array([0.5, 1.2])
        original = p.grad.copy()
        returned = clip_grad_norm_([p], max_norm=10.0)

        np.testing.assert_array_equal(p.grad, original)
        assert abs(returned - np.sqrt(0.25 + 1.44)) < 1e-10

    def test_aggregates_across_parameters(self, multi_params):
        """Multiple parameters share one norm; clipping scales all uniformly.
        None-grad parameters are ignored without error."""
        params = multi_params
        expected_norm = np.sqrt(0.5**2 + (-1.5) ** 2 + 3.0**2)

        returned = clip_grad_norm_(params, max_norm=1.0)

        assert abs(returned - expected_norm) < 1e-10

        # All active gradients were scaled
        combined_after = np.sqrt(
            np.sum(params[0].grad ** 2) + np.sum(params[1].grad ** 2)
        )
        assert abs(combined_after - 1.0) < 1e-10

        # Frozen parameter untouched
        assert params[2].grad is None

    def test_all_none_grad_returns_zero(self):
        """If every gradient is None, return 0.0, no error."""
        p1 = Parameter(np.array([1.0]))
        p1.grad = None
        p2 = Parameter(np.array([2.0]))
        p2.grad = None
        returned = clip_grad_norm_([p1, p2], max_norm=1.0)
        assert returned == 0.0


# =========================================================
# 3. _total_grad_norm (helper)
# =========================================================


class TestTotalGradNorm:
    def test_aggregates_multi_params(self, multi_params):
        """Computes combined L2 norm across all parameters with gradients."""
        norm = _total_grad_norm(multi_params, norm_type=2.0)
        expected = np.sqrt(0.5**2 + (-1.5) ** 2 + 3.0**2)
        assert abs(norm - expected) < 1e-10

    def test_empty_none_all_return_zero(self):
        """No gradients → norm is 0.0."""
        assert _total_grad_norm([]) == 0.0
        p = Parameter(np.array([1.0]))
        p.grad = None
        assert _total_grad_norm([p]) == 0.0
