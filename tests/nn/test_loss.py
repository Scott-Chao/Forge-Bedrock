import numpy as np
import pytest
from core.autograd import Value
from core.nn import (
    BCELoss,
    CrossEntropyLoss,
    HuberLoss,
    L1Loss,
    MSELoss,
)


def _numerical_grad_array(f, x0, h=1e-6):
    grad = np.zeros_like(x0)
    for idx in np.ndindex(x0.shape):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[idx] += h
        x_minus[idx] -= h
        grad[idx] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


# =========================================================
# Shared helper
# =========================================================


def _finite_diff_check(loss_fn, pred_data, target_data, atol=1e-6):
    """Assert that autograd gradient matches finite-difference approx."""
    pred = Value(pred_data.copy())
    target = Value(target_data.copy())
    loss = loss_fn(pred, target)
    loss.backward()

    def f(p):
        return loss_fn(Value(p), Value(target_data)).data

    expected = _numerical_grad_array(f, pred_data)
    np.testing.assert_allclose(pred.grad, expected, atol=atol)


# =========================================================
# 1. Forward — MSELoss
# =========================================================


class TestMSELossForward:
    def test_perfect_prediction(self):
        pred = Value(np.array([1.0, 2.0, 3.0]))
        target = Value(np.array([1.0, 2.0, 3.0]))
        loss = MSELoss()(pred, target)
        assert loss.data == 0.0

    def test_mse_values(self):
        loss = MSELoss()(Value(np.array([0.0, 0.0])), Value(np.array([1.0, 1.0])))
        np.testing.assert_allclose(loss.data, 1.0)


# =========================================================
# 2. Forward — L1Loss
# =========================================================


class TestL1LossForward:
    def test_perfect_prediction(self):
        loss = L1Loss()(Value(np.array([1.0, -2.0])), Value(np.array([1.0, -2.0])))
        assert loss.data == 0.0

    def test_l1_values(self):
        loss = L1Loss()(Value(np.array([2.0, 0.5])), Value(np.array([1.0, 1.0])))
        np.testing.assert_allclose(loss.data, 0.75)  # (|1|+|0.5|)/2


# =========================================================
# 3. Forward — HuberLoss
# =========================================================


class TestHuberLossForward:
    def test_perfect_prediction(self):
        loss = HuberLoss()(Value(np.array([5.0])), Value(np.array([5.0])))
        assert loss.data == 0.0

    def test_mse_region(self):
        """Inside delta, Huber behaves like MSE: 0.5 * d^2."""
        pred = Value(np.array([2.5]))
        target = Value(np.array([3.0]))
        loss = HuberLoss(delta=1.0)(pred, target)
        np.testing.assert_allclose(loss.data, 0.5 * 0.5**2)

    def test_l1_region(self):
        """Outside delta, Huber behaves like L1 minus offset."""
        pred = Value(np.array([5.0]))
        target = Value(np.array([0.0]))
        delta = 1.0
        loss = HuberLoss(delta=delta)(pred, target)
        # delta * |d| - 0.5 * delta^2 = 1*5 - 0.5 = 4.5
        np.testing.assert_allclose(loss.data, delta * 5.0 - 0.5 * delta**2)


# =========================================================
# 4. Forward — BCELoss
# =========================================================


class TestBCELossForward:
    @pytest.mark.parametrize("p, y", [(1.0, 1.0), (0.0, 0.0)])
    def test_perfect_prediction(self, p, y):
        loss = BCELoss()(Value(np.array([p])), Value(np.array([y])))
        assert loss.data < 1e-10

    def test_uncertain_prediction(self):
        """p=0.5, y=[1,0] → -(log(0.5) + log(0.5))/2 = -log(0.5) ≈ 0.6931."""
        loss = BCELoss()(Value(np.array([0.5, 0.5])), Value(np.array([1.0, 0.0])))
        np.testing.assert_allclose(loss.data, -np.log(0.5), rtol=1e-6)


# =========================================================
# 5. Forward — CrossEntropyLoss
# =========================================================


class TestCrossEntropyLossForward:
    def test_perfect_prediction(self):
        """Logits where correct class gets very high score, loss ≈ 0."""
        logits = Value(np.array([[1e6, 0.0, 0.0]]))
        target = Value(np.array([0]))
        loss = CrossEntropyLoss()(logits, target)
        assert loss.data < 1e-6

    def test_ce_values(self):
        logits = np.array([[0.0, 0.0]])
        loss = CrossEntropyLoss()(Value(logits), Value(np.array([0])))
        np.testing.assert_allclose(loss.data, 0.693147, rtol=1e-3)


# =========================================================
# 6. Backward — finite-difference gradient verification
# =========================================================


class TestBackward:
    def test_l1_gradient(self):
        rng = np.random.default_rng(42)
        pred_data = rng.normal(0, 1, (4,))
        target_data = rng.normal(0, 1, (4,))
        _finite_diff_check(L1Loss(), pred_data, target_data)

    @pytest.mark.parametrize("shape", [(4, 1)])
    def test_huber_gradient(self, shape):
        rng = np.random.default_rng(42)
        pred_data = rng.normal(0, 1, shape)
        target_data = rng.normal(0, 1, shape)
        _finite_diff_check(HuberLoss(delta=1.0), pred_data, target_data)

    def test_huber_gradient_mixed_region(self):
        """Mix of points inside and outside delta."""
        pred_data = np.array([0.1, 0.5, 2.0, -1.0, 3.0])
        target_data = np.zeros(5)
        _finite_diff_check(HuberLoss(delta=1.0), pred_data, target_data)

    @pytest.mark.parametrize("shape", [(4, 1)])
    def test_bce_gradient(self, shape):
        rng = np.random.default_rng(42)
        pred_data = rng.uniform(0.1, 0.9, shape)
        target_data = rng.integers(0, 2, shape).astype(float)
        _finite_diff_check(BCELoss(), pred_data, target_data)

    @pytest.mark.parametrize("n_classes", [3])
    def test_cross_entropy_gradient(self, n_classes):
        rng = np.random.default_rng(42)
        batch_size = 4
        logits_data = rng.normal(0, 1, (batch_size, n_classes))
        target_data = rng.integers(0, n_classes, batch_size)
        _finite_diff_check(CrossEntropyLoss(), logits_data, target_data, atol=1e-5)

    def test_mse_gradient(self):
        rng = np.random.default_rng(42)
        pred_data = rng.normal(0, 1, (2, 3))
        target_data = rng.normal(0, 1, (2, 3))
        _finite_diff_check(MSELoss(), pred_data, target_data)

    def test_loss_is_scalar(self):
        pred = Value(np.random.randn(4, 3))
        target = Value(np.random.randn(4, 3))
        for loss_fn in [MSELoss(), L1Loss(), HuberLoss()]:
            loss = loss_fn(pred, target)
            assert isinstance(loss.data, (int, float, np.floating))


# =========================================================
# 7. Edge cases
# =========================================================


class TestEdgeCases:
    def test_bce_clipping_prevents_nan(self):
        """BCE with extreme predictions (near 0 or 1) should not NaN."""
        loss_fn = BCELoss()
        # p=1e-15, y=0 → closer to 0-log(1) = 0, not NaN
        pred = Value(np.array([1e-15, 1 - 1e-15, 1.0, 0.0]))
        target = Value(np.array([0.0, 1.0, 1.0, 0.0]))
        loss = loss_fn(pred, target)
        assert np.isfinite(loss.data)

    def test_huber_large_delta_approximates_half_mse(self):
        """Large delta → all points quadratic, Huber ≈ 0.5 * MSE."""
        pred_data = np.array([2.0, -1.0, 0.5])
        target_data = np.array([0.0, 0.0, 0.0])
        ref = 0.5 * MSELoss()(Value(pred_data), Value(target_data)).data
        val = HuberLoss(delta=10.0)(Value(pred_data), Value(target_data)).data
        np.testing.assert_allclose(val, ref, rtol=1e-3)

    def test_cross_entropy_large_logits(self):
        """Very large/small logits should not overflow."""
        logits = Value(np.array([[1e6, -1e6, 0.0]]))
        target = Value(np.array([0]))
        loss = CrossEntropyLoss()(logits, target)
        assert np.isfinite(loss.data)

    def test_l1_negative_gradient_sign(self):
        """Gradient should be positive when pred > target."""
        pred = Value(np.array([5.0]))
        target = Value(np.array([3.0]))
        loss = L1Loss()(pred, target)
        loss.backward()
        assert pred.grad[0] > 0


# =========================================================
# 8. Module integration
# =========================================================


class TestIntegration:
    def test_repr(self):
        assert repr(L1Loss()) == "L1Loss()"
        assert repr(HuberLoss(delta=2.0)) == "HuberLoss(delta=2.0)"
        assert repr(CrossEntropyLoss()) == "CrossEntropyLoss()"

    def test_parameters_empty(self):
        for loss_fn in [L1Loss(), HuberLoss(), BCELoss(), CrossEntropyLoss()]:
            assert list(loss_fn.parameters()) == []
