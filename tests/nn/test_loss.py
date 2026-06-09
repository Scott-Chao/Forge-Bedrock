import pytest
import numpy as np

from core.autograd import Value
from core.nn import MSELoss


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
# 1. Forward — MSE computation
# =========================================================


class TestForward:
    def test_perfect_prediction(self):
        pred = Value(np.array([1.0, 2.0, 3.0]))
        target = Value(np.array([1.0, 2.0, 3.0]))
        loss = MSELoss()(pred, target)
        assert loss.data == 0.0

    @pytest.mark.parametrize(
        "pred, target, expected",
        [
            (np.array([0.0, 0.0]), np.array([1.0, 1.0]), 1.0),  # (1+1)/2=1
            (np.array([2.0, 3.0]), np.array([1.0, 1.0]), 2.5),  # (1+4)/2=2.5
        ],
    )
    def test_mse_values(self, pred, target, expected):
        loss = MSELoss()(Value(pred), Value(target))
        np.testing.assert_allclose(loss.data, expected)


# =========================================================
# 2. Backward — gradient against finite difference
# =========================================================


class TestBackward:
    @pytest.mark.parametrize("shape", [(3,), (4, 1), (2, 3)])
    def test_gradient_matches_finite_difference(self, shape):
        rng = np.random.default_rng(42)
        pred_data = rng.normal(0, 1, shape)
        target_data = rng.normal(0, 1, shape)

        pred = Value(pred_data.copy())
        target = Value(target_data.copy())
        loss = MSELoss()(pred, target)
        loss.backward()

        def f(p):
            return MSELoss()(Value(p), Value(target_data)).data

        expected = _numerical_grad_array(f, pred_data)
        np.testing.assert_allclose(pred.grad, expected, atol=1e-6)

    def test_loss_is_scalar(self):
        pred = Value(np.random.randn(4, 3))
        target = Value(np.random.randn(4, 3))
        loss = MSELoss()(pred, target)
        assert isinstance(loss.data, (int, float, np.floating))


# =========================================================
# 3. Module integration
# =========================================================


class TestIntegration:
    def test_parameters_empty(self):
        loss = MSELoss()
        params = list(loss.parameters())
        assert params == []

    def test_repr(self):
        assert repr(MSELoss()) == "MSELoss()"

    def test_in_training_loop(self):
        """Verify loss drops after a simple gradient step on Linear."""
        from core.nn import Linear

        rng = np.random.default_rng(42)
        layer = Linear(2, 1)
        x = Value(rng.normal(0, 1, (4, 2)))
        target = Value(rng.normal(0, 1, (4, 1)))
        loss_fn = MSELoss()

        loss_before = loss_fn(layer(x), target).data

        loss_fn(layer(x), target).backward()
        for p in layer.parameters():
            p.data -= 0.1 * p.grad
        layer.zero_grad()

        loss_after = loss_fn(layer(x), target).data
        assert loss_after <= loss_before * 1.01, "loss should not increase"
