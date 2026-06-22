import numpy as np
import pytest
from core.autograd import Value
from core.autograd import functional as F
from core.nn import ReLU, Sigmoid, Tanh


def _numerical_grad(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


# =========================================================
# 1. Correctness — forward matches functional module
# =========================================================


class TestForward:
    @pytest.mark.parametrize(
        "cls, fn, x, expected",
        [
            (ReLU, F.relu, np.array([-2, -1, 0, 1, 2]), np.array([0, 0, 0, 1, 2])),
            (
                Tanh,
                F.tanh,
                np.array([-2, -1, 0, 1, 2]),
                np.tanh(np.array([-2, -1, 0, 1, 2])),
            ),
            (
                Sigmoid,
                F.sigmoid,
                np.array([-2, -1, 0, 1, 2]),
                1 / (1 + np.exp(-np.array([-2, -1, 0, 1, 2]))),
            ),
        ],
    )
    def test_forward_matches_functional(self, cls, fn, x, expected):
        wrapper = cls()
        v = Value(x.copy())
        out_wrapper = wrapper(v)
        out_fn = fn(Value(x.copy()))
        np.testing.assert_allclose(out_wrapper.data, expected)
        np.testing.assert_allclose(out_fn.data, expected)


# =========================================================
# 2. Backward — gradient matches finite difference
# =========================================================


class TestBackward:
    @pytest.mark.parametrize(
        "cls, fn_name", [(ReLU, "relu"), (Tanh, "tanh"), (Sigmoid, "sigmoid")]
    )
    def test_grad_matches_finite_difference(self, cls, fn_name):
        rng = np.random.default_rng(42)
        x_data = rng.normal(0, 1, (4, 3))
        x = Value(x_data.copy())
        wrapper = cls()
        y = wrapper(x)
        y.backward(np.ones_like(y.data))

        fn = getattr(F, fn_name)

        def f(arr):
            return fn(Value(arr)).data.sum()

        h = 1e-6
        expected = np.zeros_like(x_data)
        for idx in np.ndindex(x_data.shape):
            x_plus, x_minus = x_data.copy(), x_data.copy()
            x_plus[idx] += h
            x_minus[idx] -= h
            expected[idx] = (f(x_plus) - f(x_minus)) / (2 * h)

        np.testing.assert_allclose(x.grad, expected, atol=1e-6)


# =========================================================
# 3. Edge — scalar and 1D inputs
# =========================================================


class TestEdgeCases:
    @pytest.mark.parametrize("cls", [ReLU, Tanh, Sigmoid])
    def test_scalar_input(self, cls):
        wrapper = cls()
        v = Value(0.5)
        out = wrapper(v)
        assert isinstance(out, Value)
        assert out.data.shape == ()  # scalar

    @pytest.mark.parametrize("cls", [ReLU, Tanh, Sigmoid])
    def test_1d_input(self, cls):
        wrapper = cls()
        v = Value(np.array([-1.0, 0.0, 1.0]))
        out = wrapper(v)
        out.backward(np.ones_like(out.data))
        assert v.grad is not None
        assert v.grad.shape == (3,)

    @pytest.mark.parametrize(
        "cls, expected_repr",
        [(ReLU, "ReLU()"), (Tanh, "Tanh()"), (Sigmoid, "Sigmoid()")],
    )
    def test_repr(self, cls, expected_repr):
        assert repr(cls()) == expected_repr
