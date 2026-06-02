import pytest
import numpy as np

from core.autograd import Value, relu, sigmoid, tanh


def _numerical_grad(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


CASES = [
    (relu, -1.0, 0.0, 0.0),
    (relu, 0.0, 0.0, 0.0),
    (relu, 2.0, 2.0, 1.0),
    (sigmoid, 0.0, 0.5, 0.25),
    (sigmoid, 100.0, 1.0, 1e-6),
    (tanh, 0.0, 0.0, 1.0),
    (tanh, 2.0, np.tanh(2), 0.0706508),
]


class TestActivations:
    @pytest.mark.parametrize("fn, val, expected_fwd, expected_bwd", CASES)
    def test_forward_and_backward(self, fn, val, expected_fwd, expected_bwd):
        x = Value(val)
        out = fn(x)
        np.testing.assert_allclose(out.data, expected_fwd, atol=1e-6)
        out.backward()
        np.testing.assert_allclose(x.grad, expected_bwd, atol=1e-6)

    @pytest.mark.parametrize(
        "fn, val",
        [
            (relu, 2.0),
            (sigmoid, 0.5),
            (tanh, 0.5),
        ],
    )
    def test_against_finite_diff(self, fn, val):
        x = Value(val)
        fn(x).backward()

        def f(v):
            return fn(Value(v)).data

        np.testing.assert_allclose(x.grad, _numerical_grad(f, val), atol=1e-5)

    @pytest.mark.parametrize(
        "expr, val",
        [
            ("tanh(relu(x))", 1.0),
            ("relu(x) * x + sigmoid(x)", 2.0),
        ],
    )
    def test_composed(self, expr, val):
        x = Value(val)
        y = eval(expr, {"x": x, "relu": relu, "sigmoid": sigmoid, "tanh": tanh})
        y.backward()

        def f(v):
            return eval(
                expr, {"x": Value(v), "relu": relu, "sigmoid": sigmoid, "tanh": tanh}
            ).data

        np.testing.assert_allclose(x.grad, _numerical_grad(f, val), atol=1e-5)
