import numpy as np
import pytest
from core.autograd import Value
from core.nn import Linear, Parameter

# =========================================================
# 1. Initialisation & Properties
# =========================================================


class TestInit:
    @pytest.mark.parametrize("bias", [True, False])
    def test_weight_shapes(self, bias):
        layer = Linear(3, 2, bias=bias)
        assert layer.weight.data.shape == (2, 3)
        assert isinstance(layer.weight, Parameter)
        if bias:
            assert layer.bias is not None
            assert layer.bias.data.shape == (2,)
            assert isinstance(layer.bias, Parameter)
        else:
            assert layer.bias is None

    def test_repr(self):
        assert "Linear(in=3, out=2, bias=True)" in repr(Linear(3, 2, bias=True))
        assert "bias=False" in repr(Linear(3, 2, bias=False))


# =========================================================
# 2. Forward Pass
# =========================================================


class TestForward:
    @pytest.mark.parametrize("bias", [True, False])
    def test_output_shape(self, bias):
        layer = Linear(4, 3, bias=bias)
        y = layer(Value(np.random.randn(5, 4)))
        assert y.data.shape == (5, 3)

    @pytest.mark.parametrize("bias", [True, False])
    def test_value_matches_numpy(self, bias):
        rng = np.random.default_rng(42)
        layer = Linear(3, 2, bias=bias)
        W = rng.normal(0, 1, (2, 3))
        b = rng.normal(0, 1, (2,)) if bias else None
        layer.weight = Parameter(W.copy())
        if bias:
            layer.bias = Parameter(b.copy())

        x_data = rng.normal(0, 1, (4, 3))
        y = layer(Value(x_data.copy()))
        ref = x_data @ W.T + (b if bias else 0)
        np.testing.assert_allclose(y.data, ref)


# =========================================================
# 3. Backward Pass (Gradient Flow)
# =========================================================


class TestBackward:
    def test_weight_and_bias_have_correct_grad_shapes(self):
        rng = np.random.default_rng(42)
        layer = Linear(3, 2, bias=True)
        y = layer(Value(rng.normal(0, 1, (4, 3))))
        y.backward(np.ones_like(y.data))
        assert layer.weight.grad.shape == (2, 3)
        assert layer.bias.grad.shape == (2,)

    def test_weight_grad_matches_finite_difference(self):
        rng = np.random.default_rng(42)
        layer = Linear(2, 3, bias=False)
        W_init = layer.weight.data.copy()
        x_data = rng.normal(0, 1, (5, 2))
        y = layer(Value(x_data.copy()))
        y.backward(np.ones_like(y.data))

        def f(W_mat):
            return (x_data @ W_mat.T).sum()

        h = 1e-6
        expected = np.zeros_like(W_init)
        for idx in np.ndindex(W_init.shape):
            W_plus, W_minus = W_init.copy(), W_init.copy()
            W_plus[idx] += h
            W_minus[idx] -= h
            expected[idx] = (f(W_plus) - f(W_minus)) / (2 * h)

        np.testing.assert_allclose(layer.weight.grad, expected, atol=1e-6)

    def test_bias_grad_matches_finite_difference(self):
        rng = np.random.default_rng(42)
        layer = Linear(2, 3, bias=True)
        b_init = layer.bias.data.copy()
        W_fixed = layer.weight.data.copy()
        x_data = rng.normal(0, 1, (5, 2))
        y = layer(Value(x_data.copy()))
        y.backward(np.ones_like(y.data))

        def f(b_vec):
            return (x_data @ W_fixed.T + b_vec).sum()

        h = 1e-6
        expected = np.zeros_like(b_init)
        for idx in np.ndindex(b_init.shape):
            b_plus, b_minus = b_init.copy(), b_init.copy()
            b_plus[idx] += h
            b_minus[idx] -= h
            expected[idx] = (f(b_plus) - f(b_minus)) / (2 * h)

        np.testing.assert_allclose(layer.bias.grad, expected, atol=1e-6)

    def test_input_grad_has_correct_shape(self):
        rng = np.random.default_rng(42)
        layer = Linear(3, 2, bias=True)
        x = Value(rng.normal(0, 1, (4, 3)))
        y = layer(x)
        y.backward(np.ones_like(y.data))
        assert x.grad.shape == (4, 3)


# =========================================================
# 4. Parameters Iterator
# =========================================================


class TestParameters:
    @pytest.mark.parametrize("bias,count", [(True, 2), (False, 1)])
    def test_parameters_count_and_type(self, bias, count):
        params = list(Linear(3, 2, bias=bias).parameters())
        assert len(params) == count
        for p in params:
            assert isinstance(p, Parameter)


# =========================================================
# 5. Edge Cases
# =========================================================


class TestEdgeCases:
    def test_zero_batch(self):
        layer = Linear(3, 2, bias=True)
        y = layer(Value(np.zeros((0, 3))))
        assert y.data.shape == (0, 2)

    def test_large_batch(self):
        layer = Linear(10, 5, bias=True)
        y = layer(Value(np.random.randn(1000, 10)))
        assert y.data.shape == (1000, 5)
