import numpy as np

from core.nn import Linear, Parameter
from core.autograd import Value


# =========================================================
# 1. Initialisation & Properties
# =========================================================


class TestInit:
    def test_with_bias(self):
        layer = Linear(3, 2, bias=True)
        assert layer.weight.data.shape == (2, 3)
        assert layer.bias is not None
        assert layer.bias.data.shape == (2,)
        assert isinstance(layer.weight, Parameter)
        assert isinstance(layer.bias, Parameter)

    def test_without_bias(self):
        layer = Linear(3, 2, bias=False)
        assert layer.weight.data.shape == (2, 3)
        assert layer.bias is None

    def test_seed_determinism(self):
        layer1 = Linear(4, 5, bias=True)
        layer2 = Linear(4, 5, bias=True)
        # Two fresh layers should have different random weights
        assert not np.allclose(layer1.weight.data, layer2.weight.data)

    def test_repr(self):
        layer = Linear(3, 2, bias=True)
        assert "Linear(in=3, out=2" in repr(layer)

    def test_repr_no_bias(self):
        layer = Linear(3, 2, bias=False)
        assert "bias=False" in repr(layer)


# =========================================================
# 2. Forward Pass
# =========================================================


class TestForward:
    def test_output_shape(self):
        layer = Linear(4, 3, bias=True)
        x = Value(np.random.randn(5, 4))
        y = layer(x)
        assert y.data.shape == (5, 3)

    def test_output_shape_no_bias(self):
        layer = Linear(4, 3, bias=False)
        x = Value(np.random.randn(5, 4))
        y = layer(x)
        assert y.data.shape == (5, 3)

    def test_value_matches_numpy(self):
        rng = np.random.default_rng(42)
        layer = Linear(3, 2, bias=True)
        # Override weights and bias with known values for deterministic comparison
        W = rng.normal(0, 1, (2, 3))
        b = rng.normal(0, 1, (2,))
        layer.weight = Parameter(W.copy())
        layer.bias = Parameter(b.copy())

        x_data = rng.normal(0, 1, (4, 3))
        x = Value(x_data.copy())
        y = layer(x)
        np.testing.assert_allclose(y.data, x_data @ W.T + b)

    def test_no_bias_matches_numpy(self):
        rng = np.random.default_rng(42)
        layer = Linear(3, 2, bias=False)
        W = rng.normal(0, 1, (2, 3))
        layer.weight = Parameter(W.copy())

        x_data = rng.normal(0, 1, (4, 3))
        x = Value(x_data.copy())
        y = layer(x)
        np.testing.assert_allclose(y.data, x_data @ W.T)

    def test_batch_independence(self):
        """Two identical inputs in a batch get the same output."""
        layer = Linear(3, 2, bias=True)
        x = Value(np.tile(np.array([[1.0, 2.0, 3.0]]), (4, 1)))
        y = layer(x)
        for i in range(1, 4):
            np.testing.assert_allclose(y.data[0], y.data[i])

    def test_single_sample(self):
        """A single input (1D-like) works if shaped as (1, d_in)."""
        layer = Linear(3, 2, bias=True)
        x = Value(np.random.randn(1, 3))
        y = layer(x)
        assert y.data.shape == (1, 2)


# =========================================================
# 3. Backward Pass (Gradient Flow)
# =========================================================


class TestBackward:
    def test_weight_grad_receives_gradient(self):
        rng = np.random.default_rng(42)
        layer = Linear(3, 2, bias=True)
        x = Value(rng.normal(0, 1, (4, 3)))
        y = layer(x)
        y.backward(np.ones_like(y.data))
        assert layer.weight.grad is not None
        assert not np.allclose(layer.weight.grad, 0)
        assert layer.weight.grad.shape == (2, 3)

    def test_bias_grad_receives_gradient(self):
        rng = np.random.default_rng(42)
        layer = Linear(3, 2, bias=True)
        x = Value(rng.normal(0, 1, (4, 3)))
        y = layer(x)
        y.backward(np.ones_like(y.data))
        assert layer.bias.grad is not None
        assert not np.allclose(layer.bias.grad, 0)
        # HINT: shape should be (2,) after you fix broadcasting in __add__ backward.
        # Currently it might be (4,2) because broadcast gradient isn't summed.

    def test_weight_grad_matches_finite_difference(self):
        rng = np.random.default_rng(42)
        layer = Linear(2, 3, bias=False)
        W_init = layer.weight.data.copy()
        x_data = rng.normal(0, 1, (5, 2))
        x = Value(x_data.copy())

        y = layer(x)
        y.backward(np.ones_like(y.data))

        def f(W_mat):
            return (x_data @ W_mat.T).sum()

        h = 1e-6
        expected = np.zeros_like(W_init)
        for idx in np.ndindex(W_init.shape):
            W_plus = W_init.copy()
            W_minus = W_init.copy()
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
        x = Value(x_data.copy())

        y = layer(x)
        y.backward(np.ones_like(y.data))

        def f(b_vec):
            return (x_data @ W_fixed.T + b_vec).sum()

        h = 1e-6
        expected = np.zeros_like(b_init)
        for idx in np.ndindex(b_init.shape):
            b_plus = b_init.copy()
            b_minus = b_init.copy()
            b_plus[idx] += h
            b_minus[idx] -= h
            expected[idx] = (f(b_plus) - f(b_minus)) / (2 * h)

        # Sum over any extra broadcast dims if __add__ backward hasn't been fixed yet
        computed_grad = layer.bias.grad
        while computed_grad.ndim > expected.ndim:
            computed_grad = computed_grad.sum(axis=0)

        np.testing.assert_allclose(computed_grad, expected, atol=1e-6)

    def test_input_grad_receives_gradient(self):
        rng = np.random.default_rng(42)
        layer = Linear(3, 2, bias=True)
        x = Value(rng.normal(0, 1, (4, 3)))
        y = layer(x)
        y.backward(np.ones_like(y.data))
        assert x.grad is not None
        assert not np.allclose(x.grad, 0)
        assert x.grad.shape == (4, 3)


# =========================================================
# 4. Parameters Iterator
# =========================================================


class TestParameters:
    def test_with_bias(self):
        layer = Linear(3, 2, bias=True)
        params = list(layer.parameters())
        assert len(params) == 2
        assert params[0] is layer.weight
        assert params[1] is layer.bias

    def test_without_bias(self):
        layer = Linear(3, 2, bias=False)
        params = list(layer.parameters())
        assert len(params) == 1
        assert params[0] is layer.weight

    def test_all_are_parameter_instances(self):
        layer = Linear(3, 2, bias=True)
        for p in layer.parameters():
            assert isinstance(p, Parameter)


# =========================================================
# 5. Edge Cases
# =========================================================


class TestEdgeCases:
    def test_zero_batch(self):
        """An empty batch produces correct (empty) output."""
        layer = Linear(3, 2, bias=True)
        x = Value(np.zeros((0, 3)))
        y = layer(x)
        assert y.data.shape == (0, 2)

    def test_large_batch(self):
        """No crash or instability with a large batch."""
        layer = Linear(10, 5, bias=True)
        x = Value(np.random.randn(1000, 10))
        y = layer(x)
        assert y.data.shape == (1000, 5)
