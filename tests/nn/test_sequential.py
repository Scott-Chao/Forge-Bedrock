import pytest
import numpy as np

from core.nn import Sequential, Linear, ReLU, Tanh, Parameter
from core.autograd import Value


class TestInit:
    def test_repr(self):
        model = Sequential([Linear(3, 2), ReLU()])
        s = repr(model)
        assert "Sequential" in s
        assert "(0): Linear(in=3, out=2, bias=True)" in s
        assert "(1): ReLU()" in s


class TestForward:
    def test_shape_multi_layer(self):
        model = Sequential([Linear(4, 8), ReLU(), Linear(8, 2)])
        y = model(Value(np.random.randn(3, 4)))
        assert y.data.shape == (3, 2)

    def test_matches_manual_forward(self):
        rng = np.random.default_rng(42)
        l1 = Linear(3, 4, bias=True)
        relu = ReLU()
        l2 = Linear(4, 2, bias=False)

        model = Sequential([l1, relu, l2])
        x = Value(rng.normal(0, 1, (6, 3)))
        y_model = model(x)

        y_manual = l2(relu(l1(x)))
        np.testing.assert_allclose(y_model.data, y_manual.data)


class TestBackward:
    def test_gradients_flow_through_all_layers(self):
        rng = np.random.default_rng(42)
        l1 = Linear(3, 4, bias=True)
        l2 = Linear(4, 2, bias=False)
        model = Sequential([l1, ReLU(), l2])

        x = Value(rng.normal(0, 1, (5, 3)))
        y = model(x)
        y.backward(np.ones_like(y.data))

        assert l1.weight.grad.shape == (4, 3)
        assert l1.bias.grad.shape == (4,)
        assert l2.weight.grad.shape == (2, 4)
        assert x.grad.shape == (5, 3)

    def test_weight_grad_matches_finite_difference(self):
        rng = np.random.default_rng(42)
        l1 = Linear(2, 3, bias=True)
        model = Sequential([l1, ReLU()])
        W_init = l1.weight.data.copy()
        x_data = rng.normal(0, 1, (5, 2))

        y = model(Value(x_data.copy()))
        y.backward(np.ones_like(y.data))

        def f(W_mat):
            l1.weight.data = W_mat
            return model(Value(x_data.copy())).data.sum()

        h = 1e-6
        expected = np.zeros_like(W_init)
        for idx in np.ndindex(W_init.shape):
            W_plus, W_minus = W_init.copy(), W_init.copy()
            W_plus[idx] += h
            W_minus[idx] -= h
            expected[idx] = (f(W_plus) - f(W_minus)) / (2 * h)
        l1.weight.data = W_init.copy()

        np.testing.assert_allclose(l1.weight.grad, expected, atol=1e-6)


class TestParameters:
    def test_parameter_count(self):
        model = Sequential([Linear(3, 4, bias=True), ReLU(), Linear(4, 2, bias=False)])
        params = list(model.parameters())
        assert len(params) == 3
        for p in params:
            assert isinstance(p, Parameter)

    def test_empty_sequential_has_no_parameters(self):
        assert list(Sequential([]).parameters()) == []


class TestEdgeCases:
    def test_zero_batch(self):
        model = Sequential([Linear(3, 2), ReLU()])
        y = model(Value(np.zeros((0, 3))))
        assert y.data.shape == (0, 2)

    def test_indexing(self):
        l1, l2, l3 = Linear(3, 4), ReLU(), Linear(4, 2)
        model = Sequential([l1, l2, l3])
        assert model[0] is l1
        assert model[1] is l2
        assert model[-1] is l3
        assert len(model) == 3
