import pytest
import numpy as np

from core.nn import Module, Parameter, Linear, Sequential, ReLU
from core.autograd import Value


# =========================================================
# 1. Parameter & Sub-module Registration
# =========================================================


class TestRegistration:
    def test_parameter_is_registered_via_setattr(self):
        class SimpleModel(Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter(np.ones((3,)))
                self.b = Parameter(np.zeros((2,)))

            def forward(self, x):
                return x

        m = SimpleModel()
        assert "w" in m._parameters
        assert "b" in m._parameters
        assert m._parameters["w"] is m.w
        assert m._parameters["b"] is m.b

    def test_submodule_is_registered_via_setattr(self):
        class NestedModel(Module):
            def __init__(self):
                super().__init__()
                self.fc = Linear(3, 2)
                self.act = ReLU()

            def forward(self, x):
                return self.act(self.fc(x))

        m = NestedModel()
        assert "fc" in m._modules
        assert "act" in m._modules
        assert m._modules["fc"] is m.fc
        assert m._modules["act"] is m.act

    def test_plain_attribute_not_registered(self):
        class SimpleModel(Module):
            def __init__(self):
                super().__init__()
                self.foo = 42
                self.bar = "hello"

            def forward(self, x):
                return x

        m = SimpleModel()
        assert len(m._parameters) == 0
        assert len(m._modules) == 0

    def test_existing_components_still_register_correctly(self):
        """La tilbake: Linear, Sequential og activation wrapperes
        arver Module og skal ha automatisk registrering."""
        layer = Linear(4, 3)
        assert "weight" in layer._parameters
        assert "bias" in layer._parameters

        seq = Sequential([Linear(4, 3), ReLU()])
        assert "0" in seq._modules
        assert "1" in seq._modules


# =========================================================
# 2. parameters() — Recursive Collection
# =========================================================


class TestParameters:
    def test_own_parameters(self):
        class SimpleModel(Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter(np.ones((3,)))
                self.b = Parameter(np.zeros((1,)))

            def forward(self, x):
                return x

        params = list(SimpleModel().parameters())
        assert len(params) == 2
        assert all(isinstance(p, Parameter) for p in params)

    def test_recursive_parameters_from_submodules(self):
        class DeepModel(Module):
            def __init__(self):
                super().__init__()
                self.fc1 = Linear(3, 4, bias=True)
                self.fc2 = Linear(4, 2, bias=False)

            def forward(self, x):
                return self.fc2(self.fc1(x))

        m = DeepModel()
        params = list(m.parameters())
        # fc1: weight + bias = 2, fc2: weight = 1
        assert len(params) == 3
        assert all(isinstance(p, Parameter) for p in params)

    def test_empty_module_has_no_parameters(self):
        m = Sequential([])
        assert list(m.parameters()) == []

    def test_parameters_are_unique(self):
        m = Linear(3, 2, bias=True)
        param_ids = [id(p) for p in m.parameters()]
        assert len(param_ids) == len(set(param_ids))

    def test_multiple_calls_return_same_objects(self):
        m = Linear(3, 2)
        p1 = list(m.parameters())
        p2 = list(m.parameters())
        for a, b in zip(p1, p2):
            assert a is b


# =========================================================
# 3. zero_grad()
# =========================================================


class TestZeroGrad:
    def test_zero_grad_sets_all_grads_to_zero(self):
        m = Linear(3, 2, bias=True)
        # Simulate non-zero gradients
        m.weight.grad = np.random.randn(2, 3)
        m.bias.grad = np.random.randn(2)

        m.zero_grad()

        np.testing.assert_array_equal(m.weight.grad, 0.0)
        np.testing.assert_array_equal(m.bias.grad, 0.0)

    def test_zero_grad_on_nested_model(self):
        model = Sequential([Linear(3, 4), ReLU(), Linear(4, 2)])
        for p in model.parameters():
            p.grad = np.random.randn(*p.data.shape)

        model.zero_grad()

        for p in model.parameters():
            np.testing.assert_array_equal(p.grad, 0.0)

    def test_zero_grad_no_params_does_not_crash(self):
        m = Sequential([])
        m.zero_grad()  # should not raise


# =========================================================
# 4. train() / eval() Mode
# =========================================================


class TestTrainEval:
    def test_default_mode_is_train(self):
        m = Linear(3, 2)
        assert m._training is True

    def test_eval_sets_training_false(self):
        m = Linear(3, 2)
        m.eval()
        assert m._training is False

    def test_train_restores_mode(self):
        m = Linear(3, 2)
        m.eval()
        m.train()
        assert m._training is True

    def test_train_mode_propagates_to_submodules(self):
        model = Sequential([Linear(3, 4), ReLU(), Linear(4, 2)])
        model.eval()
        for child in model._modules.values():
            assert child._training is False

        model.train()
        for child in model._modules.values():
            assert child._training is True

    def test_eval_returns_none_and_does_not_break(self):
        m = Linear(3, 2)
        assert m.eval() is None


# =========================================================
# 5. forward() and __call__()
# =========================================================


class TestForwardAndCall:
    def test_base_module_raises_not_implemented(self):
        m = Module()
        with pytest.raises(NotImplementedError, match="forward"):
            m(42)

    def test_subclass_forward_is_callable(self):
        m = Linear(3, 2)
        x = Value(np.random.randn(4, 3))
        y = m(x)
        assert y.data.shape == (4, 2)

    def test_call_delegates_to_forward(self):
        m = Linear(3, 2)
        x = Value(np.random.randn(4, 3))
        assert m(x).data.shape == m.forward(x).data.shape


# =========================================================
# 6. Inheritance — all existing layers are Modules
# =========================================================


class TestInheritance:
    @pytest.mark.parametrize(
        "cls, kwargs",
        [
            (Linear, {"in_features": 3, "out_features": 2}),
            (ReLU, {}),
            (Sequential, {"layers": []}),
        ],
    )
    def test_all_layers_inherit_from_module(self, cls, kwargs):
        assert issubclass(cls, Module)

    def test_sequential_with_layers_inherits_parameters(self):
        model = Sequential([Linear(3, 4, bias=True), Linear(4, 2, bias=False)])
        params = list(model.parameters())
        assert len(params) == 3  # 2 weights + 1 bias
        for p in params:
            assert isinstance(p, Parameter)
