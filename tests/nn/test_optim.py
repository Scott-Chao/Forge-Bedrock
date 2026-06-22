import numpy as np
from core.nn import SGD
from core.nn.parameter import Parameter

# =========================================================
# 1. Correctness
# =========================================================


class TestSGDStep:
    def test_parameter_updated_in_negative_gradient_direction(self):
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        p.grad = np.array([0.1, -0.2, 0.3])
        optim = SGD([p], lr=0.5)

        optim.step()

        expected = np.array([1.0, 2.0, 3.0]) - 0.5 * np.array([0.1, -0.2, 0.3])
        np.testing.assert_allclose(p.data, expected)

    def test_zero_grad(self):
        p = Parameter(np.array([1.0, 2.0]))
        p.grad = np.array([0.5, -0.5])
        optim = SGD([p])

        optim.zero_grad()

        assert p.grad == 0.0
