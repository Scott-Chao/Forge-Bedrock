import numpy as np
from core.nn import Parameter
from core.nn.regularizer import l1_penalty, l2_penalty

# =========================================================
# 1. L2 regularizer
# =========================================================


class TestL2Penalty:
    def test_forward_value(self):
        """L2 = lambda/2 * sum(param²)."""
        p = Parameter(np.array([3.0, 4.0]))
        penalty = l2_penalty([p], lambda_=0.1)
        expected = 0.1 / 2 * (9.0 + 16.0)
        assert abs(penalty.data - expected) < 1e-10

    def test_backward_gradient(self):
        """dL2/dparam = lambda * param added to param.grad."""
        p = Parameter(np.array([3.0, -4.0]))
        p.grad = np.array([0.0, 0.0])
        penalty = l2_penalty([p], lambda_=0.1)
        p.grad = 0.0  # reset
        penalty.backward()
        np.testing.assert_allclose(p.grad, 0.1 * np.array([3.0, -4.0]))

    def test_lambda_zero_penalty_zero(self):
        """lambda=0 → penalty=0 and no gradient contribution."""
        p = Parameter(np.array([5.0, -2.0]))
        p.grad = np.array([1.0, 1.0])
        penalty = l2_penalty([p], lambda_=0.0)
        assert penalty.data == 0.0
        penalty.backward()
        np.testing.assert_array_equal(p.grad, np.array([1.0, 1.0]))


# =========================================================
# 2. L1 regularizer
# =========================================================


class TestL1Penalty:
    def test_forward_value(self):
        """L1 = lambda * sum(|param|)."""
        p = Parameter(np.array([3.0, -4.0]))
        penalty = l1_penalty([p], lambda_=0.1)
        expected = 0.1 * (3.0 + 4.0)
        assert abs(penalty.data - expected) < 1e-10

    def test_backward_gradient(self):
        """dL1/dparam = lambda * sign(param)."""
        p = Parameter(np.array([3.0, -4.0, 0.0]))
        p.grad = np.array([0.0, 0.0, 0.0])
        penalty = l1_penalty([p], lambda_=0.1)
        p.grad = 0.0
        penalty.backward()
        np.testing.assert_allclose(p.grad, 0.1 * np.array([1.0, -1.0, 0.0]))


# =========================================================
# 3. Edge cases (shared)
# =========================================================


class TestEdgeCases:
    def test_empty_parameters(self):
        """No parameters → penalty is 0."""
        assert l2_penalty([], lambda_=0.1).data == 0.0
        assert l1_penalty([], lambda_=0.1).data == 0.0

    def test_multi_param_aggregation(self):
        """Penalty aggregates over multiple parameters."""
        p1 = Parameter(np.array([1.0, 2.0]))
        p2 = Parameter(np.array([3.0]))
        penalty = l2_penalty([p1, p2], lambda_=0.5)
        expected = 0.5 / 2 * (1.0 + 4.0 + 9.0)
        assert abs(penalty.data - expected) < 1e-10
