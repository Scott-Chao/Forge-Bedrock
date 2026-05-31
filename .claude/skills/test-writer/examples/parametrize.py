"""Example parametrized tests — test same logic across multiple inputs."""
import operator

import numpy as np
import pytest

from core.linalg import Matrix


@pytest.mark.parametrize("m, n", [(5, 5), (8, 3), (3, 8)])
def test_parametrized_shapes(m, n):
    A = Matrix(np.random.randn(m, n))


operations = [
    (operator.add, np.add),
    (operator.sub, np.subtract),
]


@pytest.mark.parametrize("op, np_op", operations)
def test_binary_op(op, np_op, random_matrices):
    arr, m = random_matrices
    scalar = 2.5

    res_m = op(m, scalar)
    res_np = np_op(arr, scalar)
    np.testing.assert_allclose(res_m.data, res_np)
