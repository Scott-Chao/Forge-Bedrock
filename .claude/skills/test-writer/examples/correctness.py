"""Example correctness test — compare custom impl against numpy as ground truth."""
import numpy as np

from core.linalg import Matrix


def test_some_operation(spd_matrix):
    arr, m = spd_matrix

    result = m.some_operation()
    expected = np.linalg.some_operation(arr)

    np.testing.assert_allclose(result.data, expected, atol=1e-10)
