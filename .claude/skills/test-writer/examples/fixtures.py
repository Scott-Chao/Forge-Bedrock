"""Example fixtures — adapted from tests/linalg/

Provide both a raw numpy array AND a Matrix object so tests can compare
against numpy directly as ground truth.
"""
import numpy as np
import pytest

from core.linalg import Matrix


@pytest.fixture
def random_shape():
    return np.random.randint(4, 8), np.random.randint(4, 8)


@pytest.fixture
def random_matrices(random_shape):
    r, c = random_shape
    arr = np.random.rand(r, c)
    return arr, Matrix(arr)


@pytest.fixture
def spd_matrix():
    """Symmetric Positive-Definite matrix via A^T A + I."""
    n = np.random.randint(4, 7)
    arr = np.random.rand(n, n)
    spd_arr = arr.T @ arr + np.eye(n)
    return spd_arr, Matrix(spd_arr)
