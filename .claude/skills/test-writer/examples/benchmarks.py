"""Example performance benchmarks — compare custom vs numpy.

Keep benchmark sizes small (16-64) since custom Python implementations
are significantly slower than NumPy's C-compiled LAPACK.
"""
import numpy as np
import pytest

from core.linalg import Matrix

BENCH_SIZE = 64


@pytest.fixture
def bench_data():
    arr = np.random.rand(BENCH_SIZE, BENCH_SIZE)
    return arr, Matrix(arr)


def test_bench_custom(benchmark, bench_data):
    _, m = bench_data
    benchmark(lambda: m.some_op())


def test_bench_numpy(benchmark, bench_data):
    arr, _ = bench_data
    benchmark(lambda: np.linalg.some_op(arr))
