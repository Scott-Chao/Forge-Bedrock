import operator

import numpy as np
import pytest
from core.linalg import Matrix


# ---------------------------------------------------------
# Fixtures (For generating test data)
# ---------------------------------------------------------
@pytest.fixture
def random_shape():
    return np.random.randint(2, 10), np.random.randint(2, 10)


@pytest.fixture
def random_matrices(random_shape):
    """Returns a Numpy array and a Matrix object of the same shape."""
    r, c = random_shape
    arr = np.random.rand(r, c)
    return arr, Matrix(arr)


# ---------------------------------------------------------
# 1. Initialization, Properties, and Basic Methods
# ---------------------------------------------------------
def test_init_and_properties():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    m = Matrix(arr)

    assert m.shape == (2, 3)
    assert m.rows == 2
    assert m.cols == 3
    np.testing.assert_array_equal(m.data, arr)
    assert "Matrix" in repr(m)


def test_getitem_setitem():
    m = Matrix.zeros(3, 3)
    m[1, 1] = 5.0
    assert m[1, 1] == 5.0
    assert m.data[1, 1] == 5.0


def test_transpose():
    arr = np.random.rand(4, 5)
    m = Matrix(arr)
    np.testing.assert_array_equal(m.T.data, arr.T)
    assert m.T.shape == (5, 4)


# ---------------------------------------------------------
# 2. Static Constructor Methods
# ---------------------------------------------------------
def test_static_constructors():
    # zeros
    m_zeros = Matrix.zeros(2, 3)
    np.testing.assert_array_equal(m_zeros.data, np.zeros((2, 3)))

    # ones
    m_ones = Matrix.ones(3, 2)
    np.testing.assert_array_equal(m_ones.data, np.ones((3, 2)))

    # eye
    m_eye = Matrix.eye(4)
    np.testing.assert_array_equal(m_eye.data, np.eye(4))

    # rand
    m_rand1 = Matrix.rand(2, 2, seed=42)
    m_rand2 = Matrix.rand(2, 2, seed=42)
    np.testing.assert_array_equal(m_rand1.data, m_rand2.data)


# ---------------------------------------------------------
# 3. Unary Operator Tests
# ---------------------------------------------------------
def test_unary_operations(random_matrices):
    arr, m = random_matrices

    # __neg__
    np.testing.assert_array_equal((-m).data, -arr)
    # __pos__
    np.testing.assert_array_equal((+m).data, +arr)
    # __abs__
    arr_neg, m_neg = -arr, -m
    np.testing.assert_array_equal(abs(m_neg).data, np.abs(arr_neg))


# ---------------------------------------------------------
# 4. Binary Operator Tests (Scalar and Matrix-Matrix)
# ---------------------------------------------------------
# Define operators to test: (symbol/operator, equivalent numpy function)
binary_ops = [
    (operator.add, np.add),
    (operator.sub, np.subtract),
    (operator.mul, np.multiply),
    (operator.truediv, np.divide),
    (operator.pow, np.power),
]


@pytest.mark.parametrize("op, np_op", binary_ops)
def test_binary_op_scalar(op, np_op, random_matrices):
    arr, m = random_matrices
    scalar = 2.5

    # Matrix op Scalar
    res_m = op(m, scalar)
    res_np = np_op(arr, scalar)
    np.testing.assert_allclose(res_m.data, res_np)

    # Scalar op Matrix (Reflected operations: __radd__, __rsub__, etc.)
    # Note: 'pow' with negative base and float exponent results in NaN,
    # so we use a strictly positive matrix for this specific test.
    arr_pos = np.abs(arr) + 1
    m_pos = Matrix(arr_pos)
    res_m_r = op(scalar, m_pos)
    res_np_r = np_op(scalar, arr_pos)
    np.testing.assert_allclose(res_m_r.data, res_np_r)


@pytest.mark.parametrize("op, np_op", binary_ops)
def test_binary_op_matrix(op, np_op):
    r, c = 3, 4
    arr1, m1_raw = np.random.rand(r, c) + 1, np.random.rand(r, c) + 1
    arr2 = arr1.copy()
    m1 = Matrix(m1_raw)
    m2 = Matrix(arr2)

    res_m = op(m1, m2)
    res_np = np_op(m1_raw, arr2)
    np.testing.assert_allclose(res_m.data, res_np)


# ---------------------------------------------------------
# 5. In-place Operator Tests (+=, -=, *=, /=)
# ---------------------------------------------------------
inplace_ops = [
    (operator.iadd, np.add),
    (operator.isub, np.subtract),
    (operator.imul, np.multiply),
    (operator.itruediv, np.divide),
]


@pytest.mark.parametrize("op, np_op", inplace_ops)
def test_inplace_op_scalar(op, np_op, random_matrices):
    arr, m = random_matrices
    scalar = 2.0

    op(m, scalar)  # Equivalent to: m += scalar
    np_op(arr, scalar, out=arr)  # Equivalent to: arr += scalar
    np.testing.assert_allclose(m.data, arr)


@pytest.mark.parametrize("op, np_op", inplace_ops)
def test_inplace_op_matrix(op, np_op):
    arr1, arr2 = np.random.rand(3, 3) + 1, np.random.rand(3, 3) + 1
    m1, m2 = Matrix(arr1.copy()), Matrix(arr2.copy())

    op(m1, m2)
    np_op(arr1, arr2, out=arr1)
    np.testing.assert_allclose(m1.data, arr1)


# ---------------------------------------------------------
# 6. Broadcasting Mechanism Tests (BroadcastEngine)
# ---------------------------------------------------------
broadcast_shapes = [
    ((3, 3), (1, 3)),  # Row broadcasting
    ((3, 3), (3, 1)),  # Column broadcasting
    ((3, 1), (1, 4)),  # Mutual broadcasting (Outer product behavior)
    ((2, 3, 4), (1, 3, 1)),  # High-dimensional broadcasting
]


@pytest.mark.parametrize("shape1, shape2", broadcast_shapes)
def test_broadcasting(shape1, shape2):
    arr1, arr2 = np.random.rand(*shape1), np.random.rand(*shape2)
    m1, m2 = Matrix(arr1), Matrix(arr2)

    # Test broadcasting via addition
    res_m = m1 + m2
    res_np = arr1 + arr2
    np.testing.assert_allclose(res_m.data, res_np)
    assert res_m.shape == res_np.shape


def test_broadcasting_inplace_error():
    # In-place operations should not change the original matrix shape
    m1 = Matrix.zeros(1, 3)
    m2 = Matrix.ones(3, 3)
    with pytest.raises(ValueError, match="Cannot broadcast to inplace output shape"):
        m1 += m2


def test_broadcasting_incompatible_error():
    m1 = Matrix.zeros(3, 3)
    m2 = Matrix.ones(2, 2)
    with pytest.raises(ValueError, match="Incompatible shapes"):
        m1 + m2


# ---------------------------------------------------------
# 7. Matrix Multiplication Tests (Matmul)
# ---------------------------------------------------------
def test_matmul():
    arr1 = np.random.rand(3, 4)
    arr2 = np.random.rand(4, 5)
    m1, m2 = Matrix(arr1), Matrix(arr2)

    # Using the @ operator
    res_m = m1 @ m2
    res_np = arr1 @ arr2
    np.testing.assert_allclose(res_m.data, res_np)

    # Test tiling logic effectiveness with a very small block size (S)
    res_m_small_S = m1.matmul(m2, S=2)
    np.testing.assert_allclose(res_m_small_S.data, res_np)


def test_matmul_incompatible_error():
    m1 = Matrix.zeros(3, 4)
    m2 = Matrix.zeros(5, 3)
    with pytest.raises(AssertionError, match="Matrix dimensions not compatible"):
        m1 @ m2


def test_unsupported_operand_error():
    m = Matrix.zeros(2, 2)
    with pytest.raises(TypeError, match="Unsupported operand type"):
        m + "string"


# =========================================================
# Performance Benchmarking (Pytest-Benchmark)
# Compare custom BroadcastEngine/Tiled Matmul vs Native Numpy
# =========================================================

# Increase size slightly to observe performance differences
BENCH_SIZE = 128


@pytest.fixture
def bench_data():
    arr1 = np.random.rand(BENCH_SIZE, BENCH_SIZE)
    arr2 = np.random.rand(BENCH_SIZE, BENCH_SIZE)
    return arr1, arr2, Matrix(arr1), Matrix(arr2)


def test_bench_add_custom(benchmark, bench_data):
    """Benchmark the custom BroadcastEngine addition."""
    _, _, m1, m2 = bench_data
    benchmark(lambda: m1 + m2)


def test_bench_add_numpy(benchmark, bench_data):
    """Benchmark native Numpy addition."""
    arr1, arr2, _, _ = bench_data
    benchmark(lambda: arr1 + arr2)


def test_bench_matmul_custom(benchmark, bench_data):
    """Benchmark custom 3-level loop tiled matrix multiplication."""
    _, _, m1, m2 = bench_data
    benchmark(lambda: m1 @ m2)


def test_bench_matmul_numpy(benchmark, bench_data):
    """Benchmark native Numpy (BLAS/LAPACK) matrix multiplication."""
    arr1, arr2, _, _ = bench_data
    benchmark(lambda: arr1 @ arr2)
