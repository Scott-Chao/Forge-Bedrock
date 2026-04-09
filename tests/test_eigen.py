import pytest
import numpy as np

from core.linalg.matrix import Matrix, QRDecomposition, EigenSolver

# =========================================================
# Fixtures (Generate test data)
# =========================================================


@pytest.fixture
def random_square_matrix():
    """Returns a random square matrix for QR decomposition tests."""
    n = np.random.randint(4, 8)
    arr = np.random.rand(n, n)
    return arr, Matrix(arr)


@pytest.fixture
def random_tall_matrix():
    """Returns a random tall matrix (rows > cols) to test QR decomposition generalization."""
    n, m = np.random.randint(6, 10), np.random.randint(3, 5)
    arr = np.random.rand(n, m)
    return arr, Matrix(arr)


@pytest.fixture
def random_symmetric_matrix():
    """
    Returns a random symmetric matrix.
    Symmetric matrices ensure real eigenvalues and stable convergence
    for the QR algorithm and Power Iteration.
    """
    n = np.random.randint(4, 7)
    # A^T * A + I ensures a symmetric positive-definite matrix
    arr = np.random.rand(n, n)
    arr = arr.T @ arr + np.eye(n)
    return arr, Matrix(arr)


# =========================================================
# 1. QR Decomposition Correctness Tests
# =========================================================


def test_qr_decomposition_square(random_square_matrix):
    arr, m = random_square_matrix
    qr = QRDecomposition(m)
    Q, R = qr.Q, qr.R

    # 1. Q must be an orthogonal matrix: Q^T @ Q ≈ I
    I_expected = np.eye(Q.rows)
    np.testing.assert_allclose((Q.T @ Q).data, I_expected, atol=1e-10)

    # 2. R must be an upper triangular matrix (elements below main diagonal are 0)
    R_lower_tri = np.tril(R.data, -1)
    np.testing.assert_allclose(R_lower_tri, 0.0, atol=1e-10)

    # 3. Reconstruct: A = Q @ R
    np.testing.assert_allclose((Q @ R).data, arr, atol=1e-10)


def test_qr_decomposition_tall(random_tall_matrix):
    arr, m = random_tall_matrix
    qr = QRDecomposition(m)
    Q, R = qr.Q, qr.R

    # Q is still a full orthogonal matrix: Q^T @ Q ≈ I (Q is n x n)
    I_expected = np.eye(Q.rows)
    np.testing.assert_allclose((Q.T @ Q).data, I_expected, atol=1e-10)

    # R is n x m and should maintain upper triangular form
    R_lower_tri = np.tril(R.data, -1)
    np.testing.assert_allclose(R_lower_tri, 0.0, atol=1e-10)

    # Reconstruct: A = Q @ R
    np.testing.assert_allclose((Q @ R).data, arr, atol=1e-10)


# =========================================================
# 2. Eigenvalue Solver Correctness Tests (EigenSolver)
# =========================================================


def test_power_iteration(random_symmetric_matrix):
    arr, m = random_symmetric_matrix

    # Use custom power iteration
    dom_eigenvalue, dom_eigenvector = m.eig(method="power")

    # Compare with Numpy's standard implementation
    np_eigvals = np.linalg.eigvals(arr)
    # Get the dominant eigenvalue (largest absolute value)
    np_dom_eigenvalue = max(np_eigvals, key=abs)

    # Verify eigenvalue proximity
    np.testing.assert_allclose(dom_eigenvalue, np_dom_eigenvalue, rtol=1e-5)

    # Verify eigenvector definition: A @ v = lambda * v
    Av = m @ dom_eigenvector
    lv = dom_eigenvalue * dom_eigenvector
    np.testing.assert_allclose(Av.data, lv.data, atol=1e-5)


def test_qr_algorithm_eigenvalues(random_symmetric_matrix):
    arr, m = random_symmetric_matrix

    # Use custom QR algorithm
    eigenvalues, eigenvectors = m.eig(method="qr")

    # Compare with Numpy's standard implementation
    np_eigvals = np.linalg.eigvals(arr)

    # Since the return order might differ, sort before comparing
    sorted_custom = np.sort(eigenvalues)
    sorted_np = np.sort(np_eigvals)

    np.testing.assert_allclose(sorted_custom, sorted_np, rtol=1e-4, atol=1e-4)


def test_eigen_exceptions():
    # Only square matrices can have eigenvalues
    m_rect = Matrix.ones(3, 4)
    with pytest.raises(ValueError, match="EigenSolver requires a square matrix"):
        m_rect.eig()

    # Handle unknown methods
    m_square = Matrix.eye(3)
    with pytest.raises(ValueError, match="Unknown eigenvalue method"):
        m_square.eig(method="magic")


# =========================================================
# 3. Performance Benchmarking (Pytest-Benchmark)
# Compare custom Householder/Power/QR-Alg vs Native Numpy (LAPACK)
# =========================================================

# Matrix sizes for benchmarking
BENCH_SIZE_QR = 64
BENCH_SIZE_EIG = 16


@pytest.fixture
def bench_data_qr():
    arr = np.random.rand(BENCH_SIZE_QR, BENCH_SIZE_QR)
    return arr, Matrix(arr)


@pytest.fixture
def bench_data_eig():
    arr = np.random.rand(BENCH_SIZE_EIG, BENCH_SIZE_EIG)
    arr = arr.T @ arr + np.eye(BENCH_SIZE_EIG)  # Ensure SPD for faster convergence
    return arr, Matrix(arr)


# -- QR Benchmark --
def test_bench_qr_custom(benchmark, bench_data_qr):
    """Benchmark: Custom Householder QR Decomposition"""
    _, m = bench_data_qr
    benchmark(lambda: QRDecomposition(m))


def test_bench_qr_numpy(benchmark, bench_data_qr):
    """Benchmark: Native Numpy QR Decomposition (BLAS/LAPACK)"""
    arr, _ = bench_data_qr
    benchmark(lambda: np.linalg.qr(arr))


# -- Power Iteration Benchmark --
def test_bench_eigen_power_custom(benchmark, bench_data_eig):
    """Benchmark: Custom Power Iteration for dominant eigenvalue"""
    _, m = bench_data_eig
    benchmark(lambda: m.eig(method="power"))


# -- QR Algorithm for all eigenvalues Benchmark --
def test_bench_eigen_qr_alg_custom(benchmark, bench_data_eig):
    """Benchmark: Custom QR Algorithm for all eigenvalues"""
    _, m = bench_data_eig
    benchmark(lambda: m.eig(method="qr"))


def test_bench_eigen_numpy(benchmark, bench_data_eig):
    """Benchmark: Native Numpy eigenvalue solver (BLAS/LAPACK)"""
    arr, _ = bench_data_eig
    # Numpy finds all eigenvalues and eigenvectors simultaneously
    benchmark(lambda: np.linalg.eig(arr))
