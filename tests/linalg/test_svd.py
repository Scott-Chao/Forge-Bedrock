import pytest
import numpy as np
from core.linalg import Matrix, SVD, PCA


# ---------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------
@pytest.fixture
def tall_matrix():
    # 10x4 matrix
    return Matrix(np.random.randn(10, 4))


@pytest.fixture
def wide_matrix():
    # 4x10 matrix
    return Matrix(np.random.randn(4, 10))


@pytest.fixture
def square_singular_matrix():
    # A 3x3 matrix where the 3rd row is a sum of the first two
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]])
    return Matrix(data)


def assert_is_orthogonal(M: Matrix, tol=1e-9):
    """Checks if M.T @ M is Identity."""
    # Note: For non-square orthogonal matrices (tall),
    # M.T @ M = I, but M @ M.T != I.
    rows, cols = M.shape
    identity = np.eye(cols)
    res = (M.T @ M).data
    np.testing.assert_allclose(res, identity, atol=tol)


# ---------------------------------------------------------
# 1. SVD Decomposition Tests
# ---------------------------------------------------------
@pytest.mark.parametrize("m, n", [(5, 5), (8, 3), (3, 8)])
def test_svd_reconstruction_and_orthogonality(m, n):
    """Verify A = U @ Sigma @ VT and U/V orthogonality."""
    A_raw = np.random.randn(m, n)
    A = Matrix(A_raw)

    svd = SVD(A, full_matrices=True)
    U, Sigma, VT = svd.U, svd.S, svd.VT

    # 1. Check Dimensions
    assert U.shape == (m, m)
    assert Sigma.shape == (m, n)
    assert VT.shape == (n, n)

    # 2. Check Orthogonality
    assert_is_orthogonal(U)
    assert_is_orthogonal(VT.T)

    # 3. Check Reconstruction: A = U @ Sigma @ VT
    A_reconstructed = U @ Sigma @ VT
    np.testing.assert_allclose(A_reconstructed.data, A_raw, atol=1e-8)


def test_svd_reduced():
    """Verify Reduced SVD dimensions."""
    m, n = 10, 4
    A = Matrix(np.random.randn(m, n))
    svd = SVD(A, full_matrices=False)

    # Reduced: U is (m, n), Sigma is (n, n), VT is (n, n)
    assert svd.U.shape == (10, 4)
    assert svd.S.shape == (4, 4)
    assert svd.VT.shape == (4, 4)

    # Still reconstructs
    recon = (svd.U @ svd.S @ svd.VT).data
    np.testing.assert_allclose(recon, A.data, atol=1e-8)


# ---------------------------------------------------------
# 2. Moore-Penrose Pseudoinverse Tests
# ---------------------------------------------------------
def test_pinv_square_invertible():
    """For invertible matrices, pinv should match inv."""
    A_raw = np.random.randn(3, 3)
    A = Matrix(A_raw)
    pinv_A = A.pinv
    inv_A = np.linalg.inv(A_raw)
    np.testing.assert_allclose(pinv_A.data, inv_A, atol=1e-8)


def test_pinv_fundamental_properties(tall_matrix):
    """Verify the Penrose conditions: A @ A+ @ A = A."""
    A = tall_matrix
    A_plus = A.pinv

    # Condition 1: A @ A+ @ A = A
    res1 = A @ A_plus @ A
    np.testing.assert_allclose(res1.data, A.data, atol=1e-8)

    # Condition 2: A+ @ A @ A+ = A+
    res2 = A_plus @ A @ A_plus
    np.testing.assert_allclose(res2.data, A_plus.data, atol=1e-8)


def test_pinv_singular(square_singular_matrix):
    """Verify pinv handles singular matrices correctly."""
    A = square_singular_matrix
    pinv_custom = A.pinv.data
    pinv_numpy = np.linalg.pinv(A.data)
    np.testing.assert_allclose(pinv_custom, pinv_numpy, atol=1e-8)


# ---------------------------------------------------------
# 3. PCA Tests
# ---------------------------------------------------------
def test_pca_fit_transform():
    # Create data with high correlation (y approx 2x)
    x = np.linspace(0, 10, 20)
    y = 2 * x + np.random.normal(0, 0.1, 20)
    data = np.column_stack([x, y])  # (20, 2)

    X = Matrix(data)
    pca = PCA(n_components=1)
    pca.fit(X)

    # Check components
    assert pca.components.shape == (1, 2)

    # Transform to 1D
    X_reduced = pca.transform(X)
    assert X_reduced.shape == (20, 1)

    # Variance ratio should be very high for the first component
    assert pca.explained_variance_ratio[0] > 0.95


def test_pca_inverse_transform():
    data = np.random.randn(50, 5)
    X = Matrix(data)

    n_comp = 3
    pca = PCA(n_components=n_comp).fit(X)
    X_red = pca.transform(X)
    X_recovered = pca.inverse_transform(X_red)

    # Note: Inverse transform won't perfectly match original data
    # if n_components < original features, but shape should match.
    assert X_recovered.shape == X.shape


# ---------------------------------------------------------
# 4. Performance Benchmarks
# ---------------------------------------------------------
# Using smaller sizes (32/64) because custom iterative Eigen/SVD
# is significantly slower than NumPy's C-compiled LAPACK.

BENCH_SIZE = 40


@pytest.fixture
def bench_matrix():
    return np.random.randn(BENCH_SIZE, BENCH_SIZE)


def test_bench_svd_custom(benchmark, bench_matrix):
    """Benchmark custom SVD implementation."""
    A = Matrix(bench_matrix)
    benchmark(lambda: SVD(A))


def test_bench_svd_numpy(benchmark, bench_matrix):
    """Benchmark NumPy's SVD (LAPACK)."""
    benchmark(lambda: np.linalg.svd(bench_matrix))


def test_bench_pinv_custom(benchmark, bench_matrix):
    """Benchmark custom Pseudoinverse."""
    A = Matrix(bench_matrix)
    benchmark(lambda: A.pinv)


def test_bench_pinv_numpy(benchmark, bench_matrix):
    """Benchmark NumPy's pinv."""
    benchmark(lambda: np.linalg.pinv(bench_matrix))


def test_bench_pca_custom(benchmark):
    """Benchmark custom PCA fit."""
    data = np.random.randn(100, 10)
    X = Matrix(data)
    pca = PCA(n_components=3)
    benchmark(lambda: pca.fit(X))
