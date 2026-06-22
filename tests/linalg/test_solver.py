import numpy as np
import pytest
from core.linalg import LU, Cholesky, Matrix, TriangularSolver


# ---------------------------------------------------------
# Fixtures (For generating specific test matrices)
# ---------------------------------------------------------
@pytest.fixture
def square_matrix_data():
    """Returns a nicely conditioned 4x4 square matrix and its Matrix object."""
    np.random.seed(42)
    # Adding np.eye to ensure it's well-conditioned and non-singular
    arr = np.random.rand(4, 4) + np.eye(4)
    return arr, Matrix(arr)


@pytest.fixture
def spd_matrix_data():
    """Returns a Symmetric Positive-Definite (SPD) matrix and its Matrix object."""
    np.random.seed(42)
    arr = np.random.rand(4, 4)
    # A^T * A + I ensures the matrix is symmetric positive-definite
    spd_arr = arr.T @ arr + np.eye(4)
    return spd_arr, Matrix(spd_arr)


@pytest.fixture
def singular_matrix_data():
    """Returns a singular matrix (determinant = 0)."""
    arr = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],  # Row 2 is a multiple of Row 1
            [7.0, 8.0, 9.0],
        ]
    )
    return arr, Matrix(arr)


# ---------------------------------------------------------
# 1. Matrix Properties (Symmetry, Determinant, Inverse)
# ---------------------------------------------------------
def test_is_symmetric(spd_matrix_data, square_matrix_data):
    spd_arr, m_spd = spd_matrix_data
    arr, m_nonsym = square_matrix_data

    assert m_spd.is_symmetric is True
    assert m_nonsym.is_symmetric is False


def test_determinant(square_matrix_data, singular_matrix_data):
    arr, m = square_matrix_data
    np.testing.assert_allclose(m.det, np.linalg.det(arr))

    _, m_sing = singular_matrix_data
    np.testing.assert_allclose(m_sing.det, 0.0, atol=1e-12)


def test_logdet(square_matrix_data):
    arr, m = square_matrix_data
    np_sign, np_logdet = np.linalg.slogdet(arr)

    m_logdet, m_sign = m.logdet

    np.testing.assert_allclose(m_logdet, np_logdet)
    assert m_sign == np_sign


def test_inverse(square_matrix_data):
    arr, m = square_matrix_data

    m_inv = m.inv
    np_inv = np.linalg.inv(arr)

    # Check if computed inverse matches numpy's inverse
    np.testing.assert_allclose(m_inv.data, np_inv, atol=1e-12)

    # Check if A * A^-1 = I
    identity = m @ m_inv
    np.testing.assert_allclose(identity.data, np.eye(m.rows), atol=1e-12)


def test_property_shape_errors():
    m = Matrix.ones(3, 4)

    with pytest.raises(ValueError, match="Only square matrices have determinants"):
        _ = m.det

    with pytest.raises(ValueError, match="Only square matrices are invertible"):
        _ = m.inv


# ---------------------------------------------------------
# 2. Triangular Solvers
# ---------------------------------------------------------
def test_triangular_solver_lower():
    L_arr = np.array([[2.0, 0.0], [3.0, 1.0]])
    B_arr = np.array([[4.0], [5.0]])

    L = Matrix(L_arr)
    B = Matrix(B_arr)

    Y = TriangularSolver.solve_lower(L, B)
    Y_np = np.linalg.solve(L_arr, B_arr)

    np.testing.assert_allclose(Y.data, Y_np)


def test_triangular_solver_upper():
    U_arr = np.array([[2.0, 3.0], [0.0, 1.0]])
    Y_arr = np.array([[8.0], [2.0]])

    U = Matrix(U_arr)
    Y = Matrix(Y_arr)

    X = TriangularSolver.solve_upper(U, Y)
    X_np = np.linalg.solve(U_arr, Y_arr)

    np.testing.assert_allclose(X.data, X_np)


# ---------------------------------------------------------
# 3. Gaussian Elimination
# ---------------------------------------------------------
def test_solve_gaussian(square_matrix_data):
    arr, mA = square_matrix_data

    # Test 1D Vector (b)
    b_arr = np.random.rand(4)
    mb = Matrix(b_arr)

    mX_1d = mA.solve(mb, method="gauss")
    X_1d_np = np.linalg.solve(arr, b_arr.reshape(-1, 1))
    np.testing.assert_allclose(mX_1d.data, X_1d_np, atol=1e-12)

    # Test 2D Matrix (B)
    B_arr = np.random.rand(4, 2)
    mB = Matrix(B_arr)

    mX_2d = mA.solve(mB, method="gauss")
    X_2d_np = np.linalg.solve(arr, B_arr)
    np.testing.assert_allclose(mX_2d.data, X_2d_np, atol=1e-12)


def test_solve_gaussian_errors(singular_matrix_data):
    _, m_sing = singular_matrix_data
    b = Matrix.ones(3, 1)

    with pytest.raises(ValueError, match="Matrix is singular or nearly singular"):
        m_sing.solve(b, method="gauss")


# ---------------------------------------------------------
# 4. LU Decomposition
# ---------------------------------------------------------
def test_lu_decomposition(square_matrix_data):
    arr, mA = square_matrix_data

    lu = LU(mA)
    P, L, U = lu.P.data, lu.L.data, lu.U.data

    # Check if P * A = L * U (Based on implementation logic)
    PA = P @ arr
    LU_res = L @ U
    np.testing.assert_allclose(PA, LU_res, atol=1e-12)

    # Check if L is lower triangular (all elements above diag are 0)
    assert np.allclose(L, np.tril(L))

    # Check if U is upper triangular (all elements below diag are 0)
    assert np.allclose(U, np.triu(U))


def test_solve_lu(square_matrix_data):
    arr, mA = square_matrix_data
    B_arr = np.random.rand(4, 3)
    mB = Matrix(B_arr)

    mX = mA.solve(mB, method="lu")
    X_np = np.linalg.solve(arr, B_arr)

    np.testing.assert_allclose(mX.data, X_np, atol=1e-12)


def test_lu_decomposition_errors(singular_matrix_data):
    _, m_sing = singular_matrix_data
    with pytest.raises(ValueError, match="Matrix is singular and cannot be decomposed"):
        LU(m_sing)


# ---------------------------------------------------------
# 5. Cholesky Decomposition
# ---------------------------------------------------------
def test_cholesky_decomposition(spd_matrix_data):
    arr, mA = spd_matrix_data

    chol = Cholesky(mA)
    L = chol.L.data

    # Check if L * L^T = A
    np.testing.assert_allclose(L @ L.T, arr, atol=1e-12)

    # Check if L is strictly lower triangular
    assert np.allclose(L, np.tril(L))


def test_solve_cholesky(spd_matrix_data):
    arr, mA = spd_matrix_data
    B_arr = np.random.rand(4, 2)
    mB = Matrix(B_arr)

    mX = mA.solve(mB, method="cholesky")
    X_np = np.linalg.solve(arr, B_arr)

    np.testing.assert_allclose(mX.data, X_np, atol=1e-12)


def test_cholesky_decomposition_errors(square_matrix_data):
    # Pass a non-symmetric matrix
    _, m_nonsym = square_matrix_data
    with pytest.raises(ValueError, match="Matrix must be symmetric"):
        Cholesky(m_nonsym)

    # Pass a symmetric but NOT positive-definite matrix
    arr_not_pd = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],  # Determinant is 1 - 4 = -3 (< 0)
        ]
    )
    m_not_pd = Matrix(arr_not_pd)
    with pytest.raises(ValueError, match="Matrix is not positive-definite"):
        Cholesky(m_not_pd)


# ---------------------------------------------------------
# 6. General Solve Router
# ---------------------------------------------------------
def test_solve_unknown_method(square_matrix_data):
    _, mA = square_matrix_data
    b = Matrix.ones(4, 1)

    with pytest.raises(ValueError, match="Unknown method: magic"):
        mA.solve(b, method="magic")


# =========================================================
# Performance Benchmarking (Pytest-Benchmark)
# Compare custom solvers vs Native Numpy (LAPACK)
# =========================================================

# Matrix size for benchmarking (Large enough to see complexity differences)
BENCH_SOLVE_SIZE = 128


@pytest.fixture
def solve_bench_data():
    """Generates a standard square system Ax = B."""
    np.random.seed(42)
    n = BENCH_SOLVE_SIZE
    A_arr = np.random.rand(n, n) + np.eye(n) * n  # Diagonally dominant for stability
    B_arr = np.random.rand(n, 1)

    return A_arr, B_arr, Matrix(A_arr), Matrix(B_arr)


@pytest.fixture
def spd_bench_data():
    """Generates a Symmetric Positive-Definite system for Cholesky."""
    np.random.seed(42)
    n = BENCH_SOLVE_SIZE
    # Create SPD matrix: (A^T * A) + n*I
    temp = np.random.rand(n, n)
    A_spd_arr = temp.T @ temp + np.eye(n) * n
    B_arr = np.random.rand(n, 1)

    return A_spd_arr, B_arr, Matrix(A_spd_arr), Matrix(B_arr)


def test_bench_solve_gauss_custom(benchmark, solve_bench_data):
    """Benchmark: Custom Gaussian Elimination."""
    _, _, mA, mB = solve_bench_data
    benchmark(lambda: mA.solve(mB, method="gauss"))


def test_bench_solve_lu_custom(benchmark, solve_bench_data):
    """Benchmark: Custom LU Decomposition solver."""
    _, _, mA, mB = solve_bench_data
    benchmark(lambda: mA.solve(mB, method="lu"))


def test_bench_solve_cholesky_custom(benchmark, spd_bench_data):
    """Benchmark: Custom Cholesky Decomposition solver."""
    _, _, mA, mB = spd_bench_data
    benchmark(lambda: mA.solve(mB, method="cholesky"))


def test_bench_solve_numpy_native(benchmark, solve_bench_data):
    """Benchmark: Native NumPy linalg.solve (Reference)."""
    A_arr, B_arr, _, _ = solve_bench_data
    # Note: NumPy uses highly optimized Fortran/C (LAPACK)
    benchmark(lambda: np.linalg.solve(A_arr, B_arr))


def test_bench_det_custom(benchmark, solve_bench_data):
    """Benchmark: Custom Determinant (via LU)."""
    _, _, mA, _ = solve_bench_data
    benchmark(lambda: mA.det)


def test_bench_det_numpy(benchmark, solve_bench_data):
    """Benchmark: Native NumPy determinant."""
    A_arr, _, _, _ = solve_bench_data
    benchmark(lambda: np.linalg.det(A_arr))
