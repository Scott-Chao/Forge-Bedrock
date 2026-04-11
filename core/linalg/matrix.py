import numpy as np


class Matrix:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64)

    def __repr__(self):
        return f"Matrix({self.data})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def rows(self):
        return self.data.shape[0]

    @property
    def cols(self):
        return self.data.shape[1]

    @property
    def T(self):
        return Matrix(self.data.T)

    @property
    def is_symmetric(self):
        return np.allclose(self.data, self.data.T)

    @property
    def det(self):
        """det(A) = det(P^-1) * det(L) * det(U)"""
        if self.rows != self.cols:
            raise ValueError("Only square matrices have determinants.")

        lu_obj = LUDecomposition(self, fail_on_singular=False)

        P_det = np.linalg.det(lu_obj.P.data)
        U_diag = np.diag(lu_obj.U.data)
        U_det = np.prod(U_diag)

        return P_det * U_det

    @property
    def logdet(self):
        """ln |det(A)| = ln |det(U)|"""
        if self.rows != self.cols:
            raise ValueError("Only square matrices have determinants.")

        lu_obj = LUDecomposition(self, fail_on_singular=False)
        P_det = np.linalg.det(lu_obj.P.data)
        U_diag = np.diag(lu_obj.U.data)

        signs = np.sign(U_diag)
        abs_diag = np.abs(U_diag)

        if np.any(abs_diag == 0):
            return -np.inf, 0

        final_sign = P_det * np.prod(signs)
        log_abs_det = np.sum(np.log(abs_diag))

        return log_abs_det, final_sign

    @property
    def inv(self):
        if self.rows != self.cols:
            raise ValueError("Only square matrices are invertible.")

        I = Matrix.eye(self.rows)
        return self.solve(I, method="lu")

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    @staticmethod
    def zeros(rows, cols):
        return Matrix(np.zeros((rows, cols)))

    @staticmethod
    def ones(rows, cols):
        return Matrix(np.ones((rows, cols)))

    @staticmethod
    def eye(n):
        mat = Matrix.zeros(n, n)
        mat.data.flat[:: n + 1] = 1.0
        return mat

    @staticmethod
    def rand(rows, cols, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return Matrix(np.random.rand(rows, cols))

    def _apply_op(self, other, op):
        if isinstance(other, (int, float, np.float64, np.int64)):
            return Matrix(op(self.data, other))
        if not isinstance(other, Matrix):
            raise TypeError(f"Unsupported operand type: {type(other)}")
        return Matrix(BroadcastEngine.execute(self, other, op))

    def _apply_inplace_op(self, other, op):
        if isinstance(other, (int, float, np.float64, np.int64)):
            op(self.data, other, out=self.data)
            return self
        if not isinstance(other, Matrix):
            raise TypeError(f"Unsupported operand type: {type(other)}")
        BroadcastEngine.execute(self, other, op, out_data=self.data)
        return self

    def __add__(self, other):
        return self._apply_op(other, np.add)

    def __iadd__(self, other):
        return self._apply_inplace_op(other, np.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._apply_op(other, np.subtract)

    def __isub__(self, other):
        return self._apply_inplace_op(other, np.subtract)

    def __rsub__(self, other):
        return (self - other) * -1

    def __mul__(self, other):
        return self._apply_op(other, np.multiply)

    def __imul__(self, other):
        return self._apply_inplace_op(other, np.multiply)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._apply_op(other, np.divide)

    def __itruediv__(self, other):
        return self._apply_inplace_op(other, np.divide)

    def __rtruediv__(self, other):
        return Matrix(other) / self

    def __pow__(self, other):
        return self._apply_op(other, np.power)

    def __rpow__(self, other):
        return Matrix(other) ** self

    def __neg__(self):
        return Matrix(-self.data)

    def __pos__(self):
        return Matrix(np.copy(self.data))

    def __abs__(self):
        return Matrix(np.abs(self.data))

    def __matmul__(self, other):
        return self.matmul(other, S=64)

    def matmul(self, other, S=64):
        assert self.cols == other.rows, (
            "Matrix dimensions not compatible for multiplication"
        )
        M, K, N = self.rows, self.cols, other.cols
        res = Matrix.zeros(M, N)

        for i1 in range(0, M, S):
            for k1 in range(0, K, S):
                for j1 in range(0, N, S):
                    i2 = min(i1 + S, M)
                    k2 = min(k1 + S, K)
                    j2 = min(j1 + S, N)

                    A = self.data[i1:i2, k1:k2]
                    B = other.data[k1:k2, j1:j2]
                    C = res.data[i1:i2, j1:j2]

                    C += A @ B

        return res

    def solve(self, B, method="lu"):
        if method == "gauss":
            return self._solve_gaussian(B)
        elif method == "lu":
            return LUDecomposition(self).solve(B)
        elif method == "cholesky":
            return CholeskyDecomposition(self).solve(B)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _solve_gaussian(self, B):
        if not isinstance(B, Matrix):
            B = Matrix(B)

        assert self.rows == self.cols, "Matrix must be square"
        assert self.rows == B.rows, "Dimension mismatch between A and B"

        n = self.rows

        if B.data.ndim == 1:
            B.data = B.data.reshape(-1, 1)
        aug = np.hstack([self.data.copy(), B.data.copy()])

        for k in range(n):
            pivot_idx = np.argmax(np.abs(aug[k:, k])) + k

            if np.abs(aug[pivot_idx, k]) < 1e-15:
                raise ValueError("Matrix is singular or nearly singular")

            if pivot_idx != k:
                aug[[k, pivot_idx]] = aug[[pivot_idx, k]]

            factors = aug[k + 1 :, k] / aug[k, k]
            aug[k + 1 :, k:] -= factors[:, np.newaxis] * aug[k, k:]

        U = Matrix(aug[:, :n])
        B_prime = Matrix(aug[:, n:])
        x = TriangularSolver.solve_upper(U, B_prime)

        return x

    def eig(self, method="qr"):
        solver = EigenSolver(self)
        if method == "qr":
            return solver.find_all_eigen()
        elif method == "power":
            return solver.power_iteration()
        else:
            raise ValueError(f"Unknown eigenvalue method: {method}")


class BroadcastEngine:
    @staticmethod
    def _get_strides(shape):
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    @staticmethod
    def _get_config(matrix1, matrix2):
        s1, s2 = matrix1.shape, matrix2.shape
        ndim = max(len(s1), len(s2))

        s1_p = (1,) * (ndim - len(s1)) + s1
        s2_p = (1,) * (ndim - len(s2)) + s2

        st1_raw = BroadcastEngine._get_strides(s1_p)
        st2_raw = BroadcastEngine._get_strides(s2_p)

        target_shape, strides1, strides2 = [], [], []

        for d1, d2, rs1, rs2 in zip(s1_p, s2_p, st1_raw, st2_raw):
            if d1 != d2 and d1 != 1 and d2 != 1:
                raise ValueError(f"Incompatible shapes: {s1} and {s2}")

            target_shape.append(max(d1, d2))
            strides1.append(rs1 if d1 != 1 else 0)
            strides2.append(rs2 if d2 != 1 else 0)

        return tuple(target_shape), tuple(strides1), tuple(strides2)

    @staticmethod
    def execute(matrix1, matrix2, op, out_data=None):
        target_shape, st1, st2 = BroadcastEngine._get_config(matrix1, matrix2)
        if out_data is None:
            out_data = np.empty(target_shape, dtype=matrix1.data.dtype)
        elif target_shape != matrix1.shape:
            raise ValueError("Cannot broadcast to inplace output shape")

        st_res = BroadcastEngine._get_strides(target_shape)

        f1, f2, fr = matrix1.data.ravel(), matrix2.data.ravel(), out_data.ravel()
        ndim = len(target_shape)

        def _worker(dim, off1, off2, off_r):
            if dim == ndim - 1:
                size = target_shape[dim]
                idx1 = np.arange(size) * st1[dim] + off1
                idx2 = np.arange(size) * st2[dim] + off2
                idx_r = np.arange(size) * st_res[dim] + off_r
                fr[idx_r] = op(f1[idx1], f2[idx2])
                return

            for i in range(target_shape[dim]):
                _worker(
                    dim + 1,
                    off1 + i * st1[dim],
                    off2 + i * st2[dim],
                    off_r + i * st_res[dim],
                )

        _worker(0, 0, 0, 0)
        return out_data


class TriangularSolver:
    @staticmethod
    def solve_lower(L, B):
        """Solve LY = B (L is lower triangular)"""
        n = L.rows

        L_data, B_data = L.data, B.data
        if B_data.ndim == 1:
            B_data = B_data.reshape(-1, 1)

        k = B.cols
        Y = np.zeros((n, k))

        for i in range(n):
            sum_ly = L_data[i, :i] @ Y[:i, :]
            Y[i, :] = (B_data[i, :] - sum_ly) / L_data[i, i]
        return Matrix(Y)

    @staticmethod
    def solve_upper(U, Y):
        """Solve UX = Y (U is upper triangular)"""
        n = U.rows

        U_data, Y_data = U.data, Y.data
        if Y_data.ndim == 1:
            Y_data = Y_data.reshape(-1, 1)

        k = Y.cols
        X = np.zeros((n, k))

        for i in range(n - 1, -1, -1):
            sum_ux = U_data[i, i + 1 :] @ X[i + 1 :, :]
            X[i, :] = (Y_data[i, :] - sum_ux) / U_data[i, i]
        return Matrix(X)


class LUDecomposition:
    def __init__(self, matrix, fail_on_singular=True):
        if matrix.rows != matrix.cols:
            raise ValueError("LU Decomposition requires a square matrix.")

        self.matrix = matrix
        self.n = matrix.rows
        self.fail_on_singular = fail_on_singular
        self.P, self.L, self.U = self._decompose()

    def _decompose(self):
        n = self.n
        P = np.eye(n)
        L = np.eye(n)
        U = self.matrix.data.copy().astype(np.float64)

        for k in range(n):
            pivot_idx = np.argmax(np.abs(U[k:, k])) + k

            if np.abs(U[pivot_idx, k]) < 1e-15:
                if self.fail_on_singular:
                    raise ValueError("Matrix is singular and cannot be decomposed.")
                else:
                    return Matrix(P), Matrix(L), Matrix(U)

            U[[k, pivot_idx]] = U[[pivot_idx, k]]
            P[[k, pivot_idx]] = P[[pivot_idx, k]]
            if k > 0:
                L[[k, pivot_idx], :k] = L[[pivot_idx, k], :k]

            factors = U[k + 1 :, k] / U[k, k]
            L[k + 1 :, k] = factors
            U[k + 1 :, k:] -= factors[:, np.newaxis] * U[k, k:]

        return Matrix(P), Matrix(L), Matrix(U)

    def solve(self, B):
        if not isinstance(B, Matrix):
            B = Matrix(B)

        PB = self.P @ B
        Y = TriangularSolver.solve_lower(self.L, PB)
        X = TriangularSolver.solve_upper(self.U, Y)
        return X


class CholeskyDecomposition:
    def __init__(self, matrix):
        if not matrix.is_symmetric:
            raise ValueError("Matrix must be symmetric for Cholesky Decomposition.")
        self.matrix = matrix
        self.n = matrix.rows
        self.L = self._decompose()

    def _decompose(self):
        n = self.n
        A = self.matrix.data
        L = np.zeros((n, n))

        for j in range(n):
            sum_sq = np.sum(L[j, :j] ** 2)
            val = A[j, j] - sum_sq

            if val <= 0:
                raise ValueError("Matrix is not positive-definite.")

            L[j, j] = np.sqrt(val)

            if j < n - 1:
                remaining_A = A[j + 1 :, j]
                sums = L[j + 1 :, :j] @ L[j, :j]
                L[j + 1 :, j] = (remaining_A - sums) / L[j, j]

        return Matrix(L)

    def solve(self, B):
        if not isinstance(B, Matrix):
            B = Matrix(B)

        Y = TriangularSolver.solve_lower(self.L, B)
        X = TriangularSolver.solve_upper(self.L.T, Y)
        return X


class EigenSolver:
    def __init__(self, matrix):
        self.matrix = matrix
        self.n = matrix.rows
        if matrix.rows != matrix.cols:
            raise ValueError("EigenSolver requires a square matrix.")

    def power_iteration(self, max_iter=1000, tol=1e-10):
        """
        Power Iteration: find the dominant eigenvalue and corresponding eigenvector.
        :return: (eigenvalue, eigenvector)
        """
        b_k = np.random.rand(self.n, 1)
        b_k /= np.linalg.norm(b_k)

        last_eigenvalue = 0

        for _ in range(max_iter):
            b_k = self.matrix.data @ b_k
            b_k /= np.linalg.norm(b_k)

            current_eigenvalue = (b_k.T @ self.matrix.data @ b_k)[0, 0]

            if np.abs(current_eigenvalue - last_eigenvalue) < tol:
                break
            last_eigenvalue = current_eigenvalue

        return last_eigenvalue, Matrix(b_k)

    def find_all_eigen(self, max_iter=1000, tol=1e-10):
        """
        Find all eigenvalues and eigenvectors using QR Algorithm.
        next_A = R @ Q = Q.T @ A @ Q
        """
        curr_A = self.matrix.data.copy()
        n = self.n
        V = np.eye(n)

        for i in range(max_iter):
            qr = QRDecomposition(Matrix(curr_A))
            Q, R = qr.Q.data, qr.R.data

            next_A = R @ Q
            V @= Q

            if np.allclose(np.diag(next_A), np.diag(curr_A), atol=tol):
                break
            curr_A = next_A

        return np.diag(curr_A), Matrix(V)


class QRDecomposition:
    def __init__(self, matrix):
        self.matrix = matrix
        self.n, self.m = matrix.rows, matrix.cols
        self.Q, self.R = self._decompose()

    def _decompose(self):
        """
        QR Decomposition by Householder Transformations.
        v = x + sign(x[0]) * ||x|| * e_1
        H = I - 2 * (v @ v.T) / (v.T @ v)
        H is symmetric and orthogonal.
        """

        Q = np.eye(self.n)
        R = self.matrix.data.copy().astype(np.float64)

        for k in range(self.m):
            if k >= self.n - 1:
                break

            x = R[k:, k : k + 1]
            norm_x = np.linalg.norm(x)

            v = x.copy()
            v[0] += np.sign(x[0, 0]) * norm_x if x[0, 0] != 0 else norm_x
            v /= np.linalg.norm(v)

            # R = H @ R = (I - 2 v @ v.T) @ R = R - 2 v @ v.T @ R
            R[k:, k:] -= 2 * v @ (v.T @ R[k:, k:])
            # Q = Q @ H = Q @ (I - 2 v @ v.T) = Q - 2 Q @ v @ v.T
            Q[:, k:] -= 2 * (Q[:, k::] @ v) @ v.T

        return Matrix(Q), Matrix(R)


class SVDDecomposition:
    def __init__(self, matrix, tol=1e-10, full_matrices=True):
        self.matrix = matrix
        self.tol = tol
        self.full_matrices = full_matrices
        self.U, self.S, self.VT = self._decompose()

    def _decompose(self):
        A = self.matrix.data
        m, n = A.shape

        if m >= n:
            U, Sigma, VT = self._decompose_tall(A, m, n)
        else:
            V, Sigma_tall, UT = self._decompose_tall(A.T, n, m)
            U, Sigma, VT = UT.T, Sigma_tall.T, V.T

        if not self.full_matrices:
            # Return Reduced SVD
            k = min(m, n)
            U = U[:, :k]
            Sigma = Sigma[:k, :k]
            VT = VT[:k, :]

        return Matrix(U), Matrix(Sigma), Matrix(VT)

    def _decompose_tall(self, A, m, n):
        ATA = A.T @ A
        eigenvalues, V_mat = Matrix(ATA).eig()
        V = V_mat.data

        singular_values = np.sqrt(np.maximum(eigenvalues, 0))
        idx = np.argsort(singular_values)[::-1]
        singular_values = singular_values[idx]
        V = V[:, idx]

        valid = singular_values > self.tol
        k = np.sum(valid)
        singular_values_valid = singular_values[valid]

        U_partial = np.zeros((m, k))
        for i in range(k):
            U_partial[:, i] = A @ V[:, i] / singular_values_valid[i]

        if self.full_matrices and k < m:
            I = np.eye(m)
            aug = np.hstack([U_partial, I])
            qr = QRDecomposition(Matrix(aug))
            U = qr.Q.data

            for i in range(k):
                if np.dot(U[:, i], U_partial[:, i]) < 0:
                    U[:, i] *= -1
        else:
            U = U_partial
            if not self.full_matrices:
                if k < n:
                    U = np.hstack([U, np.zeros((m, n - k))])

        Sigma = np.zeros((m, n))
        for i in range(min(k, m, n)):
            Sigma[i, i] = singular_values_valid[i]

        return U, Sigma, V.T


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.singular_values = None
        self.explained_variance_ratio = None

    def fit(self, X):
        n_samples = X.rows
        self.mean = np.mean(X.data, axis=0)
        X_centered = X.data - self.mean

        svd = SVDDecomposition(Matrix(X_centered))

        self.components = Matrix(svd.VT.data[: self.n_components, :])
        full_singular_values = svd.S.data.diagonal()
        self.singular_values = full_singular_values[: self.n_components]

        explained_variance = (self.singular_values**2) / (n_samples - 1)
        total_variance = np.sum(explained_variance)
        explained_variance = explained_variance[: self.n_components]
        self.explained_variance_ratio = explained_variance / total_variance

        return self

    def transform(self, X):
        X_centered = X.data - self.mean
        return Matrix(X_centered @ self.components.data.T)

    def inverse_transform(self, X_reduced):
        return Matrix(X_reduced.data @ self.components.data + self.mean)
