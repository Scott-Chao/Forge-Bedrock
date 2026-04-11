import numpy as np
from .matrix import Matrix
from .solvers import TriangularSolver


class LU:
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


class Cholesky:
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


class QR:
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


class SVD:
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

        # Reduced SVD
        if not self.full_matrices:
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
            qr = QR(Matrix(aug))
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
