import numpy as np
from .matrix import Matrix


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
            from .decompositions import QR

            qr = QR(Matrix(curr_A))
            Q, R = qr.Q.data, qr.R.data

            next_A = R @ Q
            V @= Q

            if np.allclose(np.diag(next_A), np.diag(curr_A), atol=tol):
                break
            curr_A = next_A

        return np.diag(curr_A), Matrix(V)
