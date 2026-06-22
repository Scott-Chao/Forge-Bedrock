from __future__ import annotations

import numpy as np
from .matrix import Matrix
from .solvers import TriangularSolver


def get_adaptive_tol(A_data: np.ndarray) -> float:
    eps = np.finfo(A_data.dtype).eps
    norm_a = np.linalg.norm(A_data, ord=np.inf)
    return max(A_data.shape) * norm_a * eps


class LU:
    def __init__(self, matrix: Matrix, fail_on_singular: bool = True) -> None:
        if matrix.rows != matrix.cols:
            raise ValueError("LU Decomposition requires a square matrix.")

        self.matrix = matrix
        self.n = matrix.rows
        self.fail_on_singular = fail_on_singular
        self.P, self.L, self.U = self._decompose()

    def _decompose(self) -> tuple[Matrix, Matrix, Matrix]:
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

    def solve(self, B: Matrix | np.ndarray) -> Matrix:
        if not isinstance(B, Matrix):
            B = Matrix(B)

        PB = self.P @ B
        Y = TriangularSolver.solve_lower(self.L, PB)
        X = TriangularSolver.solve_upper(self.U, Y)
        return X


class Cholesky:
    def __init__(self, matrix: Matrix) -> None:
        if not matrix.is_symmetric:
            raise ValueError("Matrix must be symmetric for Cholesky Decomposition.")
        self.matrix = matrix
        self.n = matrix.rows
        self.L = self._decompose()

    def _decompose(self) -> Matrix:
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

    def solve(self, B: Matrix | np.ndarray) -> Matrix:
        if not isinstance(B, Matrix):
            B = Matrix(B)

        Y = TriangularSolver.solve_lower(self.L, B)
        X = TriangularSolver.solve_upper(self.L.T, Y)
        return X


class QR:
    def __init__(self, matrix: Matrix) -> None:
        self.matrix = matrix
        self.n, self.m = matrix.rows, matrix.cols
        self.data = matrix.data.copy().astype(np.float64)
        self.betas = np.zeros(min(self.m, self.n))
        self.tol = get_adaptive_tol(matrix.data)
        self._decompose()

    def _decompose(self) -> None:
        """
        QR Decomposition by Householder Transformations.
        v = x + sign(x[0]) * ||x|| * e_1
        H = I - 2 * (v @ v.T) / (v.T @ v)
        H is symmetric and orthogonal.
        """

        for k in range(min(self.n - 1, self.m)):
            x = self.data[k:, k]
            norm_x = np.linalg.norm(x)

            if norm_x <= self.tol:
                self.data[k:, k] = 0.0
                self.betas[k] = 0.0
                continue

            rho = -np.sign(x[0]) * norm_x if x[0] != 0 else -norm_x

            v_first = x[0] - rho
            v_remaining = x[1:] / v_first

            self.data[k, k] = rho
            self.data[k + 1 :, k] = v_remaining

            self.betas[k] = 2.0 / (1.0 + np.sum(v_remaining**2))

            # A = (I - beta * v @ v.T) @ A
            if k < self.m - 1:
                # v = [1; self.data[k+1:, k]]
                A_sub = self.data[k:, k + 1 :]
                # v.T @ A_sub = A_sub[0, :] + v_rem.T @ A_sub[1:, :]
                v_dot_A = A_sub[0, :] + self.data[k + 1 :, k] @ A_sub[1:, :]
                # A_sub = A_sub - beta * v @ (v.T @ A_sub)
                A_sub[0, :] -= self.betas[k] * v_dot_A
                A_sub[1:, :] -= self.betas[k] * np.outer(self.data[k + 1 :, k], v_dot_A)

    @property
    def R(self) -> Matrix:
        R = np.triu(self.data)
        return Matrix(R)

    @property
    def Q(self) -> Matrix:
        n = self.n
        Q = np.eye(n)

        for k in range(min(n - 1, self.m), -1, -1):
            if k >= len(self.betas) or self.betas[k] == 0:
                continue

            beta = self.betas[k]
            v_remaining = self.data[k + 1 :, k]

            # Q[k:, k:] = (I - beta * v @ v.T) @ Q[k:, k:]
            sub_Q = Q[k:, k:]
            v_dot_Q = sub_Q[0, :] + v_remaining @ sub_Q[1:, :]
            sub_Q[0, :] -= beta * v_dot_Q
            sub_Q[1:, :] -= beta * np.outer(v_remaining, v_dot_Q)

        return Matrix(Q)


class Hessenberg:
    def __init__(self, matrix: Matrix) -> None:
        self.matrix = matrix
        self.n = matrix.rows
        self.data = matrix.data.copy().astype(np.float64)
        self.betas = np.zeros(self.n - 2)
        self.tol = get_adaptive_tol(self.matrix.data)
        if matrix.rows != matrix.cols:
            raise ValueError("Hessenberg requires a square matrix.")
        self._decompose()

    def _decompose(self) -> None:
        """
        Reduce A to Upper Hessenberg form H using Householder reflections.
        H = Q.T @ A @ Q
        """
        n = self.n

        for k in range(n - 2):
            x = self.data[k + 1 :, k]
            norm_x = np.linalg.norm(x)

            if norm_x <= self.tol:
                self.data[k + 1 :, k] = 0.0
                self.betas[k] = 0.0
                continue

            rho = -np.sign(x[0]) * norm_x if x[0] != 0 else -norm_x

            v_first = x[0] - rho
            v_remaining = x[1:] / v_first

            self.data[k + 1, k] = rho
            self.data[k + 2 :, k] = v_remaining
            self.betas[k] = 2.0 / (1.0 + np.sum(v_remaining**2))

            beta = self.betas[k]

            # Left multiply: Apply reflection to rows
            sub_H_left = self.data[k + 1 :, k + 1 :]
            v_dot_H_left = sub_H_left[0, :] + v_remaining @ sub_H_left[1:, :]
            sub_H_left[0, :] -= beta * v_dot_H_left
            sub_H_left[1:, :] -= beta * np.outer(v_remaining, v_dot_H_left)

            # Right multiply: Apply reflection to columns
            sub_H_right = self.data[:, k + 1 :]
            v_dot_H_right = sub_H_right[:, 0] + sub_H_right[:, 1:] @ v_remaining
            sub_H_right[:, 0] -= beta * v_dot_H_right
            sub_H_right[:, 1:] -= beta * np.outer(v_dot_H_right, v_remaining)

    @property
    def H(self) -> Matrix:
        H = np.triu(self.data, -1)
        return Matrix(H)

    @property
    def Q(self) -> Matrix:
        n = self.n
        Q = np.eye(n)

        for k in range(n - 3, -1, -1):
            v_remaining = self.data[k + 2 :, k]
            beta = self.betas[k]

            sub_Q = Q[k + 1 :, k + 1 :]
            v_dot_Q = sub_Q[0, :] + v_remaining @ sub_Q[1:, :]
            sub_Q[0, :] -= beta * v_dot_Q
            sub_Q[1:, :] -= beta * np.outer(v_remaining, v_dot_Q)

        return Matrix(Q)


class Schur:
    def __init__(
        self, matrix: Matrix, max_iter: int = 1000, use_shifts: bool = True
    ) -> None:
        self.matrix = matrix
        self.max_iter = max_iter
        self.use_shifts = use_shifts
        self.tol = get_adaptive_tol(matrix.data)
        self.T, self.Q = self._decompose()

    def _decompose(self) -> tuple[Matrix, Matrix]:
        hess = Hessenberg(self.matrix)
        T = hess.H.data.copy()
        Q = hess.Q.data.copy()
        n = T.shape[0]

        curr_n = n
        iter_count = 0

        while curr_n > 1 and iter_count < self.max_iter:
            # Check Deflation
            if np.abs(T[curr_n - 1, curr_n - 2]) < self.tol:
                T[curr_n - 1, curr_n - 2] = 0.0
                curr_n -= 1
                continue

            # Calculate Wilkinson Shift
            mu = 0.0
            if self.use_shifts:
                mu = self._get_wilkinson_shift(
                    T[curr_n - 2, curr_n - 2],
                    T[curr_n - 2, curr_n - 1],
                    T[curr_n - 1, curr_n - 2],
                    T[curr_n - 1, curr_n - 1],
                )

                diag_idx = np.diag_indices(curr_n)
                T[diag_idx] -= mu

            # QR Step using Givens
            self._qr_step_givens(T[:curr_n, :curr_n], Q[:, :curr_n])

            # Add Shift Back
            if self.use_shifts:
                T[diag_idx] += mu

            iter_count += 1

        return Matrix(T), Matrix(Q)

    @staticmethod
    def generate_givens(a: float, b: float) -> tuple[float, float]:
        """Compute cosine and sine for Givens rotations."""
        if b == 0:
            return 1.0, 0.0
        r = np.hypot(a, b)
        return a / r, b / r

    def _qr_step_givens(
        self, H: np.ndarray, Q_total: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform one QR step on a Hessenberg matrix using Givens rotations."""
        n = H.shape[0]
        rotations = []

        # Left multiplications
        for i in range(n - 1):
            c, s = self.generate_givens(H[i, i], H[i + 1, i])
            rotations.append((c, s))

            rot = np.array([[c, s], [-s, c]])
            H[i : i + 2, i:] = rot @ H[i : i + 2, i:]
            H[i + 1, i] = 0.0

        # Right multiplications
        for i in range(n - 1):
            c, s = rotations[i]
            rot = np.array([[c, -s], [s, c]])
            H[: i + 2, i : i + 2] = H[: i + 2, i : i + 2] @ rot
            Q_total[:, i : i + 2] = Q_total[:, i : i + 2] @ rot

        return H, Q_total

    def _get_wilkinson_shift(
        self, a11: float, a12: float, a21: float, a22: float
    ) -> float:
        """"""
        d = (a11 - a22) / 2.0
        disc = d**2 + a12 * a21
        if disc < 0:
            return a22
        denom = np.abs(d) + np.sqrt(disc)
        if denom == 0:
            return a22
        mu = a22 - (np.sign(d) if d != 0 else 1.0) * (a12 * a21) / denom
        return mu


class SVD:
    def __init__(
        self,
        matrix: Matrix,
        method: str = "jacobi",
        full_matrices: bool = True,
        **kwargs: float,
    ) -> None:
        self.matrix = matrix
        self.tol = get_adaptive_tol(matrix.data)
        self.full_matrices = full_matrices
        self.U, self.S, self.VT = self._decompose(method, **kwargs)

    def _decompose(self, method: str, **kwargs: float) -> tuple[Matrix, Matrix, Matrix]:
        A_raw = self.matrix.data
        m, n = A_raw.shape
        is_wide = m < n
        A_tall = A_raw.T if is_wide else A_raw
        if is_wide:
            m, n = n, m

        if method == "qr":
            U, s, VT = self._decompose_tall_qr(A_tall)
        elif method == "jacobi":
            max_sweeps = kwargs.get("max_sweeps", 100)
            U, s, VT = self._decompose_tall_jacobi(A_tall, max_sweeps)
        else:
            raise ValueError(f"Unknown method: {method}")

        norms = np.linalg.norm(U, axis=0)
        target_cols = m if self.full_matrices else n
        if np.any(norms < 0.5) or U.shape[1] < target_cols:
            U = self._orthogonal_completion(U, target_cols)

        Sigma = np.zeros((m, n))
        for i in range(len(s)):
            Sigma[i, i] = s[i]

        if is_wide:
            U, Sigma, VT = VT.T, Sigma.T, U.T

        # Reduced SVD
        if not self.full_matrices:
            k = min(m, n)
            U = U[:, :k]
            Sigma = Sigma[:k, :k]
            VT = VT[:k, :]

        return Matrix(U), Matrix(Sigma), Matrix(VT)

    def _decompose_tall_qr(
        self, A: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, n = A.shape
        ATA = A.T @ A
        eigenvalues, V_mat = Matrix(ATA).eig()
        V = V_mat.data

        singular_values = np.sqrt(np.maximum(eigenvalues, 0))
        idx = np.argsort(singular_values)[::-1]
        singular_values = singular_values[idx]
        V = V[:, idx]

        U = np.zeros((m, n))
        for i in range(n):
            if singular_values[i] > self.tol:
                U[:, i] = A @ V[:, i] / singular_values[i]
            else:
                U[:, i] = 0.0

        return U, singular_values, V.T

    def _decompose_tall_jacobi(
        self, A: np.ndarray, max_sweeps: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stable SVD via One-Sided Jacobi Rotations"""
        A = A.copy()
        m, n = A.shape
        V = np.eye(n)

        for _ in range(max_sweeps):
            converged = True
            for i in range(n):
                for j in range(i + 1, n):
                    w = A[:, i] @ A[:, j]
                    if abs(w) <= self.tol:
                        continue

                    converged = False
                    vx = A[:, i] @ A[:, i]
                    vy = A[:, j] @ A[:, j]

                    xi = (vy - vx) / (2 * w)
                    sign_xi = 1.0 if xi >= 0 else -1.0
                    t = sign_xi / (abs(xi) + np.sqrt(xi**2 + 1))
                    c = 1 / np.sqrt(1 + t**2)
                    s = c * t

                    rot = np.array([[c, s], [-s, c]])
                    A[:, [i, j]] = A[:, [i, j]] @ rot
                    V[:, [i, j]] = V[:, [i, j]] @ rot

            if converged:
                break

        singular_values = np.linalg.norm(A, axis=0)
        idx = np.argsort(singular_values)[::-1]
        singular_values = singular_values[idx]
        A = A[:, idx]
        V = V[:, idx]

        U = np.zeros((m, n))
        for i in range(n):
            if singular_values[i] > self.tol:
                U[:, i] = A[:, i] / singular_values[i]
            else:
                U[:, i] = 0.0

        return U, singular_values, V.T

    def _orthogonal_completion(
        self, U_partial: np.ndarray, target_cols: int
    ) -> np.ndarray:
        m = U_partial.shape[0]
        k = U_partial.shape[1]
        if k >= target_cols:
            return U_partial[:, :target_cols]

        I = np.eye(m)
        aug = np.hstack([U_partial, I])
        qr = QR(Matrix(aug))
        U = qr.Q.data

        for i in range(k):
            if np.dot(U[:, i], U_partial[:, i]) < 0:
                U[:, i] *= -1

        return U
