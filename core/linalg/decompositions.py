import numpy as np
from .matrix import Matrix
from .solvers import TriangularSolver


def get_adaptive_tol(A_data):
    eps = np.finfo(A_data.dtype).eps
    norm_a = np.linalg.norm(A_data, ord=np.inf)
    return max(A_data.shape) * norm_a * eps


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
        self.tol = get_adaptive_tol(matrix.data)
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

            if norm_x <= self.tol:
                R[k:, k] = 0.0
                continue

            v = x.copy()
            v[0] += np.sign(x[0, 0]) * norm_x if x[0, 0] != 0 else norm_x
            v /= np.linalg.norm(v)

            # R = H @ R = (I - 2 v @ v.T) @ R = R - 2 v @ v.T @ R
            R[k:, k:] -= 2 * v @ (v.T @ R[k:, k:])
            # Q = Q @ H = Q @ (I - 2 v @ v.T) = Q - 2 Q @ v @ v.T
            Q[:, k:] -= 2 * (Q[:, k::] @ v) @ v.T

        return Matrix(Q), Matrix(R)


class Hessenberg:
    def __init__(self, matrix):
        self.matrix = matrix
        self.n = matrix.rows
        self.tol = get_adaptive_tol(self.matrix.data)
        if matrix.rows != matrix.cols:
            raise ValueError("Hessenberg requires a square matrix.")
        self.H, self.Q = self._decompose()

    def _decompose(self):
        """
        Reduce A to Upper Hessenberg form H using Householder reflections.
        H = Q.T @ A @ Q
        """
        n = self.n
        H = self.matrix.data.copy().astype(np.float64)
        Q = np.eye(n)

        for k in range(n - 2):
            x = H[k + 1 :, k : k + 1]
            norm_x = np.linalg.norm(x)

            if norm_x <= self.tol:
                H[k + 1 :, k] = 0.0
                continue

            v = x.copy()
            v[0, 0] += np.sign(x[0, 0]) * norm_x if x[0, 0] != 0 else norm_x
            v /= np.linalg.norm(v)

            # Left multiply: Apply reflection to rows
            H[k + 1 :, k:] -= 2 * v @ (v.T @ H[k + 1 :, k:])
            # Right multiply: Apply reflection to columns
            H[:, k + 1 :] -= 2 * (H[:, k + 1 :] @ v) @ v.T
            # Accumulate the transformation matrix Q
            Q[:, k + 1 :] -= 2 * (Q[:, k + 1 :] @ v) @ v.T

        return Matrix(H), Matrix(Q)


class Schur:
    def __init__(self, matrix, max_iter=1000, use_shifts=True):
        self.matrix = matrix
        self.max_iter = max_iter
        self.use_shifts = use_shifts
        self.tol = get_adaptive_tol(matrix.data)
        self.T, self.Q = self._decompose()

    def _decompose(self):
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
    def generate_givens(a, b):
        """Compute cosine and sine for Givens rotations."""
        if b == 0:
            return 1.0, 0.0
        r = np.hypot(a, b)
        return a / r, b / r

    def _qr_step_givens(self, H, Q_total):
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

    def _get_wilkinson_shift(self, a11, a12, a21, a22):
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
    def __init__(self, matrix, method="jacobi", full_matrices=True, **kwargs):
        self.matrix = matrix
        self.tol = get_adaptive_tol(matrix.data)
        self.full_matrices = full_matrices
        self.U, self.S, self.VT = self._decompose(method, **kwargs)

    def _decompose(self, method, **kwargs):
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

    def _decompose_tall_qr(self, A):
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

    def _decompose_tall_jacobi(self, A, max_sweeps):
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

    def _orthogonal_completion(self, U_partial, target_cols):
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
