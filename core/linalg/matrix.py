from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .broadcast import BroadcastEngine


class Matrix:
    def __init__(self, data: np.ndarray | list | tuple):
        self.data = np.array(data, dtype=np.float64)

    def __repr__(self) -> str:
        return f"Matrix({self.data})"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def rows(self) -> int:
        return self.data.shape[0]

    @property
    def cols(self) -> int:
        return self.data.shape[1]

    @property
    def T(self) -> Matrix:
        return Matrix(self.data.T)

    @property
    def is_symmetric(self) -> bool:
        return np.allclose(self.data, self.data.T)

    @property
    def det(self) -> float:
        """det(A) = det(P^-1) * det(L) * det(U)"""
        from .decompositions import LU

        if self.rows != self.cols:
            raise ValueError("Only square matrices have determinants.")

        lu_obj = LU(self, fail_on_singular=False)

        P_det = np.linalg.det(lu_obj.P.data)
        U_diag = np.diag(lu_obj.U.data)
        U_det = np.prod(U_diag)

        return P_det * U_det

    @property
    def logdet(self) -> tuple[float, int]:
        """ln |det(A)| = ln |det(U)|"""
        from .decompositions import LU

        if self.rows != self.cols:
            raise ValueError("Only square matrices have determinants.")

        lu_obj = LU(self, fail_on_singular=False)
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
    def inv(self) -> Matrix:
        if self.rows != self.cols:
            raise ValueError("Only square matrices are invertible.")

        I = Matrix.eye(self.rows)
        return self.solve(I, method="lu")

    @property
    def pinv(self) -> Matrix:
        """
        Moore-Penrose pseudo-inverse
        A+ = (A.T @ A)^-1 @ A.T
        A+ = V @ Sigma+ @ U.T
        """
        from .decompositions import SVD

        svd = SVD(self)
        U, S, VT = svd.U, svd.S, svd.VT

        m, n = self.rows, self.cols
        sigma_plus = np.zeros((n, m))

        singular_values = np.diag(S.data)

        tol = max(m, n) * np.spacing(np.max(singular_values))

        for i, s in enumerate(singular_values):
            if s > tol:
                sigma_plus[i, i] = 1.0 / s

        return VT.T @ Matrix(sigma_plus) @ U.T

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.data[key] = value

    @staticmethod
    def zeros(rows: int, cols: int) -> Matrix:
        return Matrix(np.zeros((rows, cols)))

    @staticmethod
    def ones(rows: int, cols: int) -> Matrix:
        return Matrix(np.ones((rows, cols)))

    @staticmethod
    def eye(n: int) -> Matrix:
        mat = Matrix.zeros(n, n)
        mat.data.flat[:: n + 1] = 1.0
        return mat

    @staticmethod
    def rand(rows: int, cols: int, seed: int | None = None) -> Matrix:
        if seed is not None:
            np.random.seed(seed)
        return Matrix(np.random.rand(rows, cols))

    def _apply_op(
        self, other: Matrix | int | float, op: Callable[..., np.ndarray]
    ) -> Matrix:
        if isinstance(other, (int, float, np.float64, np.int64)):
            return Matrix(op(self.data, other))
        if not isinstance(other, Matrix):
            raise TypeError(f"Unsupported operand type: {type(other)}")
        return Matrix(BroadcastEngine.execute(self, other, op))

    def _apply_inplace_op(
        self, other: Matrix | int | float, op: Callable[..., np.ndarray]
    ) -> Matrix:
        if isinstance(other, (int, float, np.float64, np.int64)):
            op(self.data, other, out=self.data)
            return self
        if not isinstance(other, Matrix):
            raise TypeError(f"Unsupported operand type: {type(other)}")
        BroadcastEngine.execute(self, other, op, out_data=self.data)
        return self

    def __add__(self, other: Matrix | int | float) -> Matrix:
        return self._apply_op(other, np.add)

    def __iadd__(self, other: Matrix | int | float) -> Matrix:
        return self._apply_inplace_op(other, np.add)

    def __radd__(self, other: Matrix | int | float) -> Matrix:
        return self.__add__(other)

    def __sub__(self, other: Matrix | int | float) -> Matrix:
        return self._apply_op(other, np.subtract)

    def __isub__(self, other: Matrix | int | float) -> Matrix:
        return self._apply_inplace_op(other, np.subtract)

    def __rsub__(self, other: Matrix | int | float) -> Matrix:
        return (self - other) * -1

    def __mul__(self, other: Matrix | int | float) -> Matrix:
        return self._apply_op(other, np.multiply)

    def __imul__(self, other: Matrix | int | float) -> Matrix:
        return self._apply_inplace_op(other, np.multiply)

    def __rmul__(self, other: Matrix | int | float) -> Matrix:
        return self.__mul__(other)

    def __truediv__(self, other: Matrix | int | float) -> Matrix:
        return self._apply_op(other, np.divide)

    def __itruediv__(self, other: Matrix | int | float) -> Matrix:
        return self._apply_inplace_op(other, np.divide)

    def __rtruediv__(self, other: Matrix | int | float) -> Matrix:
        return Matrix(other) / self

    def __pow__(self, other: Matrix | int | float) -> Matrix:
        return self._apply_op(other, np.power)

    def __rpow__(self, other: Matrix | int | float) -> Matrix:
        return Matrix(other) ** self

    def __neg__(self) -> Matrix:
        return Matrix(-self.data)

    def __pos__(self) -> Matrix:
        return Matrix(np.copy(self.data))

    def __abs__(self) -> Matrix:
        return Matrix(np.abs(self.data))

    def __matmul__(self, other: Matrix) -> Matrix:
        return self.matmul(other, S=64)

    def matmul(self, other: Matrix, S: int = 64) -> Matrix:
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

    def solve(self, B: Matrix | np.ndarray, method: str = "lu") -> Matrix:
        from .decompositions import LU, Cholesky

        if method == "gauss":
            return self._solve_gaussian(B)
        elif method == "lu":
            return LU(self).solve(B)
        elif method == "cholesky":
            return Cholesky(self).solve(B)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _solve_gaussian(self, B: Matrix | np.ndarray) -> Matrix:
        from .solvers import TriangularSolver

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

    def eig(self, method: str = "qr") -> tuple[np.ndarray, Matrix]:
        from .solvers import EigenSolver

        solver = EigenSolver(self)
        if method == "qr":
            return solver.find_all_eigen()
        elif method == "power":
            return solver.power_iteration()
        else:
            raise ValueError(f"Unknown eigenvalue method: {method}")
