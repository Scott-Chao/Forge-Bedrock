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

        lu_obj = LUDecomposition(self)

        P_det = np.linalg.det(lu_obj.P.data)
        U_diag = np.diag(lu_obj.U.data)
        U_det = np.prod(U_diag)

        return P_det * U_det

    @property
    def logdet(self):
        """ln |det(A)| = ln |det(U)|"""
        if self.rows != self.cols:
            raise ValueError("Only square matrices have determinants.")

        lu_obj = LUDecomposition(self)
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

        n = self.rows
        I = Matrix.eye(n)

        lu_obj = LUDecomposition(self)

        inv_cols = []
        for i in range(n):
            e_i = Matrix(I.data[:, i])
            x_i = lu_obj.solve(e_i)
            inv_cols.append(x_i.data)

        return Matrix(np.hstack(inv_cols))

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

    def solve(self, b, method="lu"):
        if method == "gauss":
            return self._solve_gaussian(b)
        elif method == "lu":
            return LUDecomposition(self).solve(b)
        elif method == "cholesky":
            return CholeskyDecomposition(self).solve(b)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _solve_gaussian(self, b):
        if not isinstance(b, Matrix):
            b = Matrix(b)

        assert self.rows == self.cols, "Matrix must be square"
        assert self.rows == b.rows, "Dimension mismatch between A and b"

        n = self.rows
        aug = np.hstack([self.data.copy(), b.data.copy()])

        for k in range(n):
            pivot_idx = np.argmax(np.abs(aug[k:, k])) + k

            if np.abs(aug[pivot_idx, k]) < 1e-15:
                raise ValueError("Matrix is singular or nearly singular")

            if pivot_idx != k:
                aug[[k, pivot_idx]] = aug[[pivot_idx, k]]

            for i in range(k + 1, n):
                factor = aug[i, k] / aug[k, k]
                aug[i, k:] -= factor * aug[k, k:]

        x = np.zeros((n, 1))
        for i in range(n - 1, -1, -1):
            sum_ax = aug[i, i + 1 : n] @ x[i + 1 : n]
            x[i] = (aug[i, n] - sum_ax) / aug[i, i]

        return Matrix(x)


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


class LUDecomposition:
    def __init__(self, matrix):
        self.matrix = matrix
        self.n = matrix.rows
        self.P, self.L, self.U = self._decompose()

    def _decompose(self):
        n = self.n
        assert n == self.matrix.cols, "Matrix must be square"

        P = np.eye(n)
        L = np.eye(n)
        U = self.matrix.data.copy().astype(np.float64)

        for k in range(n):
            pivot_idx = np.argmax(np.abs(U[k:, k])) + k

            if np.abs(U[pivot_idx, k]) < 1e-15:
                raise ValueError("Matrix is singular and cannot be decomposed.")

            U[[k, pivot_idx]] = U[[pivot_idx, k]]
            P[[k, pivot_idx]] = P[[pivot_idx, k]]
            if k > 0:
                L[[k, pivot_idx], :k] = L[[pivot_idx, k], :k]

            for i in range(k + 1, n):
                factor = U[i, k] / U[k, k]
                L[i, k] = factor
                U[i, k:] -= factor * U[k, k:]

        return Matrix(P), Matrix(L), Matrix(U)

    def solve(self, b):
        if not isinstance(b, Matrix):
            b = Matrix(b)

        Pb = self.P @ b
        y = self._forward_substitution(self.L, Pb)
        x = self._backward_substitution(self.U, y)
        return x

    @staticmethod
    def _forward_substitution(L, b):
        """Solve Ly = b (L is lower triangular)"""
        n = L.rows
        y = np.zeros((n, 1))
        L_data, b_data = L.data, b.data
        for i in range(n):
            sum_ly = L_data[i, :i] @ y[:i]
            y[i] = (b_data[i] - sum_ly) / L_data[i, i]
        return Matrix(y)

    @staticmethod
    def _backward_substitution(U, y):
        """Solve Ux = y (U is upper triangular)"""
        n = U.rows
        x = np.zeros((n, 1))
        U_data, y_data = U.data, y.data
        for i in range(n - 1, -1, -1):
            sum_ux = U_data[i, i + 1 :] @ x[i + 1 :]
            x[i] = (y_data[i] - sum_ux) / U_data[i, i]
        return Matrix(x)


class CholeskyDecomposition:
    def __init__(self, matrix):
        self.matrix = matrix
        self.n = matrix.rows
        if not matrix.is_symmetric:
            raise ValueError("Matrix must be symmetric for Cholesky Decomposition.")
        self.L = self._decompose()

    def _decompose(self):
        n = self.n
        A = self.matrix.data
        L = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1):
                s = L[i, :j] @ L[j, :j]
                if i == j:
                    val = A[i, i] - s
                    if val <= 0:
                        raise ValueError("Matrix is not positive-definite.")
                    L[i, j] = np.sqrt(val)
                else:
                    L[i, j] = (A[i, j] - s) / L[j, j]

        return Matrix(L)

    def solve(self, b):
        if not isinstance(b, Matrix):
            b = Matrix(b)

        y = LUDecomposition._forward_substitution(self.L, b)
        x = LUDecomposition._backward_substitution(self.L.T, y)
        return x
