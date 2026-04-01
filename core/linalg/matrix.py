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
