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
        if isinstance(other, Matrix):
            return Matrix(op(self.data, other.data))
        elif isinstance(other, (int, float, np.float64, np.int64)):
            return Matrix(op(self.data, other))
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

    def _apply_inplace_op(self, other, op):
        if isinstance(other, Matrix):
            op(self.data, other.data, out=self.data)
        elif isinstance(other, (int, float, np.float64, np.int64)):
            op(self.data, other, out=self.data)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")
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
