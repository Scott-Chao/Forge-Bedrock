from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .matrix import Matrix


class BroadcastEngine:
    @staticmethod
    def _get_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    @staticmethod
    def _get_config(
        matrix1: Matrix, matrix2: Matrix
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
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
    def execute(
        matrix1: Matrix,
        matrix2: Matrix,
        op: Callable[..., np.ndarray],
        out_data: np.ndarray | None = None,
    ) -> np.ndarray:
        target_shape, st1, st2 = BroadcastEngine._get_config(matrix1, matrix2)
        if out_data is None:
            out_data = np.empty(target_shape, dtype=matrix1.data.dtype)
        elif target_shape != matrix1.shape:
            raise ValueError("Cannot broadcast to inplace output shape")

        st_res = BroadcastEngine._get_strides(target_shape)

        f1, f2, fr = matrix1.data.ravel(), matrix2.data.ravel(), out_data.ravel()
        ndim = len(target_shape)

        def _worker(dim: int, off1: int, off2: int, off_r: int) -> None:
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
