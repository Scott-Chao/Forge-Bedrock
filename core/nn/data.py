"""
core/nn/data.py — Data loading and mini-batch iteration utilities.

Provides a simple DataLoader that splits dataset arrays into
mini-batches, with optional shuffling, following the same
iterator pattern as PyTorch's DataLoader.
"""

from __future__ import annotations

import numpy as np


class DataLoader:
    """Iterate over a dataset in mini-batches.

    Parameters
    ----------
    X : np.ndarray
        Input features, shape (N, d_in) where N is the total number
        of samples.
    y : np.ndarray
        Targets, shape (N, d_out).
    batch_size : int, default=32
        Number of samples per batch.
    shuffle : bool, default=True
        Whether to shuffle the data at the start of each epoch.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self._pos = 0

    def __iter__(self) -> DataLoader:
        self._pos = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pos >= len(self.indices):
            raise StopIteration
        idx = self.indices[self._pos : self._pos + self.batch_size]
        self._pos += self.batch_size
        return self.X[idx], self.y[idx]

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return (len(self.X) + self.batch_size - 1) // self.batch_size
