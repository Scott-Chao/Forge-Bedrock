import pytest
import numpy as np

from core.nn import DataLoader


# =========================================================
# 1. Correctness — batch sizes and iteration count
# =========================================================


class TestBatching:
    def test_batch_sizes_and_count(self):
        N, d_in, d_out = 50, 4, 2
        X = np.random.randn(N, d_in)
        y = np.random.randn(N, d_out)
        batch_size = 16

        loader = DataLoader(X, y, batch_size, shuffle=False)
        assert len(loader) == 4  # ceil(50 / 16)

        batches = list(loader)
        assert len(batches) == 4

        # First 3 full batches: size 16, last batch: size 2
        for i in range(3):
            bx, by = batches[i]
            assert bx.shape == (16, d_in)
            assert by.shape == (16, d_out)
            # Non-shuffled: data appears in original order
            np.testing.assert_array_equal(bx, X[i * 16 : (i + 1) * 16])
            np.testing.assert_array_equal(by, y[i * 16 : (i + 1) * 16])

        bx_last, by_last = batches[3]
        assert bx_last.shape == (2, d_in)
        assert by_last.shape == (2, d_out)
        np.testing.assert_array_equal(bx_last, X[48:])
        np.testing.assert_array_equal(by_last, y[48:])


# =========================================================
# 2. Edge — exact divisibility and single-element batch
# =========================================================


class TestEdgeCases:
    def test_exact_division(self):
        """batch_size divides N evenly — all batches the same size."""
        N, d = 32, 3
        X = np.random.randn(N, d)
        y = np.random.randn(N, 1)
        loader = DataLoader(X, y, batch_size=8, shuffle=False)
        assert len(loader) == 4
        for bx, by in loader:
            assert bx.shape[0] == 8
            assert by.shape[0] == 8

    def test_batch_size_larger_than_dataset(self):
        """One batch wraps the entire dataset when batch_size > N."""
        N, d = 5, 2
        X = np.random.randn(N, d)
        y = np.random.randn(N, 1)
        loader = DataLoader(X, y, batch_size=100, shuffle=False)
        assert len(loader) == 1
        bx, by = next(iter(loader))
        assert bx.shape == (5, d)
        assert by.shape == (5, 1)


# =========================================================
# 3. Shuffle — order differs between epochs
# =========================================================


class TestShuffle:
    def test_shuffle_changes_order(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 5))
        y = rng.normal(0, 1, (100, 1))

        loader = DataLoader(X, y, batch_size=10, shuffle=True)

        # Collect first-batch indices from epoch 1
        epoch1_first = next(iter(loader))[0].copy()
        # Collect first-batch indices from epoch 2
        epoch2_first = next(iter(loader))[0].copy()

        # With 100 samples and batch_size=10, the probability that two
        # randomly-shuffled epochs pick the EXACT same 10 samples is
        # negligible.  Use assert not (==) because the arr + allclose
        # combo can be fragile for shuffled data.
        assert not np.allclose(epoch1_first, epoch2_first)
