"""Empirical distribution estimation from raw data.

This module implements the most fundamental form of density estimation:
turning raw observations into an empirical probability distribution.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Bin width / count selectors
# ---------------------------------------------------------------------------


def sturges_rule(n: int) -> int:
    """Sturges' rule: k = ceil(log2(n) + 1).

    Best for unimodal, roughly normal data. Tends to oversmooth
    when n is large or the distribution is multi-modal.
    """
    return int(np.ceil(np.log2(n) + 1))


def rice_rule(n: int) -> int:
    """Rice rule: k = ceil(2 * n**(1/3)).

    More conservative than Sturges — produces more bins, which is
    generally safer for exploratory analysis.
    """
    return int(np.ceil(2 * n ** (1 / 3)))


def fd_bin_width(data: np.ndarray) -> float:
    """Freedman-Diaconis rule: optimal bin width h = 2 * IQR / n**(1/3).

    Uses interquartile range, making it robust to outliers.
    Returns the *width*, not the count — you'll need to derive count
    from (max - min) / h.
    """
    IQR = np.percentile(data, 75) - np.percentile(data, 25)
    return 2 * IQR / (len(data) ** (1 / 3))


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


class Histogram:
    """Density estimation via a histogram.

    Wraps numpy.histogram with convenient bin-selection strategies and
    density=True by default so the result integrates to 1.

    Parameters
    ----------
    bins : int, str, or sequence, optional
        Number of bins, a strategy name ('sturges', 'rice'), or
        explicit bin edges. Default 'sturges'.
    density : bool
        If True (default), normalise so that the integral over the
        range equals 1 — a proper probability density estimate.
    """

    def __init__(self, bins: int | str = "sturges", density: bool = True):
        self.bins = bins
        self.density = density

    _STRATEGIES = {"sturges", "rice", "fd"}

    def _resolve_bins(self, data: np.ndarray) -> int | np.ndarray:
        """Convert the user's 'bins' argument into a concrete value for np.histogram."""
        if not isinstance(self.bins, str):
            return self.bins
        if self.bins not in self._STRATEGIES:
            raise ValueError(
                f"Unknown bin strategy '{self.bins}'. "
                f"Choose from {sorted(self._STRATEGIES)}."
            )
        if self.bins == "sturges":
            return sturges_rule(len(data))
        elif self.bins == "rice":
            return rice_rule(len(data))
        elif self.bins == "fd":
            h = fd_bin_width(data)
            return max(1, int(np.ptp(data) / h)) if h > 0.0 else 1

    def fit(self, data: np.ndarray) -> "Histogram":
        """Compute the histogram from raw data.

        Stores the bin edges and densities for later querying.

        Parameters
        ----------
        data : np.ndarray
            1-D array of observations.

        Returns
        -------
        self
        """
        bins = self._resolve_bins(data)
        self._hist, self._bin_edges = np.histogram(
            data, bins=bins, density=self.density
        )
        return self

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the estimated density at one or more points."""
        x = np.asarray(x, dtype=float)
        idx = np.clip(
            np.searchsorted(self._bin_edges, x, side="right") - 1,
            0,
            len(self._hist) - 1,
        )
        result = np.where(
            (x < self._bin_edges[0]) | (x > self._bin_edges[-1]), 0.0, self._hist[idx]
        )
        return result.item() if result.ndim == 0 else result

    @property
    def bin_widths(self) -> np.ndarray:
        """Width of each bin (bin_edges[1:] - bin_edges[:-1])."""
        return self._bin_edges[1:] - self._bin_edges[:-1]


# ---------------------------------------------------------------------------
# Empirical Distribution (ECDF-based)
# ---------------------------------------------------------------------------


class EmpiricalDistribution:
    """Empirical distribution based on the ECDF.

    Represents a distribution that places probability mass 1/n on each
    observed data point. This is the non-parametric, discrete counterpart
    to the histogram.

    Parameters
    ----------
    data : np.ndarray
        1-D array of observations.
    """

    def __init__(self, data: np.ndarray):
        self._data = np.sort(data)
        self._n = len(data)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Empirical CDF: F_n(t) = (1/n) * sum(1_{x_i <= t})."""
        return np.searchsorted(self._data, x, side="right") / self._n

    def quantile(self, q: float | np.ndarray) -> float | np.ndarray:
        """Inverse CDF / quantile function."""
        return np.quantile(self._data, q)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw n bootstrap samples from the empirical distribution."""
        rng = np.random.default_rng() if rng is None else rng
        return rng.choice(self._data, size=n, replace=True)

    @property
    def mean(self) -> float:
        """Sample mean: the expected value under the empirical distribution."""
        return np.mean(self._data)

    @property
    def variance(self) -> float:
        """Sample variance (unbiased, ddof=1) under the empirical distribution."""
        return np.var(self._data, ddof=1)
