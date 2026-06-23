"""Theoretical probability distributions.

Implements four fundamental distributions that directly correspond to
the loss functions we'll derive in Phase 3 (MLE → Loss):

    Gaussian       → MSE          (regression)
    Laplacian      → L1 / Huber   (robust regression)
    Bernoulli      → BCE          (binary classification)
    Categorical    → CrossEntropy (multi-class classification)

Each distribution supports both density evaluation and sampling.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Gaussian (Normal) Distribution
# ---------------------------------------------------------------------------

class Gaussian:
    """Gaussian (normal) distribution: N(x | mu, sigma^2).

    Parameters
    ----------
    mu : float
        Mean parameter.
    sigma : float
        Standard deviation (> 0).
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self._mu = mu
        self._sigma = sigma

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Probability density function.

        p(x) = 1/sqrt(2 pi sigma^2) * exp(-(x - mu)^2 / (2 sigma^2))
        """
        z = (x - self._mu) / self._sigma
        p = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi * self._sigma ** 2)
        return p

    def log_pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Log-PDF (more numerically stable than log(pdf))."""
        z = (x - self._mu) / self._sigma
        log_p = -z ** 2 / 2 - 0.5 * np.log(2 * np.pi) - np.log(self._sigma)
        return log_p

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample from the Gaussian using Box-Muller transform."""
        rng = np.random.default_rng() if rng is None else rng
        u1, u2 = rng.random(n), rng.random(n)
        z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        return self._mu + z * self._sigma

    @property
    def mean(self) -> float:
        return self._mu

    @property
    def variance(self) -> float:
        return self._sigma ** 2


# ---------------------------------------------------------------------------
# Bernoulli Distribution
# ---------------------------------------------------------------------------

class Bernoulli:
    """Bernoulli distribution: Bern(x | p).

    Models a single coin-flip. x = 1 with probability p, x = 0 with prob 1-p.

    Parameters
    ----------
    p : float
        Probability of success (1), in [0, 1].
    """

    def __init__(self, p: float = 0.5):
        self._p = p

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Probability mass function: p(x) = p^x * (1-p)^(1-x)."""
        x = np.asarray(x)
        mask = (x == 0) | (x == 1)
        p = np.where(mask, self._p ** x * (1 - self._p) ** (1 - x), 0.0)
        return p.item() if p.ndim == 0 else p

    def log_pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Log-PMF: x*log(p) + (1-x)*log(1-p)."""
        x = np.asarray(x)
        mask = (x == 0) | (x == 1)
        log_p = np.where(
            mask,
            x * np.log(self._p) + (1 - x) * np.log(1 - self._p),
            -np.inf,
        )
        return log_p.item() if log_p.ndim == 0 else log_p

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample n independent outcomes."""
        rng = np.random.default_rng() if rng is None else rng
        u = rng.uniform(size=n)
        return np.where(u <= self._p, 1, 0)

    @property
    def mean(self) -> float:
        return self._p

    @property
    def variance(self) -> float:
        return self._p * (1 - self._p)


# ---------------------------------------------------------------------------
# Categorical Distribution
# ---------------------------------------------------------------------------

class Categorical:
    """Categorical distribution: Cat(x | probs).

    Generalises Bernoulli to K categories (a K-sided die).

    Parameters
    ----------
    probs : np.ndarray
        Probability vector of shape (K,), non-negative and summing to 1.
    """

    def __init__(self, probs: np.ndarray | None = None):
        self._probs = probs

    def pmf(self, k: int | np.ndarray) -> float | np.ndarray:
        """Probability mass function."""
        return self._probs[k]

    def log_pmf(self, k: int | np.ndarray) -> float | np.ndarray:
        """Log-PMF: log(probs[k])."""
        return np.log(self._probs[k])

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample n independent category indices."""
        rng = np.random.default_rng() if rng is None else rng
        u = rng.uniform(0, 1, size=n)
        cumsum = np.cumsum(self._probs)
        cumsum[-1] = 1.0
        return np.searchsorted(cumsum, u)

    @property
    def mean(self) -> np.ndarray:
        """Expected value (same as probs, since E[one-hot] = probs)."""
        return self._probs

    @property
    def variance(self) -> np.ndarray:
        """Variance for each category: p_k * (1 - p_k)."""
        return self._probs * (1 - self._probs)

    @property
    def covariance(self) -> np.ndarray:
        """Full K x K covariance matrix: diag(p) - p p^T."""
        return np.diag(self._probs) - np.outer(self._probs, self._probs)


# ---------------------------------------------------------------------------
# Laplacian Distribution
# ---------------------------------------------------------------------------

class Laplacian:
    """Laplacian distribution: Lap(x | mu, b).

    Also known as the double-exponential distribution.
    Has heavier tails than the Gaussian — robust to outliers.

    Parameters
    ----------
    mu : float
        Location parameter (mean, median, mode — all equal).
    b : float
        Scale parameter (> 0). Variance = 2 * b^2.
    """

    def __init__(self, mu: float = 0.0, b: float = 1.0):
        self._mu = mu
        self._b = b

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """PDF: p(x) = 1/(2b) * exp(-|x - mu| / b)."""
        z = np.abs(x - self._mu) / self._b
        p = np.exp(-z) / (2 * self._b)
        return p

    def log_pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Log-PDF: -log(2b) - |x - mu| / b."""
        z = np.abs(x - self._mu) / self._b
        log_p = -np.log(2 * self._b) - z
        return log_p

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample via inverse-CDF transform."""
        rng = np.random.default_rng() if rng is None else rng
        u = rng.uniform(size=n)
        return self._mu - self._b * np.sign(u - 0.5) * np.log(1 - 2 * np.abs(u - 0.5))

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """CDF: F(x) = 0.5 * exp((x - mu)/b) for x < mu,
                   1 - 0.5 * exp(-(x - mu)/b) for x >= mu.
        """
        z = (x - self._mu) / self._b
        return np.where(x < self._mu, 0.5 * np.exp(z), 1 - 0.5 * np.exp(-z))

    @property
    def mean(self) -> float:
        return self._mu

    @property
    def variance(self) -> float:
        return 2.0 * self._b ** 2
