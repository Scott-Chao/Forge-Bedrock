"""Bias-Variance Decomposition: empirical simulation.

Estimate the three components of a model's expected prediction error:

    E[(y - ŷ)²] = Bias²(x) + Var(x) + σ²

By training many models on bootstrap-resampled datasets from a known
ground-truth function, we can measure each component empirically.

This bridges the probability distributions we've implemented (for
generating noise) with the neural network models from Phase 2.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Ground-truth generators (known f(x) for synthetic experiments)
# ---------------------------------------------------------------------------


def sinusoidal(x: np.ndarray) -> np.ndarray:
    """Ground truth: f(x) = sin(2πx)."""
    return np.sin(2 * np.pi * x)


def polynomial_3(x: np.ndarray) -> np.ndarray:
    """Ground truth: f(x) = x + x² - 0.5 * x³."""
    return x + x**2 - 0.5 * x**3


def step_function(x: np.ndarray) -> np.ndarray:
    """Ground truth: f(x) = 1 where x > 0, else 0."""
    return np.where(x > 0, 1, 0)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_data(
    f: callable,
    n_samples: int,
    noise_std: float,
    x_range: tuple[float, float] = (0.0, 1.0),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data: y = f(x) + ε, where ε ~ N(0, noise_std²).

    Parameters
    ----------
    f : callable
        Ground-truth function f(x).
    n_samples : int
        Number of training points.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    x_range : tuple
        (min, max) for uniform x sampling.
    rng : np.random.Generator, optional
        For reproducible randomness.

    Returns
    -------
    x : np.ndarray, shape (n_samples,)
        Input features, uniformly sampled.
    y : np.ndarray, shape (n_samples,)
        Noisy observations: y = f(x) + ε.
    """
    rng = np.random.default_rng() if rng is None else rng
    x = rng.uniform(x_range[0], x_range[1], size=n_samples)
    y = f(x) + rng.normal(0, noise_std, size=n_samples)
    return x, y


# ---------------------------------------------------------------------------
# Bootstrap training
# ---------------------------------------------------------------------------


def bootstrap_datasets(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstraps: int,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate n_bootstraps resampled datasets with replacement."""
    rng = np.random.default_rng() if rng is None else rng
    n = len(x)
    datasets = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        datasets.append((x[idx], y[idx]))
    return datasets


def train_on_bootstraps(
    model_class: type,
    bootstrap_sets: list[tuple[np.ndarray, np.ndarray]],
    model_kwargs: dict | None = None,
) -> list:
    """Train one model per bootstrap dataset.

    Parameters
    ----------
    model_class : type
        A callable class that takes no arguments in __init__ and has
        .fit(x, y) and .predict(x) methods (e.g., sklearn's regressors,
        or a simple polynomial regression implemented below).
    bootstrap_sets : list of (x, y) tuples
    model_kwargs : dict, optional
        Additional kwargs passed to model_class().

    Returns
    -------
    models : list
        Trained model instances, one per bootstrap.
    """
    models = []
    if model_kwargs is None:
        model_kwargs = {}

    for x_boot, y_boot in bootstrap_sets:
        model = model_class(**model_kwargs)
        model.fit(x_boot, y_boot)
        models.append(model)

    return models


# ---------------------------------------------------------------------------
# Bias-Variance computation
# ---------------------------------------------------------------------------


def bias_variance_decomposition(
    models: list,
    x_test: np.ndarray,
    f_true: np.ndarray,
    noise_std: float,
) -> dict:
    """Compute bias², variance, and total error at each test point.

    For each test point x_i, we have M predictions (one per model).
    Then:
        ŷ_bar(x_i)     = mean of predictions
        Bias²(x_i)     = (f_true(x_i) - ŷ_bar(x_i))²
        Var(x_i)       = variance of predictions around ŷ_bar
        Total(x_i)     = mean squared error vs noisy observations (estimated)
        Irreducible    = noise_std² (known from our synthetic setup)

    Parameters
    ----------
    models : list
        Trained model instances, each with .predict().
    x_test : np.ndarray
        Test points to evaluate at, shape (n_test, 1).
    f_true : np.ndarray
        True function values at x_test, shape (n_test,).
    noise_std : float
        Known noise standard deviation (for computing irreducible error).

    Returns
    -------
    result : dict with keys:
        'bias_sq'      : np.ndarray, shape (n_test,)
        'variance'     : np.ndarray, shape (n_test,)
        'irreducible'  : float
        'total'        : np.ndarray, shape (n_test,)  — sum of the three
        'avg_bias_sq'  : float (average over test points)
        'avg_variance' : float (average over test points)
    """
    preds_list = [m.predict(x_test).flatten() for m in models]
    predictions = np.column_stack(preds_list)

    y_bar = np.mean(predictions, axis=1)
    bias_sq = (f_true - y_bar) ** 2
    variance = np.var(predictions, axis=1)
    irreducible = noise_std**2

    return {
        "bias_sq": bias_sq,
        "variance": variance,
        "irreducible": irreducible,
        "total": bias_sq + variance + irreducible,
        "avg_bias_sq": float(bias_sq.mean()),
        "avg_variance": float(variance.mean()),
    }


# ---------------------------------------------------------------------------
# Simple models for experimentation (no sklearn dependency)
# ---------------------------------------------------------------------------


class PolynomialRegressor:
    """Simple polynomial regression for bias-variance experiments.

    Fits a polynomial of specified degree to data via least squares,
    using the normal equation from Phase 1.

    Parameters
    ----------
    degree : int
        Degree of the polynomial. degree=0 is constant, degree=1 is linear.
    """

    def __init__(self, degree: int = 1):
        self._degree = degree
        self._coef = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit polynomial coefficients via the normal equation."""
        X = np.vander(x, self._degree + 1, increasing=True)
        self._coef = np.linalg.lstsq(X, y)[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict using the fitted polynomial coefficients."""
        X = np.vander(x, self._degree + 1, increasing=True)
        return X @ self._coef


class ConstantRegressor:
    """Always predicts the mean of y. Max bias, zero variance."""

    def __init__(self):
        self._mean = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self._mean = y.mean()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(x, self._mean, dtype=float)
