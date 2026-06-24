"""Tests for core.prob.bias_variance — Bias-Variance Decomposition."""

import numpy as np
import pytest
from core.prob.bias_variance import (
    ConstantRegressor,
    PolynomialRegressor,
    bias_variance_decomposition,
    bootstrap_datasets,
    generate_data,
    sinusoidal,
    step_function,
    train_on_bootstraps,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# =========================================================
# 1. Ground-truth functions
# =========================================================


class TestGroundTruth:
    def test_known_points(self):
        """Verify sinusoidal and step at analytically known values."""
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(sinusoidal(x), [0, 1, 0, -1, 0], atol=1e-10)
        np.testing.assert_array_equal(step_function(np.array([-1, 0, 1])), [0, 0, 1])


# =========================================================
# 2. Data generation
# =========================================================


class TestGenerateData:
    def test_output_shapes(self, rng):
        x, y = generate_data(sinusoidal, n_samples=100, noise_std=0.3, rng=rng)
        assert x.shape == (100,)
        assert y.shape == (100,)

    def test_noise_level_matches(self, rng):
        x, y = generate_data(sinusoidal, n_samples=5000, noise_std=0.5, rng=rng)
        assert np.std(y - sinusoidal(x)) == pytest.approx(0.5, abs=0.05)


# =========================================================
# 3. Bootstrap resampling
# =========================================================


class TestBootstrap:
    def test_bootstraps_correct_count_and_with_replacement(self, rng):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10, 20, 30, 40, 50])
        boots = bootstrap_datasets(x, y, n_bootstraps=50, rng=rng)
        assert len(boots) == 50
        assert all(len(xb) == 5 and len(yb) == 5 for xb, yb in boots)
        assert any(len(np.unique(xb)) < 5 for xb, _ in boots)


# =========================================================
# 4. Model classes
# =========================================================


class TestModel:
    def test_polynomial_interpolates_exactly(self):
        """A degree (n-1) polynomial interpolates n points exactly."""
        x = np.array([0.0, 0.3, 0.7, 1.0])
        y = np.array([0.5, 1.2, 0.8, 0.3])
        model = PolynomialRegressor(degree=3)
        model.fit(x, y)
        np.testing.assert_allclose(model.predict(x), y, atol=1e-10)

    def test_polynomial_degree_0_is_constant(self, rng):
        x, y = generate_data(sinusoidal, 20, 0.3, rng=rng)
        model = PolynomialRegressor(degree=0)
        model.fit(x, y)
        preds = model.predict(np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(preds, y.mean(), atol=1e-10)

    def test_constant_regressor(self):
        x = np.array([0.1, 0.2, 0.3, 0.4])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        model = ConstantRegressor()
        model.fit(x, y)
        preds = model.predict(np.linspace(0, 1, 50))
        assert preds.shape == (50,)
        np.testing.assert_allclose(preds, 5.0, atol=1e-10)


# =========================================================
# 5. Training pipeline
# =========================================================


class TestTraining:
    def test_train_on_bootstraps_produces_fitted_models(self, rng):
        x, y = generate_data(sinusoidal, 20, 0.3, rng=rng)
        boots = bootstrap_datasets(x, y, n_bootstraps=5, rng=rng)
        models = train_on_bootstraps(PolynomialRegressor, boots)
        assert len(models) == 5
        assert all(not np.isnan(m.predict(np.array([0.5]))).any() for m in models)


# =========================================================
# 6. Bias-Variance Decomposition
# =========================================================


class TestBiasVarianceDecomposition:
    @pytest.fixture
    def setup(self, rng):
        x, y = generate_data(sinusoidal, 50, 0.3, rng=rng)
        boots = bootstrap_datasets(x, y, n_bootstraps=100, rng=rng)
        models = train_on_bootstraps(PolynomialRegressor, boots, {"degree": 3})
        x_test = np.linspace(0, 1, 200)
        f_true = sinusoidal(x_test).flatten()
        return models, x_test, f_true

    def test_expected_keys(self, setup):
        models, x_test, f_true = setup
        result = bias_variance_decomposition(models, x_test, f_true, 0.3)
        assert set(result) == {
            "bias_sq",
            "variance",
            "irreducible",
            "total",
            "avg_bias_sq",
            "avg_variance",
        }

    def test_components_sum_to_total(self, setup):
        models, x_test, f_true = setup
        result = bias_variance_decomposition(models, x_test, f_true, 0.3)
        np.testing.assert_allclose(
            result["total"],
            result["bias_sq"] + result["variance"] + result["irreducible"],
        )

    def test_irreducible_matches_noise(self, setup):
        models, x_test, f_true = setup
        result = bias_variance_decomposition(models, x_test, f_true, noise_std=0.3)
        assert result["irreducible"] == pytest.approx(0.09)

    def test_low_degree_high_bias_low_variance(self, rng):
        """degree=1 has high bias, low variance — classic trade-off."""
        x, y = generate_data(sinusoidal, 30, 0.3, rng=rng)
        boots = bootstrap_datasets(x, y, n_bootstraps=50, rng=rng)
        models = train_on_bootstraps(PolynomialRegressor, boots, {"degree": 1})
        x_test = np.linspace(0, 1, 100)
        result = bias_variance_decomposition(models, x_test, sinusoidal(x_test), 0.3)
        assert result["avg_bias_sq"] > result["avg_variance"] + 0.1
