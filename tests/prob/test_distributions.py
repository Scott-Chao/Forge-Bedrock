import numpy as np
import pytest
from core.prob.distributions import Bernoulli, Categorical, Gaussian, Laplacian

# =========================================================
# Gaussian
# =========================================================

class TestGaussian:
    def test_pdf_peak_at_mean(self):
        g = Gaussian(mu=2.0, sigma=1.5)
        assert g.pdf(2.0) == pytest.approx(1 / np.sqrt(2 * np.pi * 1.5 ** 2))

    def test_pdf_symmetric(self):
        g = Gaussian(mu=0.0, sigma=1.0)
        assert g.pdf(1.0) == pytest.approx(g.pdf(-1.0))

    def test_log_pdf_matches_np_log_pdf(self):
        g = Gaussian(mu=0.5, sigma=2.0)
        xs = np.array([-3.0, 0.0, 0.5, 3.0])
        np.testing.assert_allclose(g.log_pdf(xs), np.log(g.pdf(xs)))

    def test_pdf_integrates_to_one(self):
        g = Gaussian(mu=0.0, sigma=1.0)
        xs = np.linspace(-10, 10, 10001)
        dx = xs[1] - xs[0]
        integral = np.sum(g.pdf(xs)) * dx
        assert integral == pytest.approx(1.0, abs=1e-3)

    def test_box_muller_sample_stats(self):
        rng = np.random.default_rng(42)
        g = Gaussian(mu=5.0, sigma=2.0)
        samples = g.sample(200_000, rng)
        assert samples.mean() == pytest.approx(5.0, abs=0.02)
        assert samples.var() == pytest.approx(4.0, abs=0.02)

    def test_box_muller_shape(self):
        g = Gaussian()
        assert g.sample(100).shape == (100,)
        assert g.sample(1).shape == (1,)

    def test_mean_and_variance(self):
        g = Gaussian(mu=-1.0, sigma=3.0)
        assert g.mean == -1.0
        assert g.variance == 9.0

    def test_rng_reproducibility(self):
        rng = np.random.default_rng(42)
        a = Gaussian(mu=0.0, sigma=1.0).sample(100, rng)
        rng = np.random.default_rng(42)
        b = Gaussian(mu=0.0, sigma=1.0).sample(100, rng)
        np.testing.assert_array_equal(a, b)


# =========================================================
# Bernoulli
# =========================================================

class TestBernoulli:
    @pytest.mark.parametrize("p", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_pmf_valid(self, p):
        b = Bernoulli(p)
        assert b.pmf(1) == pytest.approx(p)
        assert b.pmf(0) == pytest.approx(1 - p)

    def test_pmf_invalid_x_returns_zero(self):
        b = Bernoulli(0.5)
        assert b.pmf(2) == 0.0
        assert b.pmf(-1) == 0.0

    def test_log_pmf_equals_log_pmf(self):
        b = Bernoulli(0.3)
        assert b.log_pmf(1) == pytest.approx(np.log(0.3))
        assert b.log_pmf(0) == pytest.approx(np.log(0.7))
        assert b.log_pmf(2) == -np.inf

    def test_sample_proportions(self):
        rng = np.random.default_rng(42)
        b = Bernoulli(p=0.6)
        samples = b.sample(200_000, rng)
        assert samples.mean() == pytest.approx(0.6, abs=0.005)

    def test_mean_and_variance(self):
        b = Bernoulli(p=0.3)
        assert b.mean == 0.3
        assert b.variance == 0.3 * 0.7


# =========================================================
# Categorical
# =========================================================

class TestCategorical:
    def test_pmf_known(self):
        c = Categorical(np.array([0.2, 0.3, 0.5]))
        assert c.pmf(0) == 0.2
        assert c.pmf(1) == 0.3
        assert c.pmf(2) == 0.5

    def test_log_pmf(self):
        c = Categorical(np.array([0.25, 0.75]))
        assert c.log_pmf(1) == pytest.approx(np.log(0.75))

    def test_sample_distribution(self):
        rng = np.random.default_rng(42)
        probs = np.array([0.1, 0.2, 0.7])
        c = Categorical(probs)
        samples = c.sample(200_000, rng)
        for k in range(3):
            assert (samples == k).mean() == pytest.approx(probs[k], abs=0.01)

    def test_covariance_properties(self):
        probs = np.array([0.2, 0.3, 0.5])
        c = Categorical(probs)
        cov = c.covariance
        assert cov.shape == (3, 3)
        np.testing.assert_allclose(np.sum(cov, axis=1), 0.0, atol=1e-15)
        np.testing.assert_allclose(cov, cov.T)

    def test_variance_vector(self):
        probs = np.array([0.2, 0.5, 0.3])
        c = Categorical(probs)
        expected = probs * (1 - probs)
        np.testing.assert_allclose(c.variance, expected)


# =========================================================
# Laplacian
# =========================================================

class TestLaplacian:
    def test_pdf_peak_at_mu(self):
        lap = Laplacian(mu=3.0, b=2.0)
        assert lap.pdf(3.0) == pytest.approx(1 / (2 * 2.0))

    def test_pdf_symmetric(self):
        lap = Laplacian(mu=0.0, b=1.0)
        assert lap.pdf(2.0) == pytest.approx(lap.pdf(-2.0))

    def test_log_pdf_matches_np_log_pdf(self):
        lap = Laplacian(mu=1.0, b=0.5)
        xs = np.array([-2.0, 0.0, 1.0, 3.0])
        np.testing.assert_allclose(lap.log_pdf(xs), np.log(lap.pdf(xs)))

    def test_pdf_integrates_to_one(self):
        lap = Laplacian(mu=0.0, b=1.0)
        xs = np.linspace(-20, 20, 20001)
        integral = np.sum(lap.pdf(xs)) * (xs[1] - xs[0])
        assert integral == pytest.approx(1.0, abs=1e-4)

    def test_cdf_known(self):
        lap = Laplacian(mu=0.0, b=1.0)
        assert lap.cdf(0.0) == 0.5
        assert lap.cdf(-1.0) == pytest.approx(0.5 * np.exp(-1))
        assert lap.cdf(1.0) == pytest.approx(1 - 0.5 * np.exp(-1))

    def test_cdf_monotonic(self):
        lap = Laplacian(mu=2.0, b=3.0)
        xs = np.linspace(-10, 14, 100)
        diffs = np.diff(lap.cdf(xs))
        assert np.all(diffs >= -1e-15)

    def test_inverse_cdf_sample_stats(self):
        rng = np.random.default_rng(42)
        lap = Laplacian(mu=0.0, b=2.0)
        samples = lap.sample(200_000, rng)
        assert samples.mean() == pytest.approx(0.0, abs=0.02)
        assert np.var(samples) == pytest.approx(8.0, abs=0.15)

    def test_mean_and_variance(self):
        lap = Laplacian(mu=3.0, b=0.5)
        assert lap.mean == 3.0
        assert lap.variance == 2 * 0.5 ** 2
