import numpy as np
import pytest
from core.prob.empirical import (
    EmpiricalDistribution,
    Histogram,
    fd_bin_width,
    rice_rule,
    sturges_rule,
)


class TestBinRules:
    def test_sturges_known(self):
        assert sturges_rule(1000) == 11

    def test_rice_known(self):
        assert rice_rule(1000) == 20

    def test_fd_constant(self):
        assert fd_bin_width(np.ones(100)) == 0.0


class TestHistogram:
    def test_density_integrates_to_one(self):
        h = Histogram().fit(np.random.randn(1000))
        assert np.sum(h._hist * h.bin_widths) == pytest.approx(1.0)

    def test_counts_without_density(self):
        data = np.random.randn(200)
        h = Histogram(density=False).fit(data)
        assert np.sum(h._hist) == len(data)

    def test_matches_numpy(self):
        data = np.random.randn(500)
        h = Histogram(bins=15, density=False).fit(data)
        np_counts, _ = np.histogram(data, bins=15)
        np.testing.assert_array_equal(h._hist, np_counts)

    def test_pdf_zero_outside_range(self):
        data = np.array([1.0, 2.0, 3.0])
        h = Histogram(bins=2).fit(data)
        assert h.pdf(-100) == 0.0
        assert h.pdf(100) == 0.0

    def test_pdf_boundary(self):
        """Points on bin edges should use the correct bin's density (side='right')."""
        data = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        h = Histogram(bins=np.array([0.0, 1.0, 2.0, 3.0]), density=True).fit(data)
        p_at_1 = h.pdf(1.0)  # should use bin [1, 2)
        p_in_bin1 = h.pdf(1.5)  # same bin
        assert p_at_1 == pytest.approx(p_in_bin1)

    @pytest.mark.parametrize("strategy", ["sturges", "rice", "fd"])
    def test_all_strategies(self, strategy):
        h = Histogram(bins=strategy).fit(np.random.randn(500))
        assert np.sum(h._hist * h.bin_widths) == pytest.approx(1.0)

    def test_constant_data_uses_one_bin(self):
        h = Histogram(bins="fd").fit(np.ones(50) * 5.0)
        assert len(h._hist) == 1


class TestEmpiricalDistribution:
    def test_cdf_known(self):
        ed = EmpiricalDistribution(np.array([1.0, 2.0, 3.0, 4.0]))
        assert ed.cdf(2.0) == 0.5
        assert ed.cdf(1.999) == 0.25

    def test_cdf_bounds_and_monotonic(self):
        data = np.random.randn(200)
        ed = EmpiricalDistribution(data)
        assert ed.cdf(-1e10) == 0.0 and ed.cdf(1e10) == 1.0
        xs = np.sort(data)
        assert np.all(np.diff(ed.cdf(xs)) >= 0)

    def test_quantile_median(self):
        ed = EmpiricalDistribution(np.random.randn(1000))
        assert ed.quantile(0.5) == pytest.approx(0.0, abs=0.2)

    def test_sample_shape(self):
        ed = EmpiricalDistribution(np.random.randn(100))
        assert ed.sample(50).shape == (50,)

    def test_mean_and_variance(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        ed = EmpiricalDistribution(data)
        assert ed.mean == 2.5
        assert ed.variance == np.var(data, ddof=1)


class TestErrorHandling:
    def test_unrecognized_strategy(self):
        with pytest.raises(ValueError, match="Unknown bin strategy"):
            Histogram(bins="invalid").fit(np.ones(10))

    def test_pdf_before_fit_raises(self):
        with pytest.raises(AttributeError):
            Histogram().pdf(0.0)
