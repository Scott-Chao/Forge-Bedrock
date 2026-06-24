"""Tests for core.prob.info_theory — Entropy, KL, Cross-Entropy, JS, Mutual Info."""

import numpy as np
import pytest
from core.prob.info_theory import (
    cross_entropy,
    entropy,
    entropy_from_counts,
    js_divergence,
    kl_divergence,
    mutual_information,
)


@pytest.fixture
def uniform():
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def skewed():
    return np.array([0.7, 0.2, 0.1]), np.array([0.3, 0.3, 0.4])


# =========================================================
# Entropy
# =========================================================


class TestEntropy:
    def test_uniform_max(self, uniform):
        assert entropy(uniform) == pytest.approx(np.log(4))

    def test_deterministic_zero(self):
        assert entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)

    def test_base_2_bits(self, uniform):
        assert entropy(uniform, base=2) == pytest.approx(2.0)

    def test_handles_zeros_gracefully(self):
        assert not np.isnan(entropy(np.array([0.5, 0.0, 0.5])))

    def test_entropy_from_counts(self):
        assert entropy_from_counts(np.array([10, 20, 30])) == pytest.approx(
            entropy(np.array([1 / 6, 1 / 3, 1 / 2]))
        )
        assert entropy_from_counts(np.array([0, 0])) == pytest.approx(0.0)


# =========================================================
# KL Divergence
# =========================================================


class TestKLDivergence:
    def test_identical_zero(self):
        p = np.array([0.2, 0.3, 0.5])
        assert kl_divergence(p, p) == pytest.approx(0.0)

    def test_nonnegative(self, skewed):
        p, q = skewed
        assert kl_divergence(p, q) >= 0
        assert kl_divergence(q, p) >= 0

    def test_asymmetric(self, skewed):
        p, q = skewed
        assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))


# =========================================================
# JS Divergence
# =========================================================


class TestJSDivergence:
    def test_symmetric(self, skewed):
        p, q = skewed
        assert js_divergence(p, q) == pytest.approx(js_divergence(q, p))

    def test_bounded_by_log2(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert 0 < js_divergence(p, q) <= np.log(2) + 1e-12


# =========================================================
# Cross-Entropy
# =========================================================


class TestCrossEntropy:
    def test_equals_entropy_when_identical(self):
        p = np.array([0.7, 0.2, 0.1])
        assert cross_entropy(p, p) == pytest.approx(entropy(p))

    def test_entropy_plus_kl(self, skewed):
        p, q = skewed
        assert cross_entropy(p, q) == pytest.approx(entropy(p) + kl_divergence(p, q))


# =========================================================
# Mutual Information
# =========================================================


class TestMutualInformation:
    def test_independent_zero(self):
        joint = np.outer(np.array([0.3, 0.7]), np.array([0.4, 0.6]))
        assert mutual_information(joint) == pytest.approx(0.0, abs=1e-15)

    def test_dependent_positive(self):
        joint = np.array([[0.1, 0.2], [0.3, 0.4]])
        assert mutual_information(joint) > 0

    def test_mi_nonnegative(self):
        assert mutual_information(np.array([[0.2, 0.1], [0.3, 0.4]])) >= 0
