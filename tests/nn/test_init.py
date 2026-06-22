import numpy as np
import pytest
from core.nn import Parameter
from core.nn.init import kaiming_uniform_, xavier_uniform_

INIT_FNS = [xavier_uniform_, kaiming_uniform_]


# =========================================================
# 1. Correctness — bound and sample statistics
# =========================================================


class TestXavierUniform:
    @pytest.mark.parametrize(
        "shape, gain",
        [
            ((64, 128), 1.0),
            ((64, 128), 2.0),
            ((64, 128), 0.5),
        ],
    )
    def test_bounds_and_stats(self, shape, gain):
        p = Parameter(np.empty(shape))
        xavier_uniform_(p, gain=gain)

        fan_in, fan_out = shape[1], shape[0]
        bound = gain * np.sqrt(6 / (fan_in + fan_out))

        assert np.all(p.data >= -bound - 1e-12)
        assert np.all(p.data <= bound + 1e-12)
        assert abs(p.data.mean()) < 0.05
        # variance = (2 * gain^2) / (fan_in + fan_out)
        expected_var = 2 * gain**2 / (fan_in + fan_out)
        assert abs(p.data.var() - expected_var) < 0.02


class TestKaimingUniform:
    @pytest.mark.parametrize(
        "shape, gain",
        [
            ((64, 128), 1.0),
            ((64, 128), np.sqrt(2)),
            ((64, 128), 2.0),
        ],
    )
    def test_bounds_and_stats(self, shape, gain):
        p = Parameter(np.empty(shape))
        kaiming_uniform_(p, gain=gain)

        fan_in = shape[1]
        bound = gain * np.sqrt(3 / fan_in)

        assert np.all(p.data >= -bound - 1e-12)
        assert np.all(p.data <= bound + 1e-12)
        assert abs(p.data.mean()) < 0.05
        expected_var = gain**2 / fan_in
        assert abs(p.data.var() - expected_var) < 0.02


# =========================================================
# 2. Edge Cases
# =========================================================


class TestEdgeCases:
    @pytest.mark.parametrize("init_fn", INIT_FNS)
    def test_minimal_1x1(self, init_fn):
        p = Parameter(np.empty((1, 1)))
        init_fn(p, gain=1.0)
        assert np.isfinite(p.data[0, 0])

    @pytest.mark.parametrize("init_fn", INIT_FNS)
    def test_non_square(self, init_fn):
        p = Parameter(np.empty((100, 10)))
        init_fn(p, gain=1.0)
        assert np.all(np.isfinite(p.data))

    @pytest.mark.parametrize("init_fn", INIT_FNS)
    def test_reproducible_with_seed(self, init_fn):
        p1 = Parameter(np.empty((32, 64)))
        p2 = Parameter(np.empty((32, 64)))
        np.random.seed(42)
        init_fn(p1, gain=1.0)
        np.random.seed(42)
        init_fn(p2, gain=1.0)
        np.testing.assert_array_equal(p1.data, p2.data)
