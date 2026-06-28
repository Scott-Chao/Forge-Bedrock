import numpy as np
import pytest
from core.nn import NAG, SGD, AdaGrad, Adam, AdamW, Momentum, Optimizer, RMSProp
from core.nn.parameter import Parameter

# =========================================================
# 1. Correctness — SGD (existing baseline)
# =========================================================


class TestSGDStep:
    def test_parameter_updated_in_negative_gradient_direction(self):
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        p.grad = np.array([0.1, -0.2, 0.3])
        optim = SGD([p], lr=0.5)

        optim.step()

        expected = np.array([1.0, 2.0, 3.0]) - 0.5 * np.array([0.1, -0.2, 0.3])
        np.testing.assert_allclose(p.data, expected)

    def test_zero_grad(self):
        p = Parameter(np.array([1.0, 2.0]))
        p.grad = np.array([0.5, -0.5])
        optim = SGD([p])

        optim.zero_grad()

        assert p.grad == 0.0


# =========================================================
# 2. Correctness — Momentum
# =========================================================


class TestMomentum:
    def test_first_step_formula(self):
        """First step: v = lr * g,  param -= v"""
        p = Parameter(np.array([1.0, 2.0]))
        p.grad = np.array([0.5, -0.3])
        opt = Momentum([p], lr=0.1, momentum=0.9)

        opt.step()

        # v = 0.9*0 + 0.1*[0.5, -0.3] = [0.05, -0.03]
        # data -= [0.05, -0.03]
        expected = np.array([1.0, 2.0]) - np.array([0.05, -0.03])
        np.testing.assert_allclose(p.data, expected)

    def test_velocity_accumulates(self):
        """Second step incorporates previous velocity."""
        p = Parameter(np.array([0.0]))
        p.grad = np.array([1.0])
        opt = Momentum([p], lr=0.1, momentum=0.9)

        opt.step()  # v1 = 0.1, data = -0.1
        p.grad = np.array([1.0])
        opt.step()  # v2 = 0.9*0.1 + 0.1 = 0.19, data = -0.1 - 0.19 = -0.29

        np.testing.assert_allclose(p.data, np.array([-0.29]))

    def test_zero_grad_preserves_velocity(self):
        """zero_grad() clears the gradient but leaves velocity intact."""
        p = Parameter(np.array([1.0]))
        p.grad = np.array([1.0])
        opt = Momentum([p], lr=0.5, momentum=0.9)
        opt.step()  # v = 0.5, data = 0.5
        opt.zero_grad()  # grad → 0.0, v still 0.5

        p.grad = np.array([1.0])
        opt.step()  # v = 0.9*0.5 + 0.5 = 0.95

        np.testing.assert_allclose(p.data, np.array([0.5 - 0.95]))

    def test_skip_parameter_with_none_grad(self):
        """Parameter with grad=None is left untouched."""
        p1 = Parameter(np.array([1.0]))
        p1.grad = np.array([0.5])
        p2 = Parameter(np.array([2.0]))
        p2.grad = None
        opt = Momentum([p1, p2], lr=0.1, momentum=0.9)
        opt.step()

        np.testing.assert_allclose(p1.data, 1.0 - 0.05)
        np.testing.assert_allclose(p2.data, 2.0)


# =========================================================
# 3. Correctness — NAG
# =========================================================


class TestNAG:
    def test_first_step_formula(self):
        """First step: v = lr * g,  param -= beta * v + lr * g"""
        p = Parameter(np.array([1.0, 2.0]))
        p.grad = np.array([0.5, -0.3])
        opt = NAG([p], lr=0.1, momentum=0.9)

        opt.step()

        # v = 0.9*0 + 0.1*[0.5, -0.3] = [0.05, -0.03]
        # param -= 0.9*[0.05, -0.03] + 0.1*[0.5, -0.3]
        v = np.array([0.05, -0.03])
        expected = np.array([1.0, 2.0]) - (0.9 * v + np.array([0.05, -0.03]))
        np.testing.assert_allclose(p.data, expected)

    def test_velocity_accumulates(self):
        """NAG velocity carries over across steps."""
        p = Parameter(np.array([0.0]))
        p.grad = np.array([1.0])
        opt = NAG([p], lr=0.1, momentum=0.9)

        opt.step()  # v1 = 0.1
        data1 = p.data.copy()

        p.grad = np.array([1.0])
        opt.step()  # v2 = 0.9*0.1 + 0.1 = 0.19
        # data2 = data1 - (0.9 * 0.19 + 0.1)

        np.testing.assert_allclose(p.data, data1 - (0.9 * 0.19 + 0.1))


# =========================================================
# 4. Convergence Behaviour
# =========================================================


class TestConvergence:
    @pytest.mark.parametrize(
        "opt_cls,kwargs,steps",
        [
            pytest.param(Momentum, dict(lr=0.1, momentum=0.8), 50, id="Momentum"),
            pytest.param(NAG, dict(lr=0.1, momentum=0.8), 50, id="NAG"),
            pytest.param(AdaGrad, dict(lr=1.0), 100, id="AdaGrad"),
            pytest.param(RMSProp, dict(lr=1.0, beta=0.9), 100, id="RMSProp"),
            pytest.param(Adam, dict(lr=0.1, betas=(0.9, 0.999)), 500, id="Adam"),
            pytest.param(
                AdamW,
                dict(lr=0.1, betas=(0.9, 0.999), weight_decay=0.01),
                500,
                id="AdamW",
            ),
        ],
    )
    def test_decreases_loss_on_quadratic(self, opt_cls, kwargs, steps):
        """All optimizers decrease loss on f(x) = ½ x² given enough steps."""
        p = Parameter(np.array([10.0]))
        opt = opt_cls([p], **kwargs)
        losses = []
        for _ in range(steps):
            p.grad = np.array([2.0 * p.data.item()])  # ∇½x² = x
            opt.step()
            losses.append(0.5 * p.data.item() ** 2)
        assert losses[-1] < losses[0] * 0.01


# =========================================================
# 5. Inheritance & Error Handling
# =========================================================


class TestOptimizerBase:
    def test_step_not_implemented(self):
        opt = Optimizer([], lr=0.01)
        with pytest.raises(NotImplementedError):
            opt.step()


# =========================================================
# 6. Correctness — AdaGrad
# =========================================================


class TestAdaGrad:
    def test_first_step_formula(self):
        """First step: cache = g², adjusted_lr = lr / (sqrt(cache) + eps)."""
        p = Parameter(np.array([4.0]))
        p.grad = np.array([3.0])
        opt = AdaGrad([p], lr=0.5, eps=1e-8)

        opt.step()

        # cache = 3² = 9
        # adjusted_lr = 0.5 / (sqrt(9) + 1e-8) = 0.5/3
        # param = 4 - (0.5/3) * 3 = 4 - 0.5 = 3.5
        expected = 3.5
        np.testing.assert_allclose(p.data, np.array([expected]))

    def test_cache_grows_monotonically(self):
        """AdaGrad cache only increases, never decreases."""
        p = Parameter(np.array([1.0]))
        opt = AdaGrad([p], lr=0.1, eps=1e-8)
        prev_cache = None
        for step_mag in [1.0, 2.0, 0.5, 3.0]:
            p.grad = np.array([step_mag])
            opt.step()
            c = opt.caches[0].copy()
            if prev_cache is not None:
                assert np.all(c >= prev_cache), "cache must be monotonic"
            prev_cache = c

    def test_zero_grad_preserves_cache(self):
        """zero_grad() clears gradient, leaves cache intact."""
        p = Parameter(np.array([1.0]))
        p.grad = np.array([2.0])
        opt = AdaGrad([p], lr=0.1)
        opt.step()  # cache = 4.0
        cache_before = opt.caches[0].copy()

        opt.zero_grad()  # grad → 0.0

        np.testing.assert_array_equal(opt.caches[0], cache_before)

    def test_skip_parameter_with_none_grad(self):
        """Parameter with grad=None is untouched, and its cache stays 0."""
        p1 = Parameter(np.array([1.0]))
        p1.grad = np.array([0.5])
        p2 = Parameter(np.array([2.0]))
        p2.grad = None
        opt = AdaGrad([p1, p2], lr=0.1)
        opt.step()

        np.testing.assert_allclose(
            p1.data, np.array([1.0 - 0.1 / np.sqrt(0.25 + 1e-8) * 0.5])
        )
        np.testing.assert_allclose(p2.data, np.array([2.0]))
        np.testing.assert_allclose(opt.caches[1], np.array([0.0]))

    def test_per_parameter_scaling(self):
        """Larger gradient dimension gets stronger LR suppression."""
        p = Parameter(np.array([0.0, 0.0]))
        p.grad = np.array([1.0, 10.0])
        opt = AdaGrad([p], lr=1.0, eps=1e-8)
        opt.step()

        # dim0: cache=1,   adjusted_lr = 1/(1+eps)=1,     step=-1.0
        # dim1: cache=100, adjusted_lr = 1/(10+eps)=0.1,  step=-1.0 (-0.1*10)
        # Both have same numerical update, but different adjusted_lrs
        assert opt.caches[0][0] < opt.caches[0][1]


# =========================================================
# 7. Correctness — RMSProp
# =========================================================


class TestRMSProp:
    def test_first_step_formula(self):
        """First step: cache = (1-beta) * g², adjusted_lr = lr / (sqrt(cache) + eps)."""
        p = Parameter(np.array([4.0]))
        p.grad = np.array([3.0])
        opt = RMSProp([p], lr=0.5, beta=0.9, eps=1e-8)

        opt.step()

        # cache = 0.9*0 + 0.1*9 = 0.9
        # adjusted_lr = 0.5 / (sqrt(0.9) + 1e-8) ≈ 0.5 / 0.94868
        # param = 4 - adjusted_lr * 3
        cache = 0.1 * 9.0
        adjusted_lr = 0.5 / (np.sqrt(cache) + 1e-8)
        expected = 4.0 - adjusted_lr * 3.0
        np.testing.assert_allclose(p.data, np.array([expected]))

    def test_cache_decays_when_gradient_drops(self):
        """RMSProp cache shrinks when gradient magnitude decreases (unlike AdaGrad)."""
        p = Parameter(np.array([0.0]))
        opt = RMSProp([p], lr=0.1, beta=0.9, eps=1e-8)

        # Initially large gradients
        for _ in range(10):
            p.grad = np.array([10.0])
            opt.step()

        peak_cache = opt.caches[0].item()

        # Then tiny gradients
        for _ in range(20):
            p.grad = np.array([0.01])
            opt.step()

        decayed_cache = opt.caches[0].item()
        assert decayed_cache < peak_cache * 0.5, (
            "RMSProp cache should decay when gradients shrink"
        )

    def test_zero_grad_preserves_cache(self):
        """zero_grad() clears gradient, leaves cache intact."""
        p = Parameter(np.array([1.0]))
        p.grad = np.array([2.0])
        opt = RMSProp([p], lr=0.1, beta=0.9)
        opt.step()
        cache_before = opt.caches[0].copy()

        opt.zero_grad()

        np.testing.assert_array_equal(opt.caches[0], cache_before)

    def test_skip_parameter_with_none_grad(self):
        """Parameter with grad=None is untouched."""
        p1 = Parameter(np.array([1.0]))
        p1.grad = np.array([0.5])
        p2 = Parameter(np.array([2.0]))
        p2.grad = None
        opt = RMSProp([p1, p2], lr=0.1, beta=0.9)
        opt.step()

        np.testing.assert_allclose(p2.data, np.array([2.0]))
        np.testing.assert_allclose(opt.caches[1], np.array([0.0]))


# =========================================================
# 8. Correctness — Adam
# =========================================================


class TestAdam:
    def test_first_step_bias_correction(self):
        """At t=1, m_hat=g and v_hat=g², so step = lr * sign(g)."""
        p = Parameter(np.array([1.0, 2.0]))
        p.grad = np.array([0.5, -0.3])
        opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8)

        opt.step()

        # m = 0.9*0 + 0.1*[0.5, -0.3] = [0.05, -0.03]
        # v = 0.999*0 + 0.001*[0.25, 0.09] = [0.00025, 0.00009]
        # m_hat = [0.05,-0.03] / (1-0.9)    = [0.5, -0.3]
        # v_hat = [2.5e-4,9e-5] / (1-0.999) = [0.25, 0.09]
        # step  = 0.1 * [0.5, -0.3] / (sqrt[0.25,0.09] + eps)
        #       = 0.1 * [0.5/0.5, -0.3/0.3] = [0.1, -0.1]
        expected = np.array([1.0, 2.0]) - np.array([0.1, -0.1])
        np.testing.assert_allclose(p.data, expected, atol=1e-6)

    def test_bias_correction_decays(self):
        """At t=2, correction factor (1-beta^t) is closer to 1."""
        p = Parameter(np.array([0.0]))
        p.grad = np.array([1.0])
        opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8)

        opt.step()  # t=1: correction = 1/(1-0.9) = 10
        data_t1 = p.data.copy()

        p.grad = np.array([1.0])
        opt.step()  # t=2: correction = 1/(1-0.9²) ≈ 5.26

        # At t=2 the correction is weaker, so the step is smaller
        step_t1 = abs(data_t1.item() - 0.0)  # first step size
        step_t2 = abs(p.data.item() - data_t1.item())  # second step size
        assert step_t2 < step_t1, "Bias correction decays, so later steps are smaller"

    def test_moment_and_state_accumulate(self):
        """m and v buffers carry over across steps and are not reset."""
        p = Parameter(np.array([0.0]))
        p.grad = np.array([1.0])
        opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8)

        opt.step()
        m_after_1 = opt.m[0].copy()

        p.grad = np.array([1.0])
        opt.step()
        m_after_2 = opt.m[0].copy()

        # m is an EWMA: m₂ = 0.9*m₁ + 0.1*g
        np.testing.assert_allclose(
            m_after_2,
            0.9 * m_after_1 + 0.1 * np.array([1.0]),
        )

    def test_zero_grad_preserves_moment_buffers(self):
        """zero_grad() clears gradient but leaves m and v intact."""
        p = Parameter(np.array([1.0]))
        p.grad = np.array([1.0])
        opt = Adam([p], lr=0.1, betas=(0.9, 0.999))

        opt.step()
        m_before = opt.m[0].copy()
        v_before = opt.v[0].copy()

        opt.zero_grad()

        np.testing.assert_array_equal(opt.m[0], m_before)
        np.testing.assert_array_equal(opt.v[0], v_before)

    def test_skip_parameter_with_none_grad(self):
        """Parameter with grad=None is untouched; its m/v stay at 0."""
        p1 = Parameter(np.array([1.0]))
        p1.grad = np.array([0.5])
        p2 = Parameter(np.array([2.0]))
        p2.grad = None
        opt = Adam([p1, p2], lr=0.1, betas=(0.9, 0.999))

        opt.step()

        np.testing.assert_allclose(p2.data, np.array([2.0]))
        np.testing.assert_allclose(opt.m[1], np.array([0.0]))
        np.testing.assert_allclose(opt.v[1], np.array([0.0]))

    def test_per_parameter_adaptive_scaling(self):
        """Different gradient magnitudes produce different adjusted_lr per dim."""
        p = Parameter(np.array([0.0, 0.0]))
        p.grad = np.array([1.0, 10.0])
        opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8)

        opt.step()

        # Both dims have same gradient direction (+), but dim1's larger
        # gradient produces larger v, so its effective step / g is smaller.
        # step1 / g1 and step2 / g2 should differ due to adaptive scaling.
        step_dim0 = p.data[0].item()
        step_dim1 = p.data[1].item()
        # Both gradients are positive → both steps are negative
        assert step_dim0 < 0.0
        assert step_dim1 < 0.0
        # But the per-gradient scaling differs: step/g for the larger dim
        # should be smaller than for the smaller dim.
        ratio_dim0 = abs(step_dim0) / 1.0
        ratio_dim1 = abs(step_dim1) / 10.0
        assert ratio_dim0 > ratio_dim1, (
            "Per-param scaling weakens updates for large-gradient dimensions"
        )


# =========================================================
# 9. Correctness — AdamW
# =========================================================


class TestAdamW:
    def test_first_step_with_weight_decay(self):
        """After Adam update, weight decay is applied: data -= lr * wd * data."""
        p = Parameter(np.array([1.0, 2.0]))
        p.grad = np.array([0.5, -0.3])
        opt = AdamW([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.5)

        opt.step()

        # First compute expected Adam step (same as TestAdam)
        m = np.array([0.05, -0.03])
        v = np.array([0.00025, 0.00009])
        m_hat = m / (1 - 0.9)
        v_hat = v / (1 - 0.999)
        adam_step = 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
        after_adam = np.array([1.0, 2.0]) - adam_step

        # Then weight decay
        wd_step = 0.1 * 0.5 * after_adam
        expected = after_adam - wd_step

        np.testing.assert_allclose(p.data, expected, atol=1e-6)

    def test_weight_decay_zero_equiv_adam(self):
        """weight_decay=0 makes AdamW identical to Adam."""
        p_w = Parameter(np.array([5.0]))
        p_w.grad = np.array([2.0])
        p_a = Parameter(np.array([5.0]))
        p_a.grad = np.array([2.0])
        opt_w = AdamW([p_w], lr=0.1, betas=(0.9, 0.999), weight_decay=0.0)
        opt_a = Adam([p_a], lr=0.1, betas=(0.9, 0.999))

        opt_w.step()
        opt_a.step()

        np.testing.assert_allclose(p_w.data, p_a.data)

    def test_weight_decay_pulls_toward_zero(self):
        """With zero gradients, weight decay alone shrinks parameters."""
        p_w = Parameter(np.array([1.0]))
        p_w.grad = np.array([0.0])
        p_a = Parameter(np.array([1.0]))
        p_a.grad = np.array([0.0])
        opt_w = AdamW([p_w], lr=0.1, betas=(0.9, 0.999), weight_decay=1.0)
        opt_a = Adam([p_a], lr=0.1, betas=(0.9, 0.999))

        opt_w.step()
        opt_a.step()

        # Adam: no gradient → no change → data stays at 1.0
        # AdamW: applies weight decay even with zero grad → data shrinks
        assert p_a.data.item() == 1.0
        assert p_w.data.item() < 1.0

    def test_skip_parameter_with_none_grad(self):
        """grad=None means no Adam update AND no weight decay for that param."""
        p1 = Parameter(np.array([1.0]))
        p1.grad = np.array([0.5])
        p2 = Parameter(np.array([2.0]))
        p2.grad = None
        opt = AdamW([p1, p2], lr=0.1, weight_decay=0.5)

        opt.step()

        np.testing.assert_allclose(p2.data, np.array([2.0]))
        np.testing.assert_allclose(opt.m[1], np.array([0.0]))
