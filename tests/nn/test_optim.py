import numpy as np
import pytest
from core.nn import AdaGrad, NAG, RMSProp, SGD, Momentum, Optimizer
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

    def test_momentum_zero_equiv_sgd(self):
        """momentum=0 degenerates to SGD."""
        p_m = Parameter(np.array([5.0]))
        p_m.grad = np.array([2.0])
        p_s = Parameter(np.array([5.0]))
        p_s.grad = np.array([2.0])
        opt_m = Momentum([p_m], lr=0.1, momentum=0.0)
        opt_s = SGD([p_s], lr=0.1)

        opt_m.step()
        opt_s.step()

        np.testing.assert_allclose(p_m.data, p_s.data)


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

    def test_larger_first_step_than_momentum(self):
        """NAG's first step is (beta+1) * lr * g, larger than Momentum's lr * g."""
        p_n = Parameter(np.array([0.0]))
        p_n.grad = np.array([1.0])
        p_m = Parameter(np.array([0.0]))
        p_m.grad = np.array([1.0])
        opt_n = NAG([p_n], lr=0.1, momentum=0.9)
        opt_m = Momentum([p_m], lr=0.1, momentum=0.9)

        opt_n.step()
        opt_m.step()

        # NAG: -(0.9*0.1 + 0.1) = -0.19
        # Momentum: -0.1
        assert abs(p_n.data.item()) > abs(p_m.data.item())

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
    @pytest.mark.parametrize("opt_cls", [Momentum, NAG])
    def test_decreases_loss_on_quadratic(self, opt_cls):
        """Both optimizers make progress on f(x) = ½ x²."""
        p = Parameter(np.array([5.0]))
        opt = opt_cls([p], lr=0.1, momentum=0.8)
        losses = []
        for _ in range(50):
            p.grad = p.data.copy()  # ∇½x² = x
            opt.step()
            losses.append(0.5 * p.data.item() ** 2)
        assert losses[-1] < losses[0] * 0.01

    def test_momentum_accelerates_constant_gradient(self):
        """With a consistent gradient, momentum accumulates velocity and moves farther."""
        p_m = Parameter(np.array([0.0]))
        p_s = Parameter(np.array([0.0]))
        opt_m = Momentum([p_m], lr=0.1, momentum=0.9)
        opt_s = SGD([p_s], lr=0.1)
        for _ in range(10):
            # Both see the same constant gradient (∇f = 1)
            p_m.grad = np.array([1.0])
            p_s.grad = np.array([1.0])
            opt_m.step()
            opt_s.step()
        # Momentum's velocity accumulates, so it moves ~4× farther than SGD
        assert abs(p_m.data.item()) > abs(p_s.data.item())


# =========================================================
# 5. Inheritance & Error Handling
# =========================================================


class TestOptimizerBase:
    def test_momentum_is_optimizer(self):
        assert isinstance(Momentum([], lr=0.01), Optimizer)

    def test_nag_is_optimizer(self):
        assert isinstance(NAG([], lr=0.01), Optimizer)

    def test_step_not_implemented(self):
        opt = Optimizer([], lr=0.01)
        with pytest.raises(NotImplementedError):
            opt.step()

    def test_adagrad_is_optimizer(self):
        assert isinstance(AdaGrad([], lr=0.01), Optimizer)

    def test_rmsprop_is_optimizer(self):
        assert isinstance(RMSProp([], lr=0.01), Optimizer)


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

    def test_converges_on_one_dimensional_quadratic(self):
        """AdaGrad decreases loss on f(x) = x²."""
        p = Parameter(np.array([10.0]))
        opt = AdaGrad([p], lr=1.0, eps=1e-8)
        losses = []
        for _ in range(100):
            p.grad = np.array([2.0 * p.data.item()])  # ∇x² = 2x
            opt.step()
            losses.append(p.data.item() ** 2)
        assert losses[-1] < losses[0] * 0.01


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

    def test_cache_approaches_steady_state(self):
        """With constant gradient, EWMA cache asymptotically approaches g²."""
        p = Parameter(np.array([0.0]))
        p.grad = np.array([5.0])
        opt = RMSProp([p], lr=0.1, beta=0.9, eps=1e-8)

        for _ in range(200):
            p.grad = np.array([5.0])
            opt.step()

        # Steady state: cache → g² = 25
        np.testing.assert_allclose(opt.caches[0], 25.0, atol=0.5)

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

    def test_beta_zero_equiv_adagrad_first_step(self):
        """beta=0 makes first step identical to AdaGrad (both cache = g²)."""
        p_r = Parameter(np.array([5.0]))
        p_r.grad = np.array([2.0])
        p_a = Parameter(np.array([5.0]))
        p_a.grad = np.array([2.0])
        opt_r = RMSProp([p_r], lr=0.5, beta=0.0, eps=1e-8)
        opt_a = AdaGrad([p_a], lr=0.5, eps=1e-8)

        opt_r.step()
        opt_a.step()

        np.testing.assert_allclose(p_r.data, p_a.data)

    def test_converges_on_one_dimensional_quadratic(self):
        """RMSProp decreases loss on f(x) = x²."""
        p = Parameter(np.array([10.0]))
        opt = RMSProp([p], lr=1.0, beta=0.9, eps=1e-8)
        losses = []
        for _ in range(100):
            p.grad = np.array([2.0 * p.data.item()])
            opt.step()
            losses.append(p.data.item() ** 2)
        assert losses[-1] < losses[0] * 0.01
