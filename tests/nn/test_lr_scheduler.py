"""Tests for learning rate schedulers (Phase 4 — Supporting Constraints).

Covers LRScheduler base, StepDecay, CosineAnnealing, Warmup, and
WarmupCosine.  Each scheduler gets a focused correctness test at its
defining formula points; integration tests verify the scheduler-optimizer
interaction.
"""

import numpy as np
import pytest
from core.nn import (
    SGD,
    CosineAnnealing,
    LRScheduler,
    StepDecay,
    Warmup,
    WarmupCosine,
)
from core.nn.parameter import Parameter

# =========================================================
# Fixtures
# =========================================================


@pytest.fixture
def simple_optim():
    """A tiny SGD optimizer with exactly one parameter."""
    p = Parameter(np.array([1.0]))
    return SGD([p], lr=0.1)


# =========================================================
# 1. Base Class
# =========================================================


class TestLRSchedulerBase:
    def test_step_not_implemented(self, simple_optim):
        """Subclasses must implement _get_lr or step() raises."""

        class IncompleteScheduler(LRScheduler):
            pass

        sch = IncompleteScheduler(simple_optim)
        with pytest.raises(NotImplementedError):
            sch.step()


# =========================================================
# 2. StepDecay
# =========================================================


class TestStepDecay:
    def test_lr_drops_at_step_size_boundary(self, simple_optim):
        """At t = step_size, lr drops by factor gamma."""
        sch = StepDecay(simple_optim, step_size=5, gamma=0.1)
        for _ in range(4):
            sch.step()
        np.testing.assert_allclose(simple_optim.lr, 0.1)
        sch.step()  # t=5 → lr = 0.1 * 0.1 = 0.01
        np.testing.assert_allclose(simple_optim.lr, 0.01)

    def test_gamma_is_one_is_no_decay(self, simple_optim):
        """gamma=1 keeps lr constant (degenerate case)."""
        sch = StepDecay(simple_optim, step_size=2, gamma=1.0)
        for _ in range(10):
            sch.step()
        np.testing.assert_allclose(simple_optim.lr, 0.1)


# =========================================================
# 3. CosineAnnealing
# =========================================================


class TestCosineAnnealing:
    def test_at_midpoint(self, simple_optim):
        """t = T_max/2 → lr = halfway between base_lr and eta_min."""
        sch = CosineAnnealing(simple_optim, T_max=100, eta_min=0.0)
        for _ in range(50):
            sch.step()
        np.testing.assert_allclose(simple_optim.lr, 0.05)

    def test_at_T_max_and_beyond(self, simple_optim):
        """At t = T_max, lr = eta_min; beyond that, clamped."""
        sch = CosineAnnealing(simple_optim, T_max=50, eta_min=0.01)
        for _ in range(50):
            sch.step()
        np.testing.assert_allclose(simple_optim.lr, 0.01)
        for _ in range(50):
            sch.step()
        np.testing.assert_allclose(simple_optim.lr, 0.01)


# =========================================================
# 4. Warmup
# =========================================================


class TestWarmup:
    def test_linear_growth_and_plateau(self, simple_optim):
        """LR grows linearly to base_lr, then stays constant."""
        sch = Warmup(simple_optim, warmup_steps=100)
        for _ in range(50):
            sch.step()
        np.testing.assert_allclose(simple_optim.lr, 0.05)  # 50/100 of base
        for _ in range(50):
            sch.step()
        np.testing.assert_allclose(simple_optim.lr, 0.1)  # reached base_lr
        sch.step()
        np.testing.assert_allclose(simple_optim.lr, 0.1)  # stays at base_lr


# =========================================================
# 5. WarmupCosine — composite schedule
# =========================================================


class TestWarmupCosine:
    def test_warmup_then_cosine_transition(self, simple_optim):
        """Warmup phase is linear; after warmup, cosine phase smoothly decays."""
        sch = WarmupCosine(simple_optim, warmup_steps=50, total_steps=250, eta_min=0.0)
        for _ in range(25):
            sch.step()
        np.testing.assert_allclose(simple_optim.lr, 0.05)  # warmup midpoint
        for _ in range(25):
            sch.step()
        np.testing.assert_allclose(simple_optim.lr, 0.1)  # warmup complete
        for _ in range(100):
            sch.step()
        # Cosine phase: t=150 → progress = 100/200 = 0.5 → cos(0.5π)=0 → lr=0.05
        np.testing.assert_allclose(simple_optim.lr, 0.05)
        for _ in range(100):
            sch.step()
        np.testing.assert_allclose(simple_optim.lr, 0.0)  # end of cycle

    def test_beyond_total_steps_clamps(self, simple_optim):
        """Past total_steps, LR stays at eta_min."""
        sch = WarmupCosine(simple_optim, warmup_steps=50, total_steps=200, eta_min=0.01)
        for _ in range(300):
            sch.step()
        np.testing.assert_allclose(simple_optim.lr, 0.01)


# =========================================================
# 6. Integration: step ordering
# =========================================================


class TestSchedulerOptimizerInteraction:
    def test_step_ordering(self):
        """scheduler.step() after optimizer.step() changes the next step's LR."""
        p = Parameter(np.array([10.0]))
        opt = SGD([p], lr=1.0)
        sch = Warmup(opt, warmup_steps=5)

        # Step 1: optimizer uses lr=1.0 (before any scheduler call)
        p.grad = np.array([1.0])
        opt.step()
        np.testing.assert_allclose(p.data, np.array([9.0]))

        # Then scheduler updates lr for step 2
        sch.step()  # lr = 1.0 * 1/5 = 0.2
        p.grad = np.array([1.0])
        opt.step()
        np.testing.assert_allclose(p.data, np.array([8.8]))
