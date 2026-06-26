import numpy as np
import pytest
from core.autograd import (
    Value,
    exp,
    log,
    log_softmax,
    relu,
    sigmoid,
    softmax,
    sqrt,
    tanh,
)


def _numerical_grad(f, x, h=1e-6):
    """Scalar finite-difference gradient (x is a number)."""
    return (f(x + h) - f(x - h)) / (2 * h)


def _numerical_grad_array(f, x0, h=1e-6):
    """Element-wise finite-difference gradient for an array input."""
    grad = np.zeros_like(x0)
    for idx in np.ndindex(x0.shape):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[idx] += h
        x_minus[idx] -= h
        grad[idx] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


# (name, fn, x, expected_data, expected_grad)
FORWARD_BACKWARD_CASES = [
    ("exp(1)", exp, 1.0, np.exp(1), np.exp(1)),
    ("exp(0)", exp, 0.0, 1.0, 1.0),
    ("exp(-2)", exp, -2.0, np.exp(-2), np.exp(-2)),
    ("log(3)", log, 3.0, np.log(3), 1 / 3),
    ("log(1)", log, 1.0, 0.0, 1.0),
    ("log(0.5)", log, 0.5, np.log(0.5), 2.0),
    ("sqrt(4)", sqrt, 4.0, 2.0, 0.25),
    ("sqrt(0)", sqrt, 0.0, 0.0, np.inf),
    ("sqrt(2)", sqrt, 2.0, np.sqrt(2), 0.5 / np.sqrt(2)),
    ("relu(2)", relu, 2.0, 2.0, 1.0),
    ("relu(-1)", relu, -1.0, 0.0, 0.0),
    ("relu(0)", relu, 0.0, 0.0, 0.0),
    ("sigmoid(0)", sigmoid, 0.0, 0.5, 0.25),
    (
        "sigmoid(2)",
        sigmoid,
        2.0,
        1 / (1 + np.exp(-2)),
        1 / (1 + np.exp(-2)) * (1 - 1 / (1 + np.exp(-2))),
    ),
    ("tanh(1)", tanh, 1.0, np.tanh(1), 1 - np.tanh(1) ** 2),
    ("tanh(0)", tanh, 0.0, 0.0, 1.0),
]

# =========================================================
# 1. Forward correctness & DAG construction
# =========================================================


class TestForward:
    @pytest.mark.parametrize(
        "name, fn, x, expected_data, _",
        FORWARD_BACKWARD_CASES,
        ids=[c[0] for c in FORWARD_BACKWARD_CASES],
    )
    def test_forward(self, name, fn, x, expected_data, _):
        v = Value(x)
        out = fn(v)
        np.testing.assert_allclose(out.data, expected_data)
        assert out._children == {v}

    def test_chain_exp_log(self):
        x = Value(3.0)
        np.testing.assert_allclose(exp(log(x)).data, 3.0, atol=1e-12)


# =========================================================
# 2. Backward pass — analytical gradient verification
# =========================================================


class TestBackward:
    @pytest.mark.parametrize(
        "name, fn, x, _, expected_grad",
        FORWARD_BACKWARD_CASES,
        ids=[c[0] for c in FORWARD_BACKWARD_CASES],
    )
    def test_backward(self, name, fn, x, _, expected_grad):
        v = Value(x)
        fn(v).backward()
        if np.isinf(expected_grad):
            assert np.isinf(v.grad)
        else:
            np.testing.assert_allclose(v.grad, expected_grad, atol=1e-12)

    def test_chain_rule_exp(self):
        x = Value(1.5)
        exp(2 * x).backward()
        np.testing.assert_allclose(x.grad, 2 * np.exp(3.0))

    def test_chain_rule_log(self):
        x = Value(3.0)
        log(x**2).backward()
        np.testing.assert_allclose(x.grad, 2.0 / 3.0)


# =========================================================
# 3. Finite difference verification
# =========================================================


class TestAgainstFiniteDifference:
    CASES = [
        ("exp(1)", exp, np.exp, 1.0),
        ("exp(-1)", exp, np.exp, -1.0),
        ("log(2)", log, np.log, 2.0),
        ("sqrt(3)", sqrt, np.sqrt, 3.0),
        ("relu(1)", relu, lambda z: np.maximum(0, z), 1.0),
        ("relu(-1)", relu, lambda z: np.maximum(0, z), -1.0),
        ("sigmoid(0)", sigmoid, lambda z: 1 / (1 + np.exp(-z)), 0.0),
        ("tanh(0.5)", tanh, np.tanh, 0.5),
    ]

    @pytest.mark.parametrize(
        "name, fn, np_fn, x",
        CASES,
        ids=[c[0] for c in CASES],
    )
    def test_finite_diff(self, name, fn, np_fn, x):
        v = Value(x)
        fn(v).backward()
        np.testing.assert_allclose(v.grad, _numerical_grad(np_fn, x), atol=1e-5)

    CHAINED_CASES = [
        ("exp(2x)", lambda v: exp(2 * v), lambda z: np.exp(2 * z), 0.5),
        ("log(x^2)", lambda v: log(v**2), lambda z: np.log(z**2), 2.0),
        ("sqrt(x^3)", lambda v: sqrt(v**3), lambda z: np.sqrt(z**3), 4.0),
        (
            "sigmoid(2x)",
            lambda v: sigmoid(2 * v),
            lambda z: 1 / (1 + np.exp(-2 * z)),
            1.0,
        ),
        ("tanh(x^2)", lambda v: tanh(v**2), lambda z: np.tanh(z**2), 0.5),
    ]

    @pytest.mark.parametrize(
        "name, val_fn, np_fn, x",
        CHAINED_CASES,
        ids=[c[0] for c in CHAINED_CASES],
    )
    def test_chained_finite_diff(self, name, val_fn, np_fn, x):
        v = Value(x)
        val_fn(v).backward()
        np.testing.assert_allclose(v.grad, _numerical_grad(np_fn, x), atol=1e-5)


# =========================================================
# 4. Edge cases
# =========================================================


class TestEdgeCases:
    def test_log_at_zero(self):
        v = Value(0.0)
        out = log(v)
        assert np.isneginf(out.data)
        with pytest.raises(ZeroDivisionError):
            out.backward()

    def test_exp_negative_large(self):
        v = Value(-100.0)
        out = exp(v)
        assert out.data < 1e-43
        out.backward()
        assert v.grad < 1e-43

    def test_exp_log_inverse(self):
        x = Value(3.0)
        exp(log(x)).backward()
        np.testing.assert_allclose(x.grad, 1.0, atol=1e-12)


# =========================================================
# 5. Softmax & Log-Softmax
# =========================================================


def _stable_softmax_ref(x: np.ndarray) -> np.ndarray:
    """NumPy reference softmax (stable), last-axis softmax."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=-1, keepdims=True)


LOGITS = np.array([1.0, 2.0, 3.0])
UPSTREAM = np.array([0.5, -0.2, 0.3])

BATCH_LOGITS = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 5.0]])
BATCH_UPSTREAM = np.array([[0.5, -0.2, 0.3], [1.0, -1.0, 2.0]])


class TestSoftmaxForward:
    def test_correct_values_1d(self):
        p = softmax(Value(LOGITS))
        np.testing.assert_allclose(np.sum(p.data), 1.0)
        np.testing.assert_allclose(p.data, _stable_softmax_ref(LOGITS), atol=1e-12)

    def test_correct_values_2d(self):
        p = softmax(Value(BATCH_LOGITS))
        np.testing.assert_allclose(np.sum(p.data, axis=1), [1.0, 1.0])
        np.testing.assert_allclose(
            p.data, _stable_softmax_ref(BATCH_LOGITS), atol=1e-12
        )

    def test_dag(self):
        x = Value(LOGITS)
        p = softmax(x)
        assert p._op == "Softmax"
        assert p._children == {x}

    @pytest.mark.parametrize(
        "x, desc",
        [
            (np.array([2.0, 2.0, 2.0]), "uniform"),
            (np.array([5.0]), "single"),
            (np.array([800.0, 801.0, 802.0]), "large positive"),
            (np.array([[0.0, 0.0], [0.0, 0.0]]), "uniform 2d"),
            (np.array([[800.0, 801.0], [0.0, 5.0]]), "mixed large 2d"),
        ],
    )
    def test_edge_cases(self, x, desc):
        p = softmax(Value(x)).data
        np.testing.assert_allclose(
            np.sum(p), x.size / x.shape[-1]
        )  # each row sums to 1
        assert not np.any(np.isnan(p))
        assert not np.any(np.isinf(p))


class TestLogSoftmaxForward:
    def test_correct_values_1d(self):
        lp = log_softmax(Value(LOGITS))
        expected = np.log(_stable_softmax_ref(LOGITS))
        np.testing.assert_allclose(lp.data, expected, atol=1e-12)

    def test_correct_values_2d(self):
        lp = log_softmax(Value(BATCH_LOGITS))
        expected = np.log(_stable_softmax_ref(BATCH_LOGITS))
        np.testing.assert_allclose(lp.data, expected, atol=1e-12)

    def test_dag(self):
        x = Value(LOGITS)
        lp = log_softmax(x)
        assert lp._op == "LogSoftmax"
        assert lp._children == {x}


class TestSoftmaxBackward:
    def test_analytical_1d(self):
        x = Value(LOGITS.copy())
        p = softmax(x)
        p.grad = UPSTREAM.copy()
        p._backward()
        p_data = p.data
        expected = p_data * (UPSTREAM - (p_data * UPSTREAM).sum(axis=-1, keepdims=True))
        np.testing.assert_allclose(x.grad, expected, atol=1e-12)

    def test_analytical_2d(self):
        x = Value(BATCH_LOGITS.copy())
        p = softmax(x)
        p.grad = BATCH_UPSTREAM.copy()
        p._backward()
        p_data = p.data
        expected = p_data * (
            BATCH_UPSTREAM - (p_data * BATCH_UPSTREAM).sum(axis=-1, keepdims=True)
        )
        np.testing.assert_allclose(x.grad, expected, atol=1e-12)

    def test_accumulation(self):
        x = Value(LOGITS.copy())
        p = softmax(x)
        p.grad = UPSTREAM.copy()
        x.grad = np.ones_like(LOGITS)  # pre-existing grad
        p._backward()
        first = x.grad.copy()
        p._backward()
        np.testing.assert_allclose(x.grad, first + (first - 1.0), atol=1e-12)

    def test_finite_diff_2d(self):
        rng = np.random.default_rng(99)
        x_data = rng.normal(0, 1, (3, 4))
        x = Value(x_data)
        loss = softmax(x).sum()
        loss.backward()

        def f(arr):
            return softmax(Value(arr)).data.sum()

        grad_numerical = _numerical_grad_array(f, x_data)
        np.testing.assert_allclose(x.grad, grad_numerical, atol=1e-5)


class TestLogSoftmaxBackward:
    def test_analytical_1d(self):
        x = Value(LOGITS.copy())
        lp = log_softmax(x)
        lp.grad = UPSTREAM.copy()
        lp._backward()
        p = np.exp(lp.data)
        expected = UPSTREAM - UPSTREAM.sum(axis=-1, keepdims=True) * p
        np.testing.assert_allclose(x.grad, expected, atol=1e-12)

    def test_analytical_2d(self):
        x = Value(BATCH_LOGITS.copy())
        lp = log_softmax(x)
        lp.grad = BATCH_UPSTREAM.copy()
        lp._backward()
        p = np.exp(lp.data)
        expected = BATCH_UPSTREAM - BATCH_UPSTREAM.sum(axis=-1, keepdims=True) * p
        np.testing.assert_allclose(x.grad, expected, atol=1e-12)

    def test_accumulation(self):
        x = Value(LOGITS.copy())
        lp = log_softmax(x)
        lp.grad = UPSTREAM.copy()
        x.grad = np.ones_like(LOGITS) * 100.0  # pre-existing grad
        lp._backward()
        first = x.grad.copy()
        lp._backward()
        np.testing.assert_allclose(x.grad, first + (first - 100.0), atol=1e-12)

    def test_finite_diff_2d(self):
        rng = np.random.default_rng(98)
        x_data = rng.normal(0, 1, (2, 5))
        x = Value(x_data)
        loss = log_softmax(x).sum()
        loss.backward()

        def f(arr):
            return log_softmax(Value(arr)).data.sum()

        grad_numerical = _numerical_grad_array(f, x_data)
        np.testing.assert_allclose(x.grad, grad_numerical, atol=1e-5)
