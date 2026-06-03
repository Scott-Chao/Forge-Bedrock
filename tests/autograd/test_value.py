import pytest
import numpy as np

from core.autograd import Value


def _numerical_grad(f, x, h=1e-6):
    """Central-difference approximation of df/dx at x."""
    return (f(x + h) - f(x - h)) / (2 * h)


# =========================================================
# 1. Initialisation & Properties
# =========================================================


class TestInit:
    def test_basic_init(self):
        v = Value(3.0)
        assert v.data == 3.0
        assert v.grad == 0.0
        assert v._children == set()
        assert v._op == ""

    def test_with_children_and_op(self):
        a, b = Value(2.0), Value(3.0)
        c = Value(5.0, children=(a, b), op="+")
        assert c._children == {a, b}
        assert c._op == "+"

    def test_repr(self):
        assert "data=4.0" in repr(Value(4.0))
        assert "grad=0.0" in repr(Value(4.0))

    def test_ensure_value_preserves_value(self):
        v = Value(5.0)
        assert v._ensure_value(v) is v


# =========================================================
# 2. DAG Construction — forward correctness & graph edges
# =========================================================


class TestDAGConstruction:
    """Verify operators build a computation graph with correct forward results."""

    def test_binary_ops(self):
        a, b = Value(2.0), Value(3.0)

        out_add = a + b
        assert out_add.data == 5.0 and out_add._op == "+"
        assert out_add._children == {a, b}

        out_mul = a * b
        assert out_mul.data == 6.0 and out_mul._op == "*"
        assert out_mul._children == {a, b}

        out_sub = a - b
        assert out_sub.data == -1.0

        out_div = a / b
        assert out_div.data == 2.0 / 3.0

    def test_pow(self):
        v = Value(2.0)
        out = v**3
        assert out.data == 8.0
        assert out._op == "**"
        assert out._children == {v}

    def test_neg(self):
        out = -Value(2.0)
        assert out.data == -2.0

    def test_chained_expression(self):
        """y = a * b + c"""
        a, b, c = Value(2.0), Value(3.0), Value(5.0)
        y = a * b + c
        assert y.data == 2 * 3 + 5

        children = y._children
        assert c in children
        t_node = [n for n in children if n is not c][0]
        assert t_node._op == "*"
        assert t_node._children == {a, b}

    def test_complex_expression(self):
        """y = (a + b) * (c - d)"""
        a, b, c, d = Value(1.0), Value(2.0), Value(5.0), Value(3.0)
        y = (a + b) * (c - d)
        np.testing.assert_allclose(y.data, (1 + 2) * (5 - 3))

    def test_expression_reuse(self):
        """Reusing a Value builds a proper DAG (not a tree)."""
        x = Value(3.0)
        assert (x * x + x).data == 12.0

    def test_reflected_ops(self):
        assert (3 + Value(2)).data == 5
        assert (3 - Value(2)).data == 1
        assert (3 * Value(2)).data == 6
        assert (6 / Value(2)).data == 3


# =========================================================
# 3. Backward pass — end-to-end gradient verification
# =========================================================


class TestBackward:
    def test_add(self):
        a, b = Value(2.0), Value(3.0)
        (a + b).backward()
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_mul(self):
        a, b = Value(2.0), Value(3.0)
        (a * b).backward()
        assert a.grad == 3.0
        assert b.grad == 2.0

    def test_pow(self):
        v = Value(2.0)
        (v**3).backward()
        assert v.grad == 12.0  # 3 * 2^2

    def test_neg(self):
        v = Value(2.0)
        (-v).backward()
        assert v.grad == -1.0

    def test_sub(self):
        a, b = Value(5.0), Value(2.0)
        (a - b).backward()
        assert a.grad == 1.0
        assert b.grad == -1.0

    def test_div(self):
        a, b = Value(6.0), Value(2.0)
        (a / b).backward()
        assert a.grad == 1.0 / 2.0
        assert b.grad == -6.0 / 4.0

    def test_variable_reuse(self):
        """y = x * x + x  =>  dy/dx = 2x + 1 = 7"""
        x = Value(3.0)
        (x * x + x).backward()
        assert x.grad == 7.0

    @pytest.mark.parametrize(
        "expr_name, forward_fn, a_val, b_val, c_val",
        [
            ("a + b", lambda a, b: a + b, 2.0, 3.0, None),
            ("a * b", lambda a, b: a * b, 2.0, 3.0, None),
            ("(a+b)*c", lambda a, b, c: (a + b) * c, 1.0, 2.0, 4.0),
            ("(a-b)/c", lambda a, b, c: (a - b) / c, 5.0, 1.0, 2.0),
        ],
    )
    def test_against_finite_difference(
        self, expr_name, forward_fn, a_val, b_val, c_val
    ):
        """Compare gradients against finite-difference approximation."""
        a, b = Value(a_val), Value(b_val)
        c = Value(c_val) if c_val is not None else None

        out = forward_fn(a, b, c) if c is not None else forward_fn(a, b)
        out.backward()

        def f_a(x):
            va, vb = Value(x), Value(b_val)
            vc = Value(c_val) if c_val is not None else None
            return (
                forward_fn(va, vb, vc).data
                if vc is not None
                else forward_fn(va, vb).data
            )

        def f_b(x):
            va, vb = Value(a_val), Value(x)
            vc = Value(c_val) if c_val is not None else None
            return (
                forward_fn(va, vb, vc).data
                if vc is not None
                else forward_fn(va, vb).data
            )

        np.testing.assert_allclose(a.grad, _numerical_grad(f_a, a_val), atol=1e-5)
        np.testing.assert_allclose(b.grad, _numerical_grad(f_b, b_val), atol=1e-5)

        if c is not None:

            def f_c(x):
                return forward_fn(Value(a_val), Value(b_val), Value(x)).data

            np.testing.assert_allclose(c.grad, _numerical_grad(f_c, c_val), atol=1e-5)


# =========================================================
# 4. Edge Cases & Error Handling
# =========================================================


class TestEdgeCases:
    def test_operation_with_zero(self):
        a, b = Value(0.0), Value(5.0)
        assert (a + b).data == 5.0
        assert (a * b).data == 0.0

    def test_operation_with_negative(self):
        a, b = Value(-2.0), Value(3.0)
        assert (a * b).data == -6.0
        assert (a + b).data == 1.0

    def test_zero_division(self):
        with pytest.raises(ZeroDivisionError):
            _ = Value(1.0) / Value(0.0)

    def test_gradient_confluence(self):
        """Single backward — gradients from multiple paths sum at shared leaf."""
        a = Value(2.0)
        (a + a).backward()
        assert a.grad == 2.0  # da/da from both branches = 1 + 1


# =========================================================
# 5. Multiple backward calls — gradient accumulation
# =========================================================


class TestGradientAccumulation:
    """Calling backward() multiple times accumulates gradients on leaves."""

    def test_single_op_accumulates(self):
        """c = a * b  →  two backward() calls double leaf grads."""
        a, b = Value(2.0), Value(3.0)
        c = a * b

        c.backward()
        assert a.grad == 3.0
        assert b.grad == 2.0

        c.backward()
        assert a.grad == 6.0  # 3 + 3
        assert b.grad == 4.0  # 2 + 2

    def test_chain_accumulates_correctly(self):
        """e = (a * b) + d  →  multi-node chain avoids stale intermediate grads."""
        a, b = Value(2.0), Value(3.0)
        x = a * b
        d = Value(5.0)
        e = x + d

        # First backward
        e.backward()
        np.testing.assert_allclose(a.grad, 3.0)
        np.testing.assert_allclose(b.grad, 2.0)
        np.testing.assert_allclose(d.grad, 1.0)

        # Second backward — must NOT leak stale x.grad
        e.backward()
        np.testing.assert_allclose(a.grad, 6.0)  # 3 + 3
        np.testing.assert_allclose(b.grad, 4.0)  # 2 + 2
        np.testing.assert_allclose(d.grad, 2.0)  # 1 + 1

    def test_three_backward_calls(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b

        c.backward()
        c.backward()
        c.backward()

        assert a.grad == 9.0  # 3 * 3
        assert b.grad == 6.0  # 3 * 2

    def test_intermediate_grad_reset(self):
        """Intermediate node grads are zeroed before each backward pass."""
        a, b = Value(2.0), Value(3.0)
        x = a * b
        d = Value(5.0)
        e = x + d

        e.backward()
        assert e.grad == 1.0
        assert x.grad == 1.0

        e.backward()
        assert e.grad == 1.0, "root is reset each time"
        assert x.grad == 1.0, "intermediate must not accumulate across calls"

    def test_reused_variable_accumulates(self):
        """y = x * x + x  →  two backward() calls double leaf grads."""
        x = Value(3.0)
        y = x * x + x

        y.backward()
        np.testing.assert_allclose(x.grad, 7.0)  # 2*3 + 1

        y.backward()
        np.testing.assert_allclose(x.grad, 14.0)  # 7 + 7

    def test_deep_chain_accumulation(self):
        """y = ((a + b) * c) / d  →  deeper chain."""
        a, b, c, d = Value(2.0), Value(3.0), Value(4.0), Value(2.0)
        y = ((a + b) * c) / d

        y.backward()
        np.testing.assert_allclose(a.grad, 2.0)  # c/d = 4/2 = 2
        np.testing.assert_allclose(b.grad, 2.0)

        y.backward()
        np.testing.assert_allclose(a.grad, 4.0)  # 2 + 2
        np.testing.assert_allclose(b.grad, 4.0)

    def test_accumulation_matches_finite_difference(self):
        """After N backward calls, accumulated grads match N×finite-diff."""
        a_val, b_val, c_val = 2.0, 3.0, 5.0

        a, b, c = Value(a_val), Value(b_val), Value(c_val)
        y = a * b + c

        for _ in range(3):
            y.backward()

        assert a.grad == b_val * 3  # 3 calls × b
        assert b.grad == a_val * 3  # 3 calls × a
        assert c.grad == 1.0 * 3  # 3 calls × 1
