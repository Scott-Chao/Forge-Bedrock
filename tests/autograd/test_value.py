import pytest
import numpy as np

from core.autograd import Value


# =========================================================
# Helper
# =========================================================

def _numerical_grad(f, x, h=1e-6):
    """Central-difference approximation of df/dx at x."""
    return (f(x + h) - f(x - h)) / (2 * h)


def _topo_sort(node):
    """Manual topological sort for a tree (sufficient for simple expressions)."""
    nodes = []

    def collect(n):
        for child in n._children:
            collect(child)
        nodes.append(n)

    collect(node)
    return list(reversed(nodes))


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
        assert v._backward is not None

    def test_with_children_and_op(self):
        a = Value(2.0)
        b = Value(3.0)
        c = Value(5.0, children=(a, b), op="+")
        assert c.data == 5.0
        assert c._children == {a, b}
        assert c._op == "+"

    def test_repr(self):
        v = Value(4.0)
        assert "data=4.0" in repr(v)
        assert "grad=0.0" in repr(v)


# =========================================================
# 2. DAG Construction — forward correctness & graph edges
# =========================================================

class TestDAGConstruction:
    """Verify that operators build a proper computation graph and produce
    correct forward results (compared against NumPy)."""

    @pytest.mark.parametrize("a, b", [(2.0, 3.0), (0.0, 0.0), (-1.5, 4.2)])
    def test_add(self, a, b):
        va, vb = Value(a), Value(b)
        out = va + vb
        assert out.data == a + b
        assert out._op == "+"
        assert out._children == {va, vb}

    @pytest.mark.parametrize("a, b", [(2.0, 3.0), (0.0, 5.0), (-1.0, -1.0)])
    def test_mul(self, a, b):
        va, vb = Value(a), Value(b)
        out = va * vb
        assert out.data == a * b
        assert out._op == "*"
        assert out._children == {va, vb}

    @pytest.mark.parametrize("base, exp", [(2.0, 3), (4.0, 0.5), (-3.0, 2)])
    def test_pow(self, base, exp):
        v = Value(base)
        out = v ** exp
        assert out.data == base ** exp
        assert out._op == "**"
        assert out._children == {v}

    @pytest.mark.parametrize("a, b", [(5.0, 2.0), (0.0, 3.0), (1.0, -4.0)])
    def test_sub(self, a, b):
        va, vb = Value(a), Value(b)
        out = va - vb
        assert out.data == a - b
        # subtraction is implemented as a + (-b), so the top-level op is "+"
        assert out._op == "+"

    @pytest.mark.parametrize("a, b", [(6.0, 2.0), (1.0, 3.0), (-4.0, 2.0)])
    def test_truediv(self, a, b):
        va, vb = Value(a), Value(b)
        out = va / vb
        assert out.data == a / b

    @pytest.mark.parametrize("a", [3.0, 0.0, -2.5])
    def test_neg(self, a):
        out = -Value(a)
        assert out.data == -a

    # --- Reflected operators ---

    @pytest.mark.parametrize("a, b", [(1.0, 2.0), (3.0, 0.5)])
    def test_radd(self, a, b):
        out = a + Value(b)
        assert out.data == a + b

    @pytest.mark.parametrize("a, b", [(2.0, 3.0), (0.5, 4.0)])
    def test_rmul(self, a, b):
        out = a * Value(b)
        assert out.data == a * b

    @pytest.mark.parametrize("a, b", [(5.0, 2.0), (0.0, 3.0)])
    def test_rsub(self, a, b):
        out = a - Value(b)
        assert out.data == a - b

    @pytest.mark.parametrize("a, b", [(6.0, 2.0), (1.0, 4.0)])
    def test_rtruediv(self, a, b):
        out = a / Value(b)
        assert out.data == a / b

    # --- Chained expressions ---

    def test_three_term_expression(self):
        """y = a * b + c"""
        a, b, c = Value(2.0), Value(3.0), Value(5.0)
        y = a * b + c
        np.testing.assert_allclose(y.data, 2 * 3 + 5)

        # y's children are: c and the (a*b) intermediate node
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
        """Reusing a Value in multiple operations builds a proper DAG (not a tree)."""
        x = Value(3.0)
        y = x * x + x
        np.testing.assert_allclose(y.data, 3 * 3 + 3)


# =========================================================
# 3. Backward closure correctness (manual trigger)
# =========================================================

class TestBackwardClosures:
    """Verify that each _backward closure computes correct local gradients
    by manually calling _backward() on each node in topological order.

    Since the full backward() method is a placeholder, we use _topo_sort
    to simulate what backward() will eventually do.
    """

    # --- Single-operation backward ---

    def test_add_backward(self):
        a, b = Value(2.0), Value(3.0)
        out = a + b
        out.grad = 1.0
        out._backward()
        np.testing.assert_allclose(a.grad, 1.0)
        np.testing.assert_allclose(b.grad, 1.0)

    def test_add_gradient_accumulation(self):
        a, b = Value(2.0), Value(3.0)
        out = a + b
        out.grad = 1.0
        out._backward()
        out._backward()
        np.testing.assert_allclose(a.grad, 2.0)
        np.testing.assert_allclose(b.grad, 2.0)

    def test_mul_backward(self):
        a, b = Value(2.0), Value(3.0)
        out = a * b
        out.grad = 1.0
        out._backward()
        np.testing.assert_allclose(a.grad, b.data)
        np.testing.assert_allclose(b.grad, a.data)

    def test_pow_backward(self):
        base = Value(2.0)
        exp = 3
        out = base ** exp
        out.grad = 1.0
        out._backward()
        np.testing.assert_allclose(base.grad, 3 * 2 ** 2)

    def test_neg_backward(self):
        a = Value(2.0)
        out = -a
        out.grad = 1.0
        out._backward()
        np.testing.assert_allclose(a.grad, -1.0)

    # --- Compound-operation backward (multi-step chain) ---

    def test_sub_backward(self):
        """a - b  =>  a + (-b)  =>  two-step backward."""
        a, b = Value(5.0), Value(2.0)
        out = a - b
        out.grad = 1.0

        for node in _topo_sort(out):
            node._backward()

        np.testing.assert_allclose(a.grad, 1.0)
        np.testing.assert_allclose(b.grad, -1.0)

    def test_div_backward(self):
        """a / b  =>  a * b**(-1)  =>  multi-step backward."""
        a, b = Value(6.0), Value(2.0)
        out = a / b
        out.grad = 1.0

        for node in _topo_sort(out):
            node._backward()

        np.testing.assert_allclose(a.grad, 1.0 / 2.0)
        np.testing.assert_allclose(b.grad, -6.0 / 4.0)

    # --- Finite-difference verification ---

    @pytest.mark.parametrize(
        "expr_name, a_val, b_val, c_val, forward_fn",
        [
            ("a + b", 2.0, 3.0, None, lambda a, b: a + b),
            ("a * b", 2.0, 3.0, None, lambda a, b: a * b),
            ("a ** 2", 2.0, 3.0, None, lambda a, b: a ** 2),
            ("(a + b) * c", 1.0, 2.0, 4.0, lambda a, b, c: (a + b) * c),
            ("a * b + c", 2.0, 3.0, 5.0, lambda a, b, c: a * b + c),
            ("(a - b) / c", 5.0, 1.0, 2.0, lambda a, b, c: (a - b) / c),
        ],
    )
    def test_against_finite_difference(self, expr_name, a_val, b_val, c_val, forward_fn):
        """Compare gradients from backward closures against finite-difference."""
        a, b, c = Value(a_val), Value(b_val), Value(c_val) if c_val is not None else None

        if c is not None:
            out = forward_fn(a, b, c)
        else:
            out = forward_fn(a, b)

        out.grad = 1.0

        # run backward via topological sort
        for node in _topo_sort(out):
            node._backward()

        # Finite-difference check for each leaf
        def f_a(x):
            va, vb, vc = Value(x), Value(b_val), Value(c_val) if c_val is not None else None
            return forward_fn(va, vb, vc).data if vc is not None else forward_fn(va, vb).data

        def f_b(x):
            va, vb, vc = Value(a_val), Value(x), Value(c_val) if c_val is not None else None
            return forward_fn(va, vb, vc).data if vc is not None else forward_fn(va, vb).data

        np.testing.assert_allclose(a.grad, _numerical_grad(f_a, a_val), atol=1e-5)
        np.testing.assert_allclose(b.grad, _numerical_grad(f_b, b_val), atol=1e-5)

        if c is not None:
            def f_c(x):
                return forward_fn(Value(a_val), Value(b_val), Value(x)).data
            np.testing.assert_allclose(c.grad, _numerical_grad(f_c, c_val), atol=1e-5)


# =========================================================
# 4. Edge Cases
# =========================================================

class TestEdgeCases:
    def test_operation_with_zero(self):
        a = Value(0.0)
        b = Value(5.0)
        assert (a + b).data == 5.0
        assert (a * b).data == 0.0

    def test_operation_with_negative(self):
        a = Value(-2.0)
        b = Value(3.0)
        assert (a * b).data == -6.0
        assert (a + b).data == 1.0

    def test_scalar_on_left(self):
        assert (3 + Value(2)).data == 5
        assert (3 - Value(2)).data == 1
        assert (3 * Value(2)).data == 6
        assert (6 / Value(2)).data == 3

    def test_zero_division(self):
        with pytest.raises(ZeroDivisionError):
            _ = Value(1.0) / Value(0.0)


# =========================================================
# 5. Error Handling
# =========================================================

class TestErrorHandling:
    def test_backward_not_implemented(self):
        v = Value(3.0)
        assert v.backward() is NotImplemented

    def test_ensure_value_preserves_value(self):
        v = Value(5.0)
        assert v._ensure_value(v) is v
