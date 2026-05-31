"""Example error handling tests — verify exceptions with correct messages."""
import pytest

from core.linalg import Matrix


def test_some_error_case():
    m = Matrix.ones(3, 4)  # non-square
    with pytest.raises(ValueError, match="requires a square matrix"):
        m.some_method()
