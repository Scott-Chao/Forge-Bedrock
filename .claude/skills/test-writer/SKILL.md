---
name: test-writer
description: >
  Write detailed pytest unit tests for any module in the Forge-Bedrock project.
  Trigger whenever the user says something like "给 XXX 写测试", "写单元测试",
  "补测试", "add tests", "write unit tests", "cover XXX with tests", or asks
  you to generate or improve test coverage for any module in this project.
  This includes asking about missing tests, edge cases not covered, or
  suggesting improvements to existing test files. This skill is specific to
  the Forge-Bedrock project's code style and testing conventions.
---

# Forge Test Writer

Write pytest unit tests for any module in the Forge-Bedrock project (linalg, autograd, probability, optimization, etc.).

## Workflow

1. **Read the source file** — understand what the module implements (classes, methods, parameters, return types, edge cases)
2. **Check the test file** — if one exists, read it to understand what's already covered; if not, prepare to create it
3. **Review reference tests** — skim one of the existing test files in `tests/` to match the established style. For new phases, also read the relevant example files below
4. **Map source to test file** — `core/linalg/<module>.py` → `tests/linalg/test_<module>.py`. For new phases, infer from the source path (e.g. `core/autograd/` → `tests/autograd/`). Merge into existing files when multiple source modules are tested together
5. **Write the tests** — follow the structure below, covering correctness, edge cases, error handling, and optionally benchmarks
6. **Verify** — run `pytest tests/<path>/test_<module>.py -x -q`

## Test Structure

```
import pytest
import numpy as np
from core.<module> import <classes>

# =========================================================
# Fixtures (Generate test data)
# =========================================================

# =========================================================
# 1. Correctness Tests
# =========================================================

# =========================================================
# 2. Edge Cases
# =========================================================

# =========================================================
# 3. Error Handling
# =========================================================

# =========================================================
# (Optional) Performance Benchmarks
# =========================================================
```

## Patterns

Read the relevant example file for the pattern you need:

| Pattern | File | When to use |
|---------|------|-------------|
| Fixtures | `examples/fixtures.py` | Providing reusable test data — return `(numpy_array, CustomObject)` pairs |
| Correctness | `examples/correctness.py` | Verifying output against numpy as ground truth |
| Parametrize | `examples/parametrize.py` | Same test logic across multiple inputs (shapes, operator pairs) |
| Error handling | `examples/error_handling.py` | Testing that code raises correct exceptions |
| Benchmarks | `examples/benchmarks.py` | Performance comparison with numpy (optional, small sizes) |

**Assertions guide:**
- `np.testing.assert_allclose(actual, expected, atol=1e-10)` — default for floats
- `np.testing.assert_array_equal(actual, expected)` — exact matches only
- `assert` — shapes, booleans, isinstance
- Adjust tolerance for iterative methods (QR, Power Iteration → `rtol=1e-5`)

**Edge cases to consider:**
- Minimal sizes (1x1, empty if applicable)
- Non-square shapes (tall and wide)
- Singular/degenerate matrices
- Identity and zero matrices
- Scalar operations (both `matrix op scalar` and `scalar op matrix`)
- In-place operations with incompatible shapes

## Adapting to Future Phases

Each phase has a different ground-truth reference:

- **Phase 1 (linalg)**: Compare against `numpy.linalg`
- **Phase 2 (Autograd)**: Compare gradients against finite-difference: `(f(x+h) - f(x-h)) / 2h`
- **Phase 3 (Probability/Stats)**: Check analytical expectations vs empirical samples; verify KL ≥ 0, cross-entropy minimized at identical distributions
- **Phase 4 (Optimization)**: Verify loss decreases monotonically; check convergence on convex problems; validate regularization penalties

When starting tests for a new phase, read one of the Phase 1 test files for structural conventions, then adapt the verification strategy using the table above.

## Design Principles

- **Test what matters**: correctness (matches reference?), edge cases (handles weird input?), error handling (fails informatively?)
- **Don't over-test internals**: test the public interface, not private details
- **Use numpy as oracle**: compare against numpy's reference output — makes tests self-documenting
- **Group related assertions**: it's fine to verify one logical operation (e.g. reconstruction + orthogonality + shape) in one test
- **Descriptive test names**: `test_pinv_fundamental_properties` > `test_pinv_1`
