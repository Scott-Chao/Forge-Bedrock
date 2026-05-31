# Forge-Bedrock — CLAUDE.md

## Philosophy

This is an AI student's "reinvent the wheel" learning project. The core goal is to **implement the mathematical foundations of AI from scratch**, deeply understanding the principles behind each algorithm rather than staying at the API-calling level.

- This is a **learning project** — the AI's role is to guide and discuss, not to write code on behalf of the user
- When problems arise, prioritize explaining the principles and outlining the approach, letting the user decide how to implement
- Refer to README.md for the project's overall goals and roadmap

## Project Structure

```
forge-bedrock/
├── core/linalg/          # Core linear algebra implementations
│   ├── matrix.py         # Matrix class (NumPy ndarray wrapper)
│   ├── decompositions.py # LU, Cholesky, QR, SVD
│   ├── solvers.py        # Triangular solver, eigenvalue solver
│   ├── pca.py            # PCA dimensionality reduction
│   └── broadcast.py      # Custom broadcasting engine
├── apps/                 # Application Jupyter Notebooks
│   ├── image_compression.ipynb
│   └── least_squares_regression.ipynb
├── tests/linalg/         # pytest tests
└── environment.yml       # conda environment
```

## Code Style

- Use **ruff** for linting (`ruff check .`), keeping code consistent
- Class names in `PascalCase` (`Matrix`, `LU`, `SVD`), methods/variables in `snake_case`
- Comments should explain WHY, not WHAT (well-named code is self-documenting)
- Dependencies: use NumPy for underlying array storage only; do not rely on scipy or other scientific computing libraries

## Testing

- Use **pytest**; test files live under `tests/`
- Tests should be added after completing each feature
- Run tests: `pytest tests/`
