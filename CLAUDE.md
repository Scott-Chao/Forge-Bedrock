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
├── core/autograd/        # Reverse-mode autograd engine (Phase 2)
│   ├── __init__.py
│   ├── value.py          # Value class with dynamic DAG construction
│   ├── functional.py     # Activation & transcendental functions
│   └── viz.py            # Computation graph visualization
├── core/nn/              # Neural network modules (Phase 2)
│   ├── __init__.py
│   ├── activation.py     # ReLU, Tanh, Sigmoid layer wrappers
│   ├── data.py           # DataLoader mini-batch iteration
│   ├── init.py           # Xavier/Glorot & He/Kaiming weight init
│   ├── linear.py         # Fully-connected (Linear) layer
│   ├── loss.py           # MSELoss for regression tasks
│   ├── module.py         # Module base class (parameter registration)
│   ├── optim.py          # SGD optimizer
│   ├── parameter.py      # Parameter class (trainable Value subclass)
│   └── sequential.py     # Sequential container for layer pipelines
├── core/prob/            # Probability & statistics (Phase 3)
│   ├── __init__.py
│   └── empirical.py      # Empirical distribution & histogram estimation
├── apps/                 # Application Jupyter Notebooks
│   ├── image_compression.ipynb
│   ├── least_squares_regression.ipynb
│   └── mlp_regression.ipynb
├── tests/linalg/         # pytest tests for linear algebra
├── tests/autograd/       # pytest tests for autograd
├── tests/nn/             # pytest tests for neural network modules
├── tests/prob/           # pytest tests for probability & statistics
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
