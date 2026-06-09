# Forge-Bedrock

**Forge-Bedrock** is a long-term, bottom-up project dedicated to "reinventing the wheel" for the fundamental building blocks of Artificial Intelligence.

The philosophy of this project is to bridge the gap between abstract mathematical theorems (Calculus, Linear Algebra, Probability) and functional code. By implementing core algorithms from scratch, this repository serves as a personal laboratory for mastering the "bedrock" of AI—transforming black-box frameworks into transparent, intuitive logic.

---

## 🛠 Project Goals
*   **Deep Understanding**: Move beyond API calls to understand the mechanical necessity of every optimization and decomposition.
*   **Mathematical Rigor**: Translate formal proofs into robust, vectorized code.
*   **Modular Architecture**: Build a decoupled system where linear algebra solvers, autograd engines, and optimizers work in harmony.

---

## 🚀 Roadmap & TODO List

### Phase 1: Linear Algebra
Implement the core routines that power data transformation and dimensionality reduction.
- [x] **Basic Matrix Operations**
    - [x] High-performance Matrix Multiplication (Tiling/Block-based logic).
    - [x] Custom Broadcasting engine for tensor alignment.
- [x] **Systems of Equations**
    - [x] Gaussian Elimination with partial pivoting.
    - [x] LU Decomposition.
    - [x] Cholesky Decomposition (for symmetric positive-definite matrices).
- [x] **Eigenvalues & Iterative Methods**
    - [x] Power Iteration (finding the dominant eigenvalue).
    - [x] QR Algorithm for finding all eigenvalues.
- [x] **Advanced Matrix Decompositions**
    - [x] **Singular Value Decomposition (SVD)**: $A = U\Sigma V^T$ implementation from scratch.
    - [x] **Principal Component Analysis (PCA)**: Dimensionality reduction using SVD/Covariance.
    - [x] **Moore-Penrose Pseudoinverse**: Solving overdetermined systems via $A^+ = (A^T A)^{-1} A^T$.
- [x] **Numerical Stability & Performance**
    - [x] **Adaptive Relative Tolerance** based on machine epsilon and matrix norms.
    - [x] **Stable SVD** via One-Sided Jacobi rotations (avoiding $A^T A$ precision loss).
    - [x] **Hessenberg Reduction** for $O(n^2)$ QR iteration acceleration.
    - [x] **Shifted QR Algorithm** with Wilkinson shifts and deflation logic.
    - [x] **In-place Householder Storage** and implicit $Q$ matrix construction.
- [x] **Applications**
    - [x] Least Squares Regression using the Normal Equation.
    - [x] Image compression via Low-Rank Approximation (SVD).

### Phase 2: Mini-Autograd (Computation Graph & Neural Networks)
Implement reverse-mode automatic differentiation (backpropagation) from scratch, inspired by Karpathy's micrograd. Starting from a scalar-level computational graph, then building a neural network library on top, with a regression demo to verify the full pipeline.
- [x] **Autograd Engine (`Value`)**
    - [x] Dynamic DAG construction via Python operator overloading (`__add__`, `__mul__`, etc.)
    - [x] Topological sort for correct backward propagation order
    - [x] Reverse-mode automatic differentiation (`.backward()`)
    - [x] Gradient accumulation across multiple backward calls
    - [x] Computation graph visualization (graphviz or textual DAG rendering)
- [x] **Core Operations with Backward Rules**
    - [x] Arithmetic: `+`, `-`, `*`, `/`, `**` (power), `neg`
    - [x] Activations: `relu`, `sigmoid`, `tanh`
    - [x] Transcendental: `exp`, `log`, `sqrt`
    - [x] (Stretch) `softmax` with stable log-sum-exp trick, `log_softmax`
    - [x] Numerical gradient verification via finite differences for every operation
- [x] **Neural Network Modules (`nn`)**
    - [x] `Parameter` class (a `Value` subclass marking trainable parameters)
    - [x] `Linear` layer (fully connected: $y = xW^T + b$) with proper shape handling
    - [x] Activation wrappers: `ReLU`, `Tanh`, `Sigmoid`
    - [x] `Sequential` container for composing multi-layer pipelines
    - [x] `Module` base class: `parameters()` iterator, `zero_grad()`, train/eval mode
    - [x] Weight initialization: Xavier/Glorot uniform, He/Kaiming uniform
- [x] **Loss (minimal, just enough for demos)**
    - [x] `MSELoss` — Mean Squared Error for regression tasks
- [ ] **Training Utilities**
    - [ ] Mini-batch iteration helpers (`DataLoader`-style batching)
    - [ ] Metric tracking: running loss
- [x] **Verification & Correctness**
    - [x] Unit tests for every operation's forward and backward pass
    - [x] Numerical gradient checking against finite differences
    - [ ] End-to-end: train a 2-layer MLP to convergence on a synthetic regression task
- [ ] **Applications**
    - [ ] Polynomial curve fitting via a small MLP
    - [ ] Regression on a synthetic dataset (e.g., noisy sine wave)

### Phase 3: Probability, Statistics & Loss Functions
- [ ] Implementation of core distributions (Gaussian, Bernoulli) via sampling.
- [ ] Information Theory metrics: Entropy, Cross-Entropy, and KL Divergence.
- [ ] Maximum Likelihood Estimation (MLE) simulations.

### Phase 4: Optimization & Training Logic
- [ ] Stochastic Gradient Descent (SGD) and Momentum.
- [ ] Adaptive methods: AdaGrad, RMSProp, and Adam.
- [ ] Regularization techniques (L1/L2 Weight Decay) from a mathematical constraint perspective.

---

## 📚 Technical Stack
*   **Language**: Python 3.x
*   **Core Library**: NumPy (used for N-dimensional array storage and basic vectorized arithmetic).
*   **Visualization**: Matplotlib (for convergence plots and decomposition results).

---

> *"What I cannot create, I do not understand." — Richard Feynman*
