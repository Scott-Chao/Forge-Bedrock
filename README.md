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
- [ ] **Numerical Stability & Performance**
    - [x] **Adaptive Relative Tolerance** based on machine epsilon and matrix norms.
    - [x] **Stable SVD** via One-Sided Jacobi rotations (avoiding $A^T A$ precision loss).
    - [x] **Hessenberg Reduction** for $O(n^2)$ QR iteration acceleration.
    - [ ] **Shifted QR Algorithm** with Wilkinson shifts and deflation logic.
    - [ ] **In-place Householder Storage** and implicit $Q$ matrix construction.
- [ ] **Applications**
    - [ ] Image compression via Low-Rank Approximation (SVD).
    - [ ] Least Squares Regression using the Normal Equation.

### Phase 2: Mini-Autograd (Computation Graph)
- [ ] Design the `Tensor` class with `data` and `grad` attributes.
- [ ] Implement a Dynamic Computational Graph using topological sort.
- [ ] Define `forward` and `backward` passes for basic operators (+, -, *, /, exp, log).

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
