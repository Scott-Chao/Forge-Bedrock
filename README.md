# Forge-Bedrock

**Forge-Bedrock** is a long-term, bottom-up project dedicated to "reinventing the wheel" for the fundamental building blocks of Artificial Intelligence.

The philosophy of this project is to bridge the gap between abstract mathematical theorems (Calculus, Linear Algebra, Probability) and functional code. By implementing core algorithms from scratch, this repository serves as a personal laboratory for mastering the "bedrock" of AI—transforming black-box frameworks into transparent, intuitive logic.

---

## 🛠 Project Goals
*   **Deep Understanding**: Move beyond API calls to understand the mechanical necessity of every optimization and decomposition.
*   **Mathematical Rigor**: Translate formal proofs into robust, vectorized code.
*   **Modular Architecture**: Build a decoupled system where linear algebra solvers, autograd engines, and optimizers work in harmony.

---

## 🚀 Roadmap

### Phase 1: Linear Algebra
Implement the core routines that power data transformation and dimensionality reduction.

### Phase 2: Mini-Autograd (Computation Graph & Neural Networks)
Implement reverse-mode automatic differentiation (backpropagation) from scratch, inspired by Karpathy's micrograd. Starting from a scalar-level computational graph, then building a neural network library on top, with a regression demo to verify the full pipeline.

### Phase 3: Probability, Statistics & Loss Functions
Empirical distribution → theoretical distributions → information theory → bias-variance → MLE → loss functions → classification demos.

### Phase 4: Optimization — from Gradient Descent to Adaptive Methods
Gradient descent fails in predictable ways. Each technique is a targeted fix: diagnose the failure mode, implement the repair.

### Phase 5: Transformer — Mini-GPT
Build a Decoder-Only Transformer (GPT-family) on PyTorch. First-principles minimal set — each component is the irreducible core; optimizations are added on top after the base works.

[Full detailed roadmap → ROADMAP.md](ROADMAP.md)

[Architecture overview → ARCHITECTURE.md](ARCHITECTURE.md)

---

## 📚 Technical Stack
*   **Language**: Python 3.x
*   **Core Library**: NumPy (used for N-dimensional array storage and basic vectorized arithmetic).
*   **Deep Learning Framework**: PyTorch (Phase 5+, for tensor computation and neural network modules).
*   **Visualization**: Matplotlib (for convergence plots and decomposition results).

---

> *"What I cannot create, I do not understand." — Richard Feynman*
