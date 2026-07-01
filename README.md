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
- [x] **Training Utilities**
    - [x] Mini-batch iteration helpers (`DataLoader`-style batching)
    - [x] SGD optimizer (parameter update with learning rate)
- [x] **Applications**
    - [x] End-to-end: train a 2-layer MLP to convergence on a synthetic regression task

### Phase 3: Probability, Statistics & Loss Functions
Empirical distribution → theoretical distributions → information theory → bias-variance → MLE → loss functions → classification demos.
- [x] **Probability Distributions**
    - [x] Empirical distribution: histogram estimation from raw data
    - [x] Theoretical: Gaussian (Box-Muller sampling), Bernoulli, Categorical, Laplacian
    - [x] Visualizing PDFs/PMFs with varying parameters
- [x] **Information Theory & Bias-Variance**
    - [x] Entropy, KL Divergence, Cross-Entropy — computing all three on synthetic distributions
    - [x] Bias-Variance Decomposition: $\mathbb{E}[(y-\hat{f})^2] = \text{Bias}^2 + \text{Var} + \sigma^2$, connecting loss choice to model behavior
- [x] **MLE → Loss Functions**
    - [x] Implement on autograd: `BCELoss`, `CrossEntropyLoss` (fused Softmax+NLL), `L1Loss`, `HuberLoss`
- [x] **Applications**
    - [x] Binary classification on the moons dataset (with BCE)
    - [x] Multi-class classification on synthetic blobs (with CrossEntropy)
    - [x] Comparing MSE vs L1 vs Huber on regression with outliers

### Phase 4: Optimization — from Gradient Descent to Adaptive Methods
Gradient descent fails in predictable ways. Each technique is a targeted fix: diagnose the failure mode, implement the repair.
- [x] **Parameter Update Rules** — better step direction & per-parameter scaling
    - [x] SGD → Momentum → NAG: curriculum from vanilla steps to velocity-based updates, fighting ravine oscillation
    - [x] AdaGrad → RMSProp: per-parameter scaling for varying curvatures, from full accumulation to sliding window
    - [x] Adam → AdamW: momentum + adaptive scaling combined; then decoupled weight decay to fix a subtle L2 interaction bug
- [x] **Supporting Constraints** — what to do when the update alone isn't enough
    - [x] Learning rate schedules: step decay, cosine annealing, warmup — the same gradient, smaller or more timely steps
    - [x] Gradient clipping: norm clipping and value clipping — when a minibatch produces a pathological gradient
    - [x] Regularization: L1 (Laplace prior → sparsity), L2/Weight Decay (Gaussian prior → shrinkage)
- [x] **Applications**
    - [x] Toy loss surface comparison (Beale / Rosenbrock) — watch optimizer behavior match theory
    - [x] Train the same classifier with SGD, Momentum, Adam — compare convergence curves

### Phase 5: Transformer — Mini-GPT
Build a Decoder-Only Transformer (GPT-family) on PyTorch. First-principles minimal set — each component is the irreducible core; optimizations are added on top after the base works.

- [x] **Attention Core** — the operation that makes Transformers work
    - [x] `scaled_dot_product_attention`: $\text{softmax}(QK^T / \sqrt{d_k}) V$, with causal masking
    - [x] `MultiHeadAttention`: parallel heads with learned projections $W^Q, W^K, W^V, W^O$
- [x] **Standard Components** — Pre-Norm block, FeedForward, Positional Encoding
    - [x] `RMSNorm`: $\gamma \odot x / \sqrt{\text{mean}(x^2) + \epsilon}$ — the minimal norm that works
    - [x] `ReLU FeedForward`: $d_{model} \to d_{ff} \to d_{model}$, $d_{ff}=4d_{model}$
    - [x] `RoPE` (Rotary Positional Encoding): relative position via rotation matrix $R_\Theta^d$
    - [x] `GPTBlock`: RMSNorm → Attention → RMSNorm → FFN + Residual (Pre-Norm)
- [x] **GPT Architecture** — the language model skeleton
    - [x] Token Embedding: char → embedding lookup (vocab_size ~70 for char-level)
    - [x] `GPT`: $N$ stacked GPTBlocks + final RMSNorm + output projection
- [x] **Inference & Sampling** — making the model generate
    - [x] `generate()`: step-by-step autoregressive decoding
    - [x] Sampling: $\text{argmax}$ → Temperature → Top-k → Top-p (nucleus)
    - [x] [Extension] KV Cache — $O(n^2) \to O(n)$ optimization
- [x] **Training Pipeline** — learning language from scratch
    - [x] Char-level corpus (TinyShakespeare / text8 / zhihu-snippets), no tokenizer, vocab_size ~70
    - [x] Training loop + loss function (cross-entropy with label shift)
    - [x] Evaluation: perplexity monitoring + periodic sampling
- [ ] **Analysis Notebooks** — understanding what you built
    - [ ] Attention pattern + RoPE heatmap visualization
    - [ ] Temperature / Top-k / Top-p sampling comparison
    - [ ] In-Context Learning demo & completion examples

---

## 📚 Technical Stack
*   **Language**: Python 3.x
*   **Core Library**: NumPy (used for N-dimensional array storage and basic vectorized arithmetic).
*   **Deep Learning Framework**: PyTorch (Phase 5+, for tensor computation and neural network modules).
*   **Visualization**: Matplotlib (for convergence plots and decomposition results).

---

> *"What I cannot create, I do not understand." — Richard Feynman*
