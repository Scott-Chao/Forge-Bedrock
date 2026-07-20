# Roadmap & TODO List

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
    - [x] SVD: $A = U\Sigma V^T$ implementation from scratch.
    - [x] PCA: Dimensionality reduction using SVD/Covariance.
    - [x] Moore-Penrose Pseudoinverse: Solving overdetermined systems via $A^+ = (A^T A)^{-1} A^T$.
- [x] **Numerical Stability & Performance**
    - [x] Adaptive Relative Tolerance based on machine epsilon and matrix norms.
    - [x] Stable SVD via One-Sided Jacobi rotations (avoiding $A^T A$ precision loss).
    - [x] Hessenberg Reduction for $O(n^2)$ QR iteration acceleration.
    - [x] Shifted QR Algorithm with Wilkinson shifts and deflation logic.
    - [x] In-place Householder Storage and implicit $Q$ matrix construction.
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
    - [x] [Extension] Repetition Penalty — discourage token loops via logit penalisation
- [x] **Training Pipeline** — learning language from scratch
    - [x] Char-level corpus (TinyShakespeare / text8 / zhihu-snippets), no tokenizer, vocab_size ~70
    - [x] Training loop + loss function (cross-entropy with label shift)
    - [x] Evaluation: perplexity monitoring + periodic sampling
- [x] **Analysis Notebooks** — understanding what you built
    - [x] Attention pattern + RoPE heatmap visualization
    - [x] Temperature / Top-k / Top-p sampling comparison

### Phase 6: Transformer — BPE, GQA, MoE
Phase 5's minimal GPT gets three modular upgrades: subword tokenization (BPE), grouped-query attention (GQA), and mixture of experts (MoE).
- [x] **BPE Tokenizer**
    - [x] Pre-tokenization: `re.findall(r'\w+', text)` whitespace split
    - [x] Core training: frequency-based merge loop, pairs within pre-tokenized words only
    - [x] Inference: encode (text → IDs via merge-rank lookup), decode (IDs → text via table)
- [x] **GQA (Grouped Query Attention)**
    - [x] MHA extension: `n_kv_heads` → smaller K/V projections, `repeat_interleave` broadcast, KV cache shape `(batch, n_kv_heads, ...)`
    - [x] Model penetration: thread `n_kv_heads` through `GPTBlock` → `GPT` → `generate()`
- [x] **MoE (Mixture of Experts)**
    - [x] Router: `W_gate ∈ R^{n_experts × d_model}`, top-2 softmax (`k=2`, `n_experts=8`)
    - [x] Sparse dispatch: token scatter → per-expert ReLU FFN → weighted combine
    - [x] Load balancing: auxiliary importance loss (coefficient ~1e-2)
    - [x] Integration: replace `GPTBlock.ffn` via `GPTConfig(moe=True, ...)`
    - [x] Training: 1 run on TinyShakespeare, loss curve vs Phase 5 baseline
- [x] **Analysis Notebooks**
    - [x] MoE routing: expert activation heatmaps, load uniformity, routing entropy

### Phase 7: Convolutional Neural Networks
Images demand different inductive biases than sequences: locality and translation equivariance. Conv2d (im2col) → Pooling → BatchNorm → ResNet → U-Net → ViT — the convolutional path and the transformer path through vision.
- [x] **Core Layers** — vision primitives
    - [x] `Conv2d` forward via im2col: `F.unfold` + `matmul`, supporting kernel_size, stride, padding, dilation
    - [x] `MaxPool2d` / `AvgPool2d`: sliding-window reduction parameterized by kernel_size, stride, padding
    - [x] `BatchNorm2d`: normalize over (N, H, W), learnable γ/β; training batch stats → inference running mean/var
    - [x] `ConvTranspose2d`: forward via transpose of the im2col matrix (matmul + fold), supporting kernel_size, stride, padding, dilation, output_padding
- [x] **ResNet** — deep residual learning
    - [x] `BasicBlock` / `BottleneckBlock`: skip connection `F(x) + x`, 1×1 conv for dimension matching
    - [x] `ResNet`: configurable depth (ResNet-18/34/50/101/152), stem + 4 stages, global avg pooling + linear head
- [ ] **U-Net** — encoder-decoder bridge
    - [ ] `DownBlock`: ResBlock → MaxPool downsample (one encoder stage)
    - [ ] `UpBlock`: ConvTranspose2d upsample → skip concat → ResBlock (one decoder stage)
    - [ ] `UNet`: encoder (DownBlock × 4) → bottleneck → decoder (UpBlock × 4)
- [ ] **ViT** — patches + transformer as the convolution alternative
    - [ ] `PatchEmbed`: image → (patch × patch) flatten → Linear projection + [CLS] token + position embedding
    - [ ] `ViT`: PatchEmbed → GPTBlock × N (no causal mask) → [CLS] → classification head
- [ ] **Demo Notebooks**
    - [ ] `resnet_cifar10.ipynb`: CIFAR-10 training with cosine LR, test accuracy
    - [ ] `unet_demo.ipynb`: shape verification + feature map comparison with/without skip connections
    - [ ] `vit_demo.ipynb`: ViT training on CIFAR-10 + comparison with ResNet
