# Forge-Bedrock — Architecture

```
forge-bedrock/
├── core/linalg/           # Core linear algebra implementations
│   ├── matrix.py          # Matrix class (NumPy ndarray wrapper)
│   ├── decompositions.py  # LU, Cholesky, QR, SVD
│   ├── solvers.py         # Triangular solver, eigenvalue solver
│   ├── pca.py             # PCA dimensionality reduction
│   └── broadcast.py       # Custom broadcasting engine
├── core/autograd/         # Reverse-mode autograd engine (Phase 2)
│   ├── value.py           # Value class with dynamic DAG construction
│   ├── functional.py      # Activation & transcendental functions
│   └── viz.py             # Computation graph visualization
├── core/nn/               # Neural network modules (Phase 2–4)
│   ├── activation.py      # ReLU, Tanh, Sigmoid layer wrappers
│   ├── clip.py            # Gradient clipping (norm & value clipping)
│   ├── data.py            # DataLoader mini-batch iteration
│   ├── init.py            # Xavier/Glorot & He/Kaiming weight init
│   ├── linear.py          # Fully-connected (Linear) layer
│   ├── loss.py            # MSE, L1, Huber, BCE, CrossEntropy losses
│   ├── lr_scheduler.py    # LR schedules (StepDecay, CosineAnnealing, Warmup, WarmupCosine)
│   ├── module.py          # Module base class (parameter registration)
│   ├── optim.py           # 7 optimizers: SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW
│   ├── parameter.py       # Parameter class (trainable Value subclass)
│   ├── regularizer.py     # L1 & L2 regularization penalties
│   └── sequential.py      # Sequential container for layer pipelines
├── core/prob/             # Probability, statistics & information theory (Phase 3)
│   ├── empirical.py       # Empirical distribution & histogram estimation
│   ├── distributions.py   # Theoretical distributions (Uniform, Bernoulli, Categorical, Normal)
│   ├── info_theory.py     # Entropy, KL Divergence, Cross-Entropy
│   └── bias_variance.py   # Bias-variance decomposition simulation
├── core/transformer/      # Decoder-only Transformer (GPT) on PyTorch (Phase 5)
│   ├── transformer.py     # All Transformer-specific components in one file:
│   │                      #   scaled_dot_product_attention + MultiHeadAttention
│   │                      #   RoPE (precompute_freqs_cis, apply_rotary_emb, RotaryEmbedding)
│   │                      #   FeedForward (d_model → d_ff → d_model, d_ff = 4×d_model)
│   │                      #   GPTBlock (RMSNorm → Attn → RMSNorm → FFN, Pre-Norm residual)
│   │                      #   GPT (TokenEmbed → N×GPTBlock → RMSNorm → lm_head → logits)
│   ├── normalization.py   # RMSNorm: γ ⊙ x / √(mean(x²) + ε) — follows nn.modules.normalization
│   ├── data.py            # Char-level corpus loading + CharLevelDataset + DataLoader builder
│   ├── embedding.py       # CharTokenizer (vocab ~70) + TokenEmbedding lookup
│   ├── kv_cache.py        # O(n²) → O(n) decode: KVCache
│   └── sampling.py        # Token sampling: argmax → Temperature → Top-k → Top-p (nucleus)
├── apps/                  # Application Jupyter Notebooks
│   ├── image_compression.ipynb        # SVD-based low-rank image compression
│   ├── least_squares_regression.ipynb # Normal Equation vs Pseudoinverse
│   ├── mlp_regression.ipynb           # 2-layer MLP fitting sin(x) from scratch
│   ├── binary_classification_moons.ipynb    # Moons dataset with BCE loss
│   ├── multiclass_classification_blobs.ipynb # Synthetic blobs + CrossEntropy
│   ├── compare_losses_regression.ipynb      # MSE vs L1 vs Huber on outliers
│   ├── bias_variance_tradeoff.ipynb         # Bias-variance decomposition demo
│   ├── pdf_pmf_visualization.ipynb          # Distribution visualisation
│   ├── toy_loss_surface_comparison.ipynb     # Beale & Rosenbrock with 7 optimisers
│   ├── optimiser_comparison_classifier.ipynb # SGD vs Momentum vs Adam on moons
│   ├── train_gpt.ipynb                 # Full training pipeline on TinyShakespeare
│   ├── analysis_attention_rope.ipynb   # Attention pattern heatmaps + RoPE visualisation
│   └── analysis_sampling.ipynb         # Temperature, Top-k, Top-p comparison
│
├── tests/                 # pytest tests mirroring core/ structure
│   ├── linalg/            # Tests for Phase 1 (matrix, eigen, solver, SVD)
│   ├── autograd/          # Tests for Phase 2 (value, functional with finite differences)
│   ├── nn/                # Tests for Phase 2–4 (all layers, losses, optimisers, schedulers)
│   ├── prob/              # Tests for Phase 3 (distributions, info theory)
│   └── transformer/       # Tests for Phase 5 (transformer, normalization, sampling, generation)
│
├── assets/                # Static resources (images, etc.)
└── environment.yml        # conda environment
```
