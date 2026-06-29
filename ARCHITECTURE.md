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
├── apps/                  # Application Jupyter Notebooks
│   ├── image_compression.ipynb
│   ├── least_squares_regression.ipynb
│   ├── mlp_regression.ipynb
│   ├── binary_classification_moons.ipynb
│   ├── multiclass_classification_blobs.ipynb
│   ├── compare_losses_regression.ipynb
│   ├── bias_variance_tradeoff.ipynb
│   ├── pdf_pmf_visualization.ipynb
│   ├── toy_loss_surface_comparison.ipynb     # Phase 4: Beale & Rosenbrock optimiser trajectories
│   └── optimiser_comparison_classifier.ipynb  # Phase 4: SGD vs Momentum vs Adam on moons
├── tests/linalg/          # pytest tests for linear algebra
├── tests/autograd/        # pytest tests for autograd
├── tests/nn/              # pytest tests for neural network modules
├── tests/prob/            # pytest tests for probability & statistics
├── assets/                # Static resources (images, etc.)
└── environment.yml        # conda environment
```
