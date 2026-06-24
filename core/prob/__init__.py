# flake8: noqa
from .bias_variance import (
    ConstantRegressor,
    PolynomialRegressor,
    bias_variance_decomposition,
    bootstrap_datasets,
    generate_data,
    polynomial_3,
    sinusoidal,
    step_function,
    train_on_bootstraps,
)
from .distributions import Bernoulli, Categorical, Gaussian, Laplacian
from .empirical import EmpiricalDistribution, Histogram
from .info_theory import (
    cross_entropy,
    entropy,
    entropy_from_counts,
    js_divergence,
    kl_divergence,
    mutual_information,
)

__all__ = [
    "Bernoulli",
    "Categorical",
    "ConstantRegressor",
    "EmpiricalDistribution",
    "Gaussian",
    "Histogram",
    "Laplacian",
    "PolynomialRegressor",
    "bias_variance_decomposition",
    "bootstrap_datasets",
    "cross_entropy",
    "entropy",
    "entropy_from_counts",
    "generate_data",
    "js_divergence",
    "kl_divergence",
    "mutual_information",
    "polynomial_3",
    "sinusoidal",
    "step_function",
    "train_on_bootstraps",
]
