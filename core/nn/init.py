"""
core/nn/init.py — Weight initialization methods for neural network layers.

All functions operate in-place on a Parameter (modifying its .data)
and follow the PyTorch convention of using a trailing underscore
to indicate in-place mutation.

References
----------
- Glorot & Bengio (2010): Understanding the difficulty of training
  deep feedforward neural networks.
- He et al. (2015): Delving deep into rectifiers: surpassing
  human-level performance on ImageNet classification.
"""

import numpy as np
from core.nn.parameter import Parameter


def xavier_uniform_(param: Parameter, gain: float = 1.0) -> None:
    """Fill `param` with values from a uniform distribution.

    Uses the Xavier/Glorot recipe:

        a = gain * sqrt(6 / (n_in + n_out))
        param.data ~ Uniform(-a, a)

    where n_in is the number of input units and n_out is the number
    of output units for the weight matrix.

    This initialization is recommended for layers where the activation
    function is symmetric and approximately linear near 0 (e.g. tanh,
    sigmoid).

    Parameters
    ----------
    param : Parameter
        The parameter to initialize in-place.
    gain : float, default=1.0
        Scaling factor.  For tanh use 1.0; for sigmoid use ~1.0
        (though sigmoid is rarely used in hidden layers).
    """
    fan_out, fan_in = param.data.shape
    a = gain * np.sqrt(6 / (fan_in + fan_out))
    param.data = np.random.uniform(-a, a, size=param.data.shape)


def kaiming_uniform_(param: Parameter, gain: float = 1.0) -> None:
    """Fill `param` with values from a uniform distribution.

    Uses the He/Kaiming recipe (designed for ReLU activations).
    Following PyTorch's convention:

        a = gain * sqrt(3 / fan_in)
        param.data ~ Uniform(-a, a)

    For ReLU (gain=sqrt(2)), a = sqrt(6 / fan_in), matching
    the original He et al. proposal of Normal(0, sqrt(2/n_in))
    converted to a uniform scale.

    This initialization is recommended for layers followed by a ReLU
    activation, because it accounts for the fact that ReLU zeros out
    roughly half of the outputs.

    Parameters
    ----------
    param : Parameter
        The parameter to initialize in-place.
    gain : float, default=1.0
        Scaling factor.  Use sqrt(2) for ReLU (to compensate for the
        factor-of-2 variance loss), 1.0 for linear.
    """
    fan_in = param.data.shape[1]
    a = gain * np.sqrt(3 / fan_in)
    param.data = np.random.uniform(-a, a, size=param.data.shape)
