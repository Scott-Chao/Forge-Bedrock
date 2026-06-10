"""
core/nn/optim.py — Optimizers for neural network training.

Implementations follow the PyTorch optimizer convention.
"""


class SGD:
    """Stochastic Gradient Descent optimizer.

    Updates each parameter along the negative gradient direction:

        param.data -= lr * param.grad

    Parameters
    ----------
    parameters : iterable of Parameter
        The model parameters to optimize.  Typically obtained from
        model.parameters().
    lr : float, default=0.001
        Learning rate (step size).
    """

    def __init__(self, parameters, lr: float = 0.001):
        self.params = list(parameters)
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0
