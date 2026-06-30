"""
core/transformer/normalization.py — Normalization layers for Transformer.

RMSNorm (Root Mean Square Normalization) is a simplified alternative to
Layer Normalization, introduced by Zhang & Sennrich (2019). It removes the
mean-centering step from LayerNorm and only normalizes by the root-mean-square
of the activations. This is the standard normalization used in modern LLMs
(Llama, Mistral, Gemma, etc.).

    RMSNorm(x) = x / RMS(x) * gamma

    where RMS(x) = sqrt(mean(x^2) + epsilon)
"""

from __future__ import annotations

import torch


class RMSNorm(torch.nn.Module):
    """Root Mean Square Normalization.

    Normalizes the input along the **last dimension** (the feature dimension),
    independently per token position.

        out = x / sqrt(mean(x^2) + eps) * gamma

    Parameters
    ----------
    d_model : int
        Feature dimension. Normalisation is applied along this axis.
    eps : float, optional (default=1e-6)
        Small constant added inside the sqrt for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.gamma = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.

        Parameters
        ----------
        x : (..., d_model)
            Input tensor. The last dimension must match d_model.

        Returns
        -------
        out : (..., d_model)
            Normalized output with the same shape as input.
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.gamma
