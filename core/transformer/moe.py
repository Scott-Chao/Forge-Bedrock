"""
core/transformer/moe.py — Mixture of Experts (MoE).

Replaces the dense FeedForward network in a GPT block with a sparse
Mixture of Experts layer. Each token is routed to a subset of experts
(typically k=2 out of n_experts=8), reducing per-token FLOPs at the
cost of higher total parameter count.

Components
----------
- MoERouter: top-k softmax gating network
- MoEFFN (future): sparse expert dispatch + weighted combination
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    """Top-k softmax router for Mixture of Experts.

    For each input token, computes expert logits via a learned gate
    matrix, selects the top-k experts, and returns normalised routing
    weights (softmax over the selected k logits).

    Mathematically:

        logits = x @ W_gate^T          ∈ ℝ^{... × n_experts}
        indices = topk(logits, k)      ∈ ℕ^{... × k}
        weights = softmax(logits[indices])  ∈ ℝ^{... × k}

    Parameters
    ----------
    d_model : int
        Input feature dimension.
    n_experts : int, optional (default=8)
        Total number of experts.
    k : int, optional (default=2)
        Number of experts to route each token to (sparsity level).
    """

    def __init__(self, d_model: int, n_experts: int = 8, k: int = 2):
        super().__init__()

        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route each token to its top-k experts.

        Parameters
        ----------
        x : (batch_size, seq_len, d_model)
            Input token representations.

        Returns
        -------
        (weights, indices) : tuple[torch.Tensor, torch.Tensor]

            weights : (batch_size, seq_len, k)
                Softmax-normalised weights for each selected expert.
                ``weights[b, t, i]`` = weight of the i-th selected
                expert for token (b, t). Sums to 1 over i.

            indices : (batch_size, seq_len, k), dtype=torch.long
                Indices of the selected experts in ``[0, n_experts)``.
        """
        logits = self.gate(x)
        weights, indices = torch.topk(logits, self.k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return weights, indices
