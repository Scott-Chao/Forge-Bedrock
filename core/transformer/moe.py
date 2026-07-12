"""
core/transformer/moe.py — Mixture of Experts (MoE).

Replaces the dense FeedForward network in a GPT block with a sparse
Mixture of Experts layer. Each token is routed to a subset of experts
(typically k=2 out of n_experts=8), reducing per-token FLOPs at the
cost of higher total parameter count.

Components
----------
- MoERouter: top-k softmax gating network
- MoEFFN: sparse expert dispatch + weighted combination + load balancing loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.transformer.transformer import FeedForward


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


class MoEFFN(nn.Module):
    """Sparse Mixture of Experts FeedForward Network.

    Replaces a standard dense FFN with a sparse MoE layer:

        1. Router selects top-k experts for each token
        2. Tokens are dispatched to their assigned experts
        3. Each expert (a ReLU FFN) processes its tokens
        4. Outputs are weighted-combined using routing weights

    An **auxiliary load-balancing loss** is computed during forward and
    returned alongside the output. This loss encourages the router to
    distribute tokens uniformly across experts.

    Parameters
    ----------
    d_model : int
        Input and output feature dimension.
    d_ff : int
        Hidden dimension of each expert's FeedForward network.
    n_experts : int, optional (default=8)
        Total number of experts.
    k : int, optional (default=2)
        Number of experts to activate per token (sparsity level).
        Must be <= n_experts.
    bias : bool, optional (default=True)
        Whether to use bias in the expert linear projections.
    aux_loss_coef : float, optional (default=1e-2)
        Coefficient scaling the auxiliary load-balancing loss.
        Typical range: 1e-3 ~ 1e-2.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        k: int = 2,
        bias: bool = True,
        aux_loss_coef: float = 1e-2,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.k = k
        self.aux_loss_coef = aux_loss_coef

        self.router = MoERouter(d_model, n_experts, k)

        self.experts = nn.ModuleList(
            [FeedForward(d_model, d_ff, bias) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: route -> dispatch -> expert forward -> weighted combine.

        Also computes the auxiliary load-balancing loss internally and
        stores it in ``self._aux_loss`` (accessible via ``self.aux_loss``).

        Parameters
        ----------
        x : (batch_size, seq_len, d_model)
            Input token representations (output of attention + residual).

        Returns
        -------
        output : (batch_size, seq_len, d_model)
            Weighted combination of expert outputs.
        """
        # ── Stage 1: Route ─────────────────────────────────────────────
        weights, indices = self.router(x)

        # ── Stage 2: Flatten batch & seq into a single token dimension ─
        x_flat = x.reshape(-1, self.d_model)
        weights_flat = weights.reshape(-1, self.k)
        indices_flat = indices.reshape(-1, self.k)

        # ── Stage 3: Dispatch and combine ─────────────────────────────
        output_flat = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            mask = indices_flat == e
            positions, which_k = torch.where(mask)
            if len(positions) == 0:
                continue
            expert_input = x_flat[positions]
            w = weights_flat[positions, which_k]
            expert_output = self.experts[e](expert_input)
            output_flat[positions] += w.unsqueeze(-1) * expert_output

        output = output_flat.reshape(x.shape)

        # ── Auxiliary Load-Balancing Loss ──────────────────────────────
        logits = self.router.gate(x)
        full_probs = F.softmax(logits, dim=-1)

        T = x.size(0) * x.size(1)
        counts = torch.bincount(
            indices_flat.reshape(-1), minlength=self.n_experts
        ).float()
        f_i = counts / (T * self.k)
        P_i = full_probs.mean(dim=(0, 1))
        self._aux_loss = self.n_experts * (f_i * P_i).sum()

        return output

    @property
    def aux_loss(self) -> torch.Tensor:
        """The auxiliary load-balancing loss from the last forward pass."""
        return self._aux_loss
