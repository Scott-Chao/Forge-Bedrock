"""
core/transformer/attention.py — Attention mechanisms for Transformer.

This module implements the core attention operation that makes Transformers
work: the Scaled Dot-Product Attention described in "Attention Is All You
Need" (Vaswani et al., 2017).

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

With causal masking (for decoder-only autoregressive models):

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Shapes
    ------
    query:  (batch_size, ..., seq_len, d_k)
    key:    (batch_size, ..., seq_len, d_k)
    value:  (batch_size, ..., seq_len, d_v)   — d_v is often == d_k
    mask:   (seq_len, seq_len) or broadcastable to (..., seq_len, seq_len)

    Returns
    -------
    output: (batch_size, ..., seq_len, d_v)
    """
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    return attention_weights @ value


def _create_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Create a causal (upper-triangular) attention mask.

    Returns a boolean mask of shape (seq_len, seq_len) where:
        mask[i, j] = True   if i >= j  (allowed: attend to current & past)
        mask[i, j] = False  if i < j   (forbidden: attend to future)
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
    return ~mask
