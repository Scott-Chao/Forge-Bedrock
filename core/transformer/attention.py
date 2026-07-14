"""
core/transformer/attention.py — Scaled dot-product attention and Multi-Head Attention.

Provides the core attention mechanism and the Multi-Head / Grouped-Query
Attention module that sits inside each GPTBlock.

Components
----------
- ``scaled_dot_product_attention`` — the basic Q·K^T / √d_k → softmax → @V
- ``_create_causal_mask`` — boolean upper-triangular mask for autoregressive decoding
- ``MultiHeadAttention`` — multi-head + grouped-query attention with optional KV cache
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from core.transformer.rope import RotaryEmbedding


# ═══════════════════════════════════════════════════════════════════════
# Scaled Dot-Product Attention
# ═══════════════════════════════════════════════════════════════════════


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


def _create_causal_mask(
    seq_len: int, device: torch.device | None = None
) -> torch.Tensor:
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


# ═══════════════════════════════════════════════════════════════════════
# Multi-Head Attention
# ═══════════════════════════════════════════════════════════════════════


class MultiHeadAttention(torch.nn.Module):
    """Multi-Head / Grouped-Query Attention.

    Supports both standard Multi-Head Attention (MHA) and Grouped-Query
    Attention (GQA, Ainslie et al., 2023).

    **MHA mode** (default, ``n_kv_heads=None``):
        Each query head has its own K, V projection. The KV cache stores
        one (K, V) pair per head.

            MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
            where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    **GQA mode** (``n_kv_heads < n_heads``):
        Multiple query heads **share** a single K, V pair. The KV cache
        stores only ``n_kv_heads`` pairs — a direct memory saving.

            n_groups = n_heads // n_kv_heads
            K = K.repeat_interleave(n_groups, dim=1)   # expand at runtime
            V = V.repeat_interleave(n_groups, dim=1)   # expand at runtime

    Important
    ---------
    In a decoder-only Transformer (GPT), Q = K = V = x (the same input).
    This is called **self-attention**: the tokens attending to themselves.
    The projections still exist; the input simply passes through three
    separate linear layers to produce Q, K, V.

    Parameters
    ----------
    d_model : int
        Input and output feature dimension.
    n_heads : int
        Number of parallel attention heads. Must divide d_model evenly.
    n_kv_heads : int | None, optional (default=None)
        Number of key/value heads (for GQA). If None, defaults to n_heads
        (standard MHA). Must divide n_heads evenly.
    dropout : float, optional (default=0.0)
        Dropout rate applied to attention weights. Not strictly necessary
        for a minimal model, but standard practice.
    bias : bool, optional (default=True)
        Whether to include bias terms in the linear projections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        if n_kv_heads is None:
            n_kv_heads = n_heads  # default: standard MHA

        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads  # dimension per head

        kv_dim = self.d_k * n_kv_heads
        self.w_q = torch.nn.Linear(d_model, d_model, bias=bias)
        self.w_k = torch.nn.Linear(d_model, kv_dim, bias=bias)
        self.w_v = torch.nn.Linear(d_model, kv_dim, bias=bias)
        self.w_o = torch.nn.Linear(d_model, d_model, bias=bias)

        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: RotaryEmbedding | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for multi-head attention, with optional KV cache.

        Always returns ``(output, (k_new, v_new))`` — the caller decides
        whether to use the new K, V (for cache storage) or discard them.

        Parameters
        ----------
        query : (batch_size, seq_len, d_model)
        key   : (batch_size, seq_len, d_model)
        value : (batch_size, seq_len, d_model)
        mask  : (seq_len, seq_len) or broadcastable — True = allowed
        rope  : RotaryEmbedding | None, optional
            If provided, applies rotary position encoding to Q and K
            after reshaping into multi-head form.
        past_kv : tuple[torch.Tensor, torch.Tensor] | None, optional
            Cached (key, value) from previous decode steps, each shaped
            (batch, n_heads, cached_len, d_k). When provided, the input
            ``key`` / ``value`` contain ONLY the new token(s); they are
            concatenated with the cached tensors along dim=-2 *before*
            the attention computation.

        Returns
        -------
        (output, (k_new, v_new)) : tuple
            output  : (batch_size, seq_len_q, d_model) — attention result.
            k_new   : (batch, n_heads, seq_len_new, d_k) — new key projection
                      (after RoPE if applicable), *without* cached content.
            v_new   : (batch, n_heads, seq_len_new, d_k) — new value projection.
            The caller (KVCache) is responsible for appending (k_new, v_new)
            to the cache — this avoids double-concatenation bugs.
        """
        batch = query.size(0)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = q.reshape(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.reshape(batch, -1, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = v.reshape(batch, -1, self.n_kv_heads, self.d_k).transpose(1, 2)

        # ── RoPE: rotate Q and K in their per-head space ──────────────
        if rope is not None:
            q = rope(q)
            k = rope(k)

        # ── KV Cache: keep new K, V reference before expansion ──
        k_new, v_new = k, v

        if past_kv is not None:
            k = torch.cat([past_kv[0], k_new], dim=-2)
            v = torch.cat([past_kv[1], v_new], dim=-2)

        # ── GQA: expand K, V to match Q head count by repeating groups ──
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        out = scaled_dot_product_attention(q, k, v, mask)

        if self.dropout is not None:
            out = self.dropout(out)

        out = out.transpose(1, 2).reshape(batch, -1, self.d_model)

        out = self.w_o(out)

        return (out, (k_new, v_new))
