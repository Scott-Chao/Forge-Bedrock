"""
core/transformer/kv_cache.py — Key-Value Cache for Autoregressive Decoding.

Mathematical Motivation
-----------------------
In autoregressive generation, at each step we compute:

    Q = x @ W_Q,    K = x @ W_K,    V = x @ W_V

where x is the token representation. Without caching, we recompute
K and V for ALL previous tokens at every step — O(n² d) wasted work.

The KV cache stores (K, V) for previously processed tokens so that
each step only computes K, V for the SINGLE new token:

    K_new = x_new @ W_K  →  K_cache = cat([K_cache, K_new], dim=-2)
    V_new = x_new @ W_V  →  V_cache = cat([V_cache, V_new], dim=-2)

Then attention uses the full cached K, V:

    Attention(Q_new, K_cache, V_cache) = softmax(Q_new @ K_cache^T / √d_k) @ V_cache

This reduces per-step attention from O(n²) to O(n).

Structure
---------
A KVCache holds one (key, value) pair per layer, stored in the
multi-head format:

    cache[layer_idx] = (k, v)
    k.shape: (batch_size, n_heads, cached_seq_len, d_k)
    v.shape: (batch_size, n_heads, cached_seq_len, d_k)
"""

from __future__ import annotations

import torch


class KVCache:
    """Key-Value cache for efficient autoregressive decoding.

    Stores cached keys and values for each transformer layer. The cache
    avoids recomputing K and V for previously generated tokens at every
    decoding step.

    Each layer's cache entry is a tuple (k, v) where:

        k : (batch_size, n_kv_heads, cached_seq_len, d_k)
        v : (batch_size, n_kv_heads, cached_seq_len, d_k)

    With GQA (Grouped Query Attention), ``n_kv_heads`` may be smaller than
    ``n_heads`` — the per-head expansion via ``repeat_interleave`` happens
    at attention time, *after* the cache lookup.

    The cache is layer-indexed so that each GPTBlock can update its own
    entry independently.
    """

    def __init__(self) -> None:
        """Initialise an empty KV cache."""
        self._cache = {}

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new (K, V) to the cache for a given layer.

        If this layer has no cached values yet (first call, e.g. during
        prefill), the cache entry is set to (k, v) directly.

        If cached values already exist (subsequent decode steps), the
        new (k, v) are concatenated along the sequence dimension (dim=-2):

            k_cache = cat([k_cache, k], dim=-2)
            v_cache = cat([v_cache, v], dim=-2)

        Parameters
        ----------
        layer_idx : int
            Index of the transformer layer this cache belongs to.
        k : (batch_size, n_heads, seq_len_new, d_k)
            Key tensor for the NEW token(s).
        v : (batch_size, n_heads, seq_len_new, d_k)
            Value tensor for the NEW token(s).

        Returns
        -------
        (k_full, v_full) : tuple[torch.Tensor, torch.Tensor]
            The FULL key and value tensors (cached + new), both shaped
            (batch_size, n_heads, cached_seq_len + seq_len_new, d_k).
            This is what attention uses as K and V.
        """
        if layer_idx not in self._cache:
            self._cache[layer_idx] = (k, v)
            return (k, v)

        k_cache, v_cache = self._cache[layer_idx]
        k_full = torch.cat([k_cache, k], dim=-2)
        v_full = torch.cat([v_cache, v], dim=-2)
        self._cache[layer_idx] = (k_full, v_full)
        return (k_full, v_full)

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Retrieve cached (K, V) for a layer without modifying the cache."""
        return self._cache.get(layer_idx, None)

    @property
    def size(self) -> int:
        """Return the number of layers currently stored in the cache."""
        return len(self._cache)

    @property
    def current_seq_len(self) -> int:
        """Return the sequence length of the cached tensors."""
        if not self._cache:
            return 0
        # All layers share the same sequence length; pick the first
        return next(iter(self._cache.values()))[0].shape[-2]

    def reset(self) -> None:
        """Clear all cached values."""
        self._cache = {}
