"""
core/transformer/rope.py — Rotary Positional Embedding (RoPE).

Provides the mathematical primitives and a convenience ``nn.Module``
wrapper for applying Rotary Position Embeddings to query and key
tensors in Transformer attention layers.

Mathematical background
-----------------------
For each pair of dimensions ``(2i, 2i+1)``, RoPE rotates the vector
by an angle proportional to the token's position:

    theta_i = base ** (-2 * i / d_model)

    RoPE(x, m)[2i]   = x[2i] * cos(m * theta_i) - x[2i+1] * sin(m * theta_i)
    RoPE(x, m)[2i+1] = x[2i] * sin(m * theta_i) + x[2i+1] * cos(m * theta_i)

These rotations can be applied efficiently without explicit matrix
construction via the ``rotate_half`` trick:

    x_rotated = x * cos(m*θ) + rotate_half(x) * sin(m*θ)

where ``rotate_half`` swaps the two halves and negates the first half.
"""

from __future__ import annotations

import torch

# ═══════════════════════════════════════════════════════════════════════
# Precomputation
# ═══════════════════════════════════════════════════════════════════════


def precompute_freqs_cis(
    d_model: int, max_seq_len: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE.

    For each pair index i in [0, d_model/2)::

        theta_i = base ** (-2 * i / d_model)

    For each position m in [0, max_seq_len)::

        cos[m] = [cos(mθ₀), cos(mθ₁), ..., cos(mθ_{d/2-1}),
                  cos(mθ₀), cos(mθ₁), ..., cos(mθ_{d/2-1})]
        sin[m] = [sin(mθ₀), sin(mθ₁), ..., sin(mθ_{d/2-1}),
                  sin(mθ₀), sin(mθ₁), ..., sin(mθ_{d/2-1})]

    Parameters
    ----------
    d_model : int
        Feature dimension (must be even).
    max_seq_len : int
        Maximum sequence length to precompute.
    base : float
        Base frequency (default 10000.0, same as original Transformer).

    Returns
    -------
    cos : (max_seq_len, d_model)
    sin : (max_seq_len, d_model)
    """
    theta = base ** (-torch.arange(0, d_model, 2).float() / d_model)
    m = torch.arange(max_seq_len)
    angles = torch.outer(m, theta)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    return cos, sin


# ═══════════════════════════════════════════════════════════════════════
# Core operations
# ═══════════════════════════════════════════════════════════════════════


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split-half swap::

        rotate_half([a, b, c, d, e, f, g, h]) = [-e, -f, -g, -h, a, b, c, d]

    Used in the efficient formula: ``x_rotated = x * cos + rotate_half(x) * sin``.
    """
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE using the efficient rotate_half formula::

        x_rotated = x * cos + rotate_half(x) * sin

    Parameters
    ----------
    x : (..., seq_len, d_model)
        Q or K tensor.
    cos : (max_seq_len, d_model)
        Precomputed cosines.
    sin : (max_seq_len, d_model)
        Precomputed sines.

    Returns
    -------
    out : (..., seq_len, d_model)
        Rotated tensor, same shape as input.
    """
    seq_len = x.size(-2)
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    return x * cos + rotate_half(x) * sin


# ═══════════════════════════════════════════════════════════════════════
# nn.Module wrapper
# ═══════════════════════════════════════════════════════════════════════


class RotaryEmbedding(torch.nn.Module):
    """Rotary Positional Embedding — convenient nn.Module wrapper.

    Precomputes the cos/sin tables once and applies RoPE to any Q/K tensor.

    Usage::

        rope = RotaryEmbedding(d_model)
        q_rope = rope(q)   # applies RoPE to Q
        k_rope = rope(k)   # applies RoPE to K
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        cos, sin = precompute_freqs_cis(d_model, max_seq_len, base)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        return apply_rotary_emb(x, self.cos[:seq_len], self.sin[:seq_len])
