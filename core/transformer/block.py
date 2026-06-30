"""
core/transformer/block.py — GPT Decoder Block (Pre-Norm).

Combines RMSNorm → MultiHeadAttention (+RoPE) → RMSNorm → FeedForward
with Pre-Norm residual connections:

    x₁ = x + Attention(RMSNorm(x))
    x₂ = x₁ + FFN(RMSNorm(x₁))

This is the standard GPT-2 / Llama-style decoder block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from core.transformer.attention import MultiHeadAttention, _create_causal_mask
from core.transformer.feedforward import FeedForward
from core.transformer.normalization import RMSNorm
from core.transformer.positional import RotaryEmbedding


class GPTBlock(nn.Module):
    """GPT Decoder Block with Pre-Norm architecture.

    A single transformer decoder block consisting of:

        1. RMSNorm → Multi-Head Self-Attention → Residual
        2. RMSNorm → Position-wise FeedForward → Residual

    The attention uses a causal mask so each token can only attend to
    itself and previous tokens (autoregressive property).

    Parameters
    ----------
    d_model : int
        Feature dimension of the model.
    n_heads : int
        Number of attention heads.
    max_seq_len : int, optional (default=2048)
        Maximum sequence length for RoPE precomputation and causal mask.
    d_ff : int | None, optional (default=None)
        FeedForward hidden dimension. If None, defaults to 4 * d_model.
    dropout : float, optional (default=0.0)
        Dropout probability for attention weights.
    bias : bool, optional (default=True)
        Whether to use bias in linear projections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        d_ff: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        self.norm_1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, bias)
        self.norm_2 = RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, bias)

        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process one transformer block.

        Parameters
        ----------
        x : (batch_size, seq_len, d_model)
            Input tensor.
        mask : (seq_len, seq_len) | None, optional
            Causal attention mask. If None, a causal mask is created
            automatically based on seq_len.

        Returns
        -------
        out : (batch_size, seq_len, d_model)
            Output tensor after self-attention + feedforward.
        """
        if mask is None:
            seq_len = x.size(1)
            mask = self._create_attn_mask(seq_len, device=x.device)

        residual = x
        x = self.norm_1(x)
        x = self.attn(x, x, x, mask, rope=self.rope)
        x = residual + x

        residual = x
        x = self.norm_2(x)
        x = self.ff(x)
        x = residual + x

        return x

    def _create_attn_mask(
        self, seq_len: int, device: torch.device | None = None
    ) -> torch.Tensor:
        """Create a causal attention mask for the given sequence length."""
        return _create_causal_mask(seq_len, device)
