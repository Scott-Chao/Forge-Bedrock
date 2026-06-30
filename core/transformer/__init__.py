"""
core/transformer — Transformer building blocks (Phase 5).

Starting from a decoder-only (GPT-family) perspective, we build each
component from first principles using PyTorch.
"""

from core.transformer.attention import MultiHeadAttention, scaled_dot_product_attention
from core.transformer.feedforward import FeedForward
from core.transformer.normalization import RMSNorm

__all__ = [
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "RMSNorm",
    "FeedForward",
]
