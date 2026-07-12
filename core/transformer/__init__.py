"""
core/transformer — Transformer building blocks (Phase 5).

Starting from a decoder-only (GPT-family) perspective, we build each
component from first principles using PyTorch.
"""

from core.transformer.data import CharLevelDataset, create_dataloaders
from core.transformer.embedding import BPETokenizer, CharTokenizer, TokenEmbedding
from core.transformer.kv_cache import KVCache
from core.transformer.moe import MoEFFN, MoERouter
from core.transformer.normalization import RMSNorm
from core.transformer.sampling import sample, sample_argmax, sample_top_k, sample_top_p
from core.transformer.transformer import (
    GPT,
    FeedForward,
    GPTBlock,
    MultiHeadAttention,
    RotaryEmbedding,
    apply_rotary_emb,
    precompute_freqs_cis,
    scaled_dot_product_attention,
)

__all__ = [
    "BPETokenizer",
    "MoERouter",
    "MoEFFN",
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "RMSNorm",
    "FeedForward",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "RotaryEmbedding",
    "GPTBlock",
    "CharTokenizer",
    "TokenEmbedding",
    "GPT",
    "KVCache",
    "CharLevelDataset",
    "create_dataloaders",
    "sample",
    "sample_argmax",
    "sample_top_k",
    "sample_top_p",
]
