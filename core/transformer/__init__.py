"""
core/transformer — Transformer building blocks (Phase 5).

Starting from a decoder-only (GPT-family) perspective, we build each
component from first principles using PyTorch.
"""

from core.transformer.attention import MultiHeadAttention, scaled_dot_product_attention
from core.transformer.checkpoint import load_checkpoint
from core.transformer.data import CharLevelDataset, create_dataloaders
from core.transformer.embedding import BPETokenizer, CharTokenizer, TokenEmbedding
from core.transformer.feedforward import FeedForward
from core.transformer.kv_cache import KVCache
from core.transformer.moe import MoEFFN, MoERouter
from core.transformer.normalization import RMSNorm
from core.transformer.rope import (
    RotaryEmbedding,
    apply_rotary_emb,
    precompute_freqs_cis,
)
from core.transformer.sampling import sample, sample_argmax, sample_top_k, sample_top_p
from core.transformer.transformer import (
    GPT,
    GPTBlock,
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
    "load_checkpoint",
    "sample",
    "sample_argmax",
    "sample_top_k",
    "sample_top_p",
]
