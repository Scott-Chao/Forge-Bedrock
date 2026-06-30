"""
core/transformer — Transformer building blocks (Phase 5).

Starting from a decoder-only (GPT-family) perspective, we build each
component from first principles using PyTorch.
"""

from core.transformer.attention import scaled_dot_product_attention

__all__ = ["scaled_dot_product_attention"]
