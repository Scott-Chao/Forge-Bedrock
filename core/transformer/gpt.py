"""
core/transformer/gpt.py — Full GPT language model.

This module assembles all previously built components into a complete
decoder-only Transformer model (GPT-family architecture):

    tokens → [TokenEmbedding] → [GPTBlock × N] → [RMSNorm] → [lm_head] → logits

The model is autoregressive: it predicts the next token at each position.
During training, cross-entropy loss is computed between logits and targets.
During inference, logits are used to sample the next token iteratively.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from core.transformer.block import GPTBlock
from core.transformer.embedding import TokenEmbedding
from core.transformer.normalization import RMSNorm


class GPT(nn.Module):
    """GPT Decoder-Only Language Model.

    Parameters
    ----------
    vocab_size : int
        Number of unique tokens in the vocabulary.
    d_model : int
        Feature dimension throughout the model.
    n_layers : int
        Number of stacked GPTBlocks.
    n_heads : int
        Number of attention heads per block.
    max_seq_len : int, optional (default=2048)
        Maximum sequence length for RoPE and causal masks.
    d_ff : int | None, optional (default=None)
        FeedForward hidden dimension. If None, defaults to 4 * d_model.
    dropout : float, optional (default=0.0)
        Dropout probability applied in attention.
    bias : bool, optional (default=True)
        Whether to use bias in linear projections.
    tie_weights : bool, optional (default=False)
        If True, share weights between input embedding and output
        projection (weight tying). When enabled, the lm_head weight
        is set to the embedding matrix and lm_head is not trained
        as a separate parameter.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int = 2048,
        d_ff: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        tie_weights: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                GPTBlock(d_model, n_heads, max_seq_len, d_ff, dropout, bias)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.embedding.weight

        self.apply(self._init_weights)

    def forward(
        self,
        tokens: torch.LongTensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the full GPT model.

        Parameters
        ----------
        tokens : (batch_size, seq_len)
            Input token IDs (integers).
        mask : (seq_len, seq_len) | None, optional
            Causal attention mask. If None, each GPTBlock will create
            its own causal mask based on seq_len.

        Returns
        -------
        logits : (batch_size, seq_len, vocab_size)
            Unnormalised scores for each token at each position.
            Shape: (batch, seq, vocab_size).
        """
        x = self.token_embedding(tokens)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 convention: N(0, 0.02).

        This method is designed to be applied with self.apply():

            self.apply(self._init_weights)

        It initializes all linear layers and embedding layers with
        a Normal(0, 0.02) distribution.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"GPT(vocab_size={self.vocab_size}, d_model={self.d_model}, "
            f"n_layers={self.n_layers}, n_heads={self.n_heads}, "
            f"params={self.num_parameters:,})"
        )
