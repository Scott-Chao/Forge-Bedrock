"""
core/transformer/embedding.py — Token embedding and character-level tokenizer.

For this project, we use a character-level tokenizer (no BPE/WordPiece)
with a vocabulary size of ~70 characters. This keeps the model simple
and lets us focus on understanding the Transformer internals.

    text → [CharTokenizer] → token_ids → [TokenEmbedding] → vectors
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _build_default_vocab() -> dict[str, int]:
    """Build the default character-level vocabulary.

    Returns a dict mapping each character to a unique integer index.

    Characters included:
        - Lowercase letters: a-z
        - Uppercase letters: A-Z
        - Digits: 0-9
        - Punctuation: . , ! ? : ; ' " - ( ) [ ] { } & % $ @ # * / \\
        - Whitespace: space, newline, tab
        - Special tokens: <PAD> <UNK> <BOS> <EOS>

    Total ~80-85 characters. The exact count is flexible.
    """
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?:;\"'()-[]{}&%$@#*/\\ \n\t"
    all_tokens = special_tokens + list(chars)
    return {token: i for i, token in enumerate(all_tokens)}


class CharTokenizer:
    """Character-level tokenizer for a minimal GPT.

    Converts text strings to/from sequences of integer token IDs using
    a simple character-to-index mapping. No BPE, no WordPiece.

    Parameters
    ----------
    vocab : dict[str, int] | None, optional
        Character-to-index mapping. If None, uses the default vocabulary
        from _build_default_vocab().
    """

    def __init__(self, vocab: dict[str, int] | None = None):
        if vocab is None:
            vocab = _build_default_vocab()
        self.vocab = vocab

        self.itos = {i: c for c, i in vocab.items()}

        self.pad_id = vocab.get("<PAD>", 0)
        self.unk_id = vocab.get("<UNK>", 1)
        self.bos_id = vocab.get("<BOS>", 2)
        self.eos_id = vocab.get("<EOS>", 3)

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Convert a text string to a list of token IDs.

        Parameters
        ----------
        text : str
            Input text (e.g., "Hello, world!").
        add_special_tokens : bool, optional (default=True)
            If True, prepend <BOS> and append <EOS> tokens.

        Returns
        -------
        ids : list[int]
            Sequence of integer token IDs.
        """
        ids = [self.vocab.get(c, self.unk_id) for c in text]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert a list of token IDs back to a string.

        Parameters
        ----------
        ids : list[int]
            Sequence of integer token IDs.
        skip_special_tokens : bool, optional (default=True)
            If True, exclude <PAD>, <UNK>, <BOS>, <EOS> from the output.

        Returns
        -------
        text : str
            Decoded string.
        """
        if skip_special_tokens:
            special = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            return "".join(self.itos[i] for i in ids if i not in special)
        return "".join(self.itos[i] for i in ids)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


class TokenEmbedding(nn.Module):
    """Token embedding layer (lookup table).

    Maps integer token IDs to dense vectors:

        output = embedding[token_ids]

    This is the first layer of the GPT model, converting character
    indices into continuous representations that flow through the
    Transformer blocks.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (number of unique tokens).
    d_model : int
        Dimension of the embedding vectors.
    padding_idx : int | None, optional (default=None)
        If specified, the embedding at padding_idx is not updated
        during training (gradient is always zero).
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Convert token IDs to embedding vectors.

        Parameters
        ----------
        tokens : (batch_size, seq_len)
            Long tensor of token IDs (integers).

        Returns
        -------
        out : (batch_size, seq_len, d_model)
            Dense embedding vectors.
        """
        return self.embedding(tokens)
