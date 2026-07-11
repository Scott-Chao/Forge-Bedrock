"""
core/transformer/embedding.py — Token embedding and tokenizers.

Tokenizers convert text to/from integer token IDs:
    - CharTokenizer: character-level (Phase 5 baseline)
    - BPETokenizer: subword-level via Byte Pair Encoding (Phase 6)

The TokenEmbedding layer maps token IDs to dense vectors.

    text → [CharTokenizer|BPETokenizer] → token_ids → [TokenEmbedding] → vectors
"""

from __future__ import annotations

import re

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


class BPETokenizer:
    """Byte Pair Encoding tokenizer.

    Learns a subword vocabulary by iteratively merging the most frequent
    adjacent character pairs within pre-tokenized words.

    Parameters
    ----------
    vocab_size : int, optional (default=256)
        Target vocabulary size including base characters and special tokens.
        Must be > len(base_characters) + len(special_tokens).
    special_tokens : list[str] | None, optional
        Special tokens like ``["<PAD>", "<UNK>", "<BOS>", "<EOS>"]``.
        These occupy the first N IDs in the vocabulary.
    regex_pattern : str, optional (default=r'\\w+|\\s+')
        Regex pattern for pre-tokenization (splitting text into "words").
        The default pattern ``r'\\w+|\\s+'`` matches word characters and
        whitespace as separate tokens, keeping spacing information in the
        vocabulary so that ``decode(encode(text))`` recovers original
        whitespace.
        Non-word, non-space characters (punctuation, apostrophes, etc.)
        are discarded.

    Attributes
    ----------
    vocab : dict[str, int]
        Token-to-ID mapping (populated after training).
    id_to_token : dict[int, str]
        ID-to-token mapping (inverse of vocab).
    merges : dict[tuple[str, str], str]
        Learned merge rules: ``(left, right) -> merged_token``.
    merge_ranks : dict[tuple[str, str], int]
        Merge priority: ``(left, right) -> rank`` (lower = learned earlier).
    """

    def __init__(
        self,
        vocab_size: int = 256,
        special_tokens: list[str] | None = None,
        regex_pattern: str = r"\w+|\s+|[^\w\s]",
    ):
        if special_tokens is None:
            special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.regex_pattern = regex_pattern

        # Populated during training
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: dict[tuple[str, str], str] = {}
        self.merge_ranks: dict[tuple[str, str], int] = {}

    def _pre_tokenize(self, text: str) -> list[str]:
        """Split raw text into pre-token "words".

        The default regex ``r'\\w+|\\s+|[^\\w\\s]'`` matches word characters,
        whitespace, and individual punctuation symbols as separate pre-tokens.
        Whitespace and punctuation are preserved so that
        ``decode(encode(text))`` can recover original formatting.

        Parameters
        ----------
        text : str
            Raw input text (e.g., ``"Hello, world!"``).

        Returns
        -------
        words : list[str]
            List of pre-tokenized words (e.g., ``["Hello", "world"]``).
        """
        return re.findall(self.regex_pattern, text)

    def _get_char_vocab(self, words: list[list[str]]) -> dict[str, int]:
        """Build initial character-level vocabulary from pre-tokenized words.

        Each word (already split into characters) supplies its individual
        characters. Each unique character gets a unique token ID starting
        after the special tokens.

        Parameters
        ----------
        words : list[list[str]]
            Each element is a word represented as a list of characters.

        Returns
        -------
        char_vocab : dict[str, int]
            Character-to-ID mapping.
        """
        chars = {c for word in words for c in word}
        start_id = len(self.special_tokens)
        char_vocab = {c: i for i, c in enumerate(sorted(chars), start=start_id)}
        return char_vocab

    def _get_pair_freqs(self, words: list[list[str]]) -> dict[tuple[str, str], int]:
        """Count frequency of adjacent character pairs across all words.

        For each word represented as a list of symbols (initially characters,
        later also merged subwords), count how often each adjacent pair
        ``(symbol[i], symbol[i+1])`` appears.

        Parameters
        ----------
        words : list[list[str]]
            Each word is a list of symbol strings (characters or subwords).

        Returns
        -------
        pair_freqs : dict[tuple[str, str], int]
            Pair-to-frequency mapping.
        """
        pair_freqs = {}
        for word in words:
            for a, b in zip(word, word[1:]):
                key = (a, b)
                pair_freqs[key] = pair_freqs.get(key, 0) + 1
        return pair_freqs

    def _merge_pair(
        self, words: list[list[str]], pair: tuple[str, str], replacement: str
    ) -> list[list[str]]:
        """Replace all occurrences of ``pair`` with ``replacement`` in every word.

        Parameters
        ----------
        words : list[list[str]]
            Current state of the corpus (each word = list of symbols).
        pair : tuple[str, str]
            The pair ``(a, b)`` to merge.
        replacement : str
            The merged symbol (typically ``a + b``).

        Returns
        -------
        new_words : list[list[str]]
            Corpus with the pair merged.
        """
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words

    def train(self, texts: list[str]) -> None:
        """Learn BPE merge rules from a corpus.

        Pipeline:
            1. Pre-tokenize each text into words
            2. Split each word into characters (initial vocabulary)
            3. Iteratively find and merge the most frequent adjacent pair
            4. Record each merge with its rank
            5. Build final vocabulary (base chars + merged tokens + special tokens)

        Parameters
        ----------
        texts : list[str]
            List of raw text strings (the training corpus).
        """
        words = [list(w) for t in texts for w in self._pre_tokenize(t)]
        self.vocab = {s: i for i, s in enumerate(self.special_tokens)}
        self.vocab.update(self._get_char_vocab(words))
        self.id_to_token = {i: c for c, i in self.vocab.items()}

        while len(self.vocab) < self.vocab_size:
            pair_freqs = self._get_pair_freqs(words)
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=lambda p: pair_freqs[p])
            if pair_freqs[best_pair] < 2:
                break

            merged = best_pair[0] + best_pair[1]
            words = self._merge_pair(words, best_pair, merged)

            next_id = len(self.vocab)
            self.merges[best_pair] = merged
            self.merge_ranks[best_pair] = len(self.merges)
            self.vocab[merged] = next_id
            self.id_to_token[next_id] = merged

    def _encode_word(self, word: str) -> list[str]:
        """Encode a single pre-tokenized word using learned BPE merges.

        1. Split the word into individual characters
        2. Repeatedly find the adjacent pair with the lowest merge rank
        3. Merge that pair
        4. Repeat until no more merges are possible

        Parameters
        ----------
        word : str
            A single pre-tokenized word (e.g., ``"low"``).

        Returns
        -------
        symbols : list[str]
            List of subword tokens (e.g., ``["low"]``).
        """
        symbols = list(word)
        while len(symbols) > 1:
            candidates = [
                (a, b)
                for a, b in zip(symbols, symbols[1:])
                if (a, b) in self.merge_ranks
            ]
            if not candidates:
                break
            best_pair = min(candidates, key=lambda p: self.merge_ranks[p])

            symbols = self._merge_pair([symbols], best_pair, self.merges[best_pair])[0]

        return symbols

    def encode(self, text: str) -> list[int]:
        """Convert a text string to a list of token IDs.

        Pipeline:
            1. Pre-tokenize the text into words
            2. For each word, apply learned merges via ``_encode_word``
            3. Convert subword tokens to token IDs via ``self.vocab``

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        ids : list[int]
            Sequence of integer token IDs.
        """
        words = self._pre_tokenize(text)
        unk_id = self.vocab.get("<UNK>")
        ids = []
        for word in words:
            for token in self._encode_word(word):
                ids.append(self.vocab.get(token, unk_id))
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert a list of token IDs back to text.

        Parameters
        ----------
        ids : list[int]
            Sequence of integer token IDs.
        skip_special_tokens : bool, optional (default=True)
            Whether to exclude special tokens from the output.

        Returns
        -------
        text : str
            Decoded text string.
        """
        tokens = [self.id_to_token[id] for id in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        return "".join(tokens)

    def __len__(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self.vocab)

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={self.vocab_size}, "
            f"learned_merges={len(self.merges)})"
        )


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
