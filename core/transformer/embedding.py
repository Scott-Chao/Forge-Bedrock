"""
core/transformer/embedding.py — Token embedding and tokenizers.

Tokenizers convert text to/from integer token IDs:
    - CharTokenizer: character-level (Phase 5 baseline)
    - BPETokenizer: subword-level via Byte Pair Encoding (Phase 6)

The TokenEmbedding layer maps token IDs to dense vectors.

    text → [CharTokenizer|BPETokenizer] → token_ids → [TokenEmbedding] → vectors

Tokenizer hierarchy
-------------------
Both tokenizer classes share common encode/decode infrastructure
through a lightweight ``Tokenizer`` base class.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter

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
    chars = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789.,!?:;\"'()-[]{}&%$@#*/\\ \n\t"
    )
    all_tokens = special_tokens + list(chars)
    return {token: i for i, token in enumerate(all_tokens)}


# Mapping of rarely-occurring Unicode characters to ASCII equivalents.
# Only characters that appear fewer than ``min_char_freq`` times in the
# corpus (and would otherwise be dropped by the frequency filter) need
# mapping. Common Unicode punctuation (em-dash, curly quotes, etc.)
# stays in the vocabulary as independent tokens.
_INVISIBLE_CHARS = {
    "\xa0": " ",  # NO-BREAK SPACE
    "​": "",  # ZERO-WIDTH SPACE
    "﻿": "",  # BOM / ZWNBSP
}
_RARE_PUNCT_MAP = {
    # Rare dashes
    "―": "-",  # HORIZONTAL BAR
    "‑": "-",  # NON-BREAKING HYPHEN
    # Rare quotes (single/double low, guillemets)
    "‚": "'",  # SINGLE LOW-9 QUOTATION MARK
    "‛": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "‟": '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "«": '"',  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    "»": '"',  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    "‹": "'",  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    "›": "'",  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    "″": '"',  # DOUBLE PRIME
    # Rare typographic symbols
    "…": "...",  # HORIZONTAL ELLIPSIS
    "•": "-",  # BULLET
    "·": "-",  # MIDDLE DOT
    "№": "No",  # NUMERO SIGN
    "©": "(c)",  # COPYRIGHT
    "®": "(r)",  # REGISTERED
}


# ═══════════════════════════════════════════════════════════════════════
# Text processing utilities (used by BPETokenizer)
# ═══════════════════════════════════════════════════════════════════════


def normalize_text(text: str) -> str:
    """Normalize text for consistent tokenizer train/encode behavior.

    Pipeline:
        1. Strip invisible characters (BOM, ZWSP, NBSP → space)
        2. Map very rare Unicode punctuation to ASCII equivalents
           (guillemets, rare quotes, typographic symbols)
        3. NFKD decompose then strip combining marks (accents off)
        4. NFKC recompose for canonical form

    Keeps common Unicode punctuation (em-dash, curly quotes, etc.)
    as independent tokens since they pass the frequency threshold.
    """
    for k, v in _INVISIBLE_CHARS.items():
        text = text.replace(k, v)
    for k, v in _RARE_PUNCT_MAP.items():
        text = text.replace(k, v)
    nfkd = unicodedata.normalize("NFKD", text)
    no_accents = "".join(c for c in nfkd if unicodedata.category(c) not in ("Mn", "Mc"))
    return unicodedata.normalize("NFKC", no_accents)


def pre_tokenize(text: str, pattern: str = r"\w+|\s+|[^\w\s]") -> list[str]:
    """Split raw text into pre-token "words" for BPE training / encoding.

    The default regex ``\\w+|\\s+|[^\\w\\s]`` matches word characters,
    whitespace, and individual punctuation symbols as separate pre-tokens.
    Whitespace and punctuation are preserved so that
    ``decode(encode(text))`` can recover original formatting.

    Parameters
    ----------
    text : str
        Raw input text (e.g., ``"Hello, world!"``).
    pattern : str, optional
        Regex pattern for splitting.

    Returns
    -------
    words : list[str]
        List of pre-tokenized words (e.g., ``["Hello", "world"]``).
    """
    return re.findall(pattern, text)


def _filter_word_counts(word_counts: Counter, kept_chars: set[str]) -> Counter:
    """Drop characters not in ``kept_chars`` from all words.

    Weighted frequencies are preserved so rare words don't get inflated.
    """
    filtered = Counter()
    for word, count in word_counts.items():
        cleaned = tuple(c for c in word if c in kept_chars)
        if cleaned:
            filtered[cleaned] += count
    return filtered


class Tokenizer:
    """Base class for tokenizers.

    Provides shared ``decode``, ``__len__``, and ``vocab_size``
    infrastructure. Subclasses must implement ``encode`` and set up
    their own ``vocab`` / ``id_to_token`` in ``__init__``.

    Parameters
    ----------
    special_tokens : list[str] | None, optional
        Tokens reserved for special purposes (PAD, UNK, BOS, EOS).
        If None, defaults to ``["<PAD>", "<UNK>", "<BOS>", "<EOS>"]``.
    """

    def __init__(self, special_tokens: list[str] | None = None):
        if special_tokens is None:
            special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.special_tokens = special_tokens
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    @property
    def pad_id(self) -> int:
        """Index of the padding token (defaults to 0)."""
        return self.vocab.get("<PAD>", 0)

    @property
    def unk_id(self) -> int:
        """Index of the unknown token (defaults to 1)."""
        return self.vocab.get("<UNK>", 1)

    @property
    def bos_id(self) -> int:
        """Index of the beginning-of-sequence token (defaults to 2)."""
        return self.vocab.get("<BOS>", 2)

    @property
    def eos_id(self) -> int:
        """Index of the end-of-sequence token (defaults to 3)."""
        return self.vocab.get("<EOS>", 3)

    def encode(self, text: str) -> list[int]:
        """Convert text to a list of token IDs.

        Subclasses must override this with their specific encoding logic.
        """
        raise NotImplementedError

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert a list of token IDs back to text.

        Parameters
        ----------
        ids : list[int]
            Sequence of integer token IDs.
        skip_special_tokens : bool, optional (default=True)
            If True, exclude special tokens from the output.

        Returns
        -------
        text : str
            Decoded string.
        """
        tokens = [self.id_to_token[i] for i in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        return "".join(tokens)

    @property
    def vocab_size(self) -> int:
        """Number of tokens currently in the vocabulary."""
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)


class CharTokenizer(Tokenizer):
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
        super().__init__()
        if vocab is None:
            vocab = _build_default_vocab()
        self.vocab = vocab
        self.id_to_token = {i: c for c, i in vocab.items()}

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

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


class BPETokenizer(Tokenizer):
    """Byte Pair Encoding tokenizer.

    Learns a subword vocabulary by iteratively merging the most frequent
    adjacent character pairs within pre-tokenized words.

    Parameters
    ----------
    vocab_size : int, optional (default=2048)
        Target vocabulary size including base characters and special tokens.
    special_tokens : list[str] | None, optional
        Special tokens like ``["<PAD>", "<UNK>", "<BOS>", "<EOS>"]``.
        These occupy the first N IDs in the vocabulary.
    regex_pattern : str, optional
        Regex pattern for pre-tokenization. The default matches word
        characters, whitespace, and individual punctuation as separate
        pre-tokens.
    min_char_freq : int, optional (default=5)
        Minimum corpus frequency for a non-ASCII character to be included
        as a base vocabulary character. ASCII characters are always kept.
        Raising this frees more vocab slots for BPE merges at the cost of
        a tiny increase in ``<UNK>`` tokens.

    Attributes
    ----------
    merges : dict[tuple[str, str], str]
        Learned merge rules: ``(left, right) -> merged_token``.
    merge_ranks : dict[tuple[str, str], int]
        Merge priority: ``(left, right) -> rank`` (lower = learned earlier).
    """

    def __init__(
        self,
        vocab_size: int = 2048,
        special_tokens: list[str] | None = None,
        regex_pattern: str = r"\w+|\s+|[^\w\s]",
        min_char_freq: int = 5,
    ):
        super().__init__(special_tokens)
        self._target_vocab_size = vocab_size
        self.regex_pattern = regex_pattern
        self._min_char_freq = min_char_freq
        self.merges: dict[tuple[str, str], str] = {}
        self.merge_ranks: dict[tuple[str, str], int] = {}

    def _get_pair_freqs(self, word_counts: Counter) -> dict[tuple[str, str], int]:
        """Count frequency of adjacent character pairs, weighted by word counts.

        Uses a Counter of unique word tuples to avoid O(corpus_size) scans.
        Each word's pair frequencies are multiplied by its occurrence count.

        Parameters
        ----------
        word_counts : Counter[tuple[str, ...]]
            Maps each unique word (as a tuple of symbols) to its frequency.

        Returns
        -------
        pair_freqs : dict[tuple[str, str], int]
            Pair-to-frequency mapping.
        """
        pair_freqs = {}
        for word, count in word_counts.items():
            for a, b in zip(word, word[1:]):
                key = (a, b)
                pair_freqs[key] = pair_freqs.get(key, 0) + count
        return pair_freqs

    def _merge_pair(
        self, word_counts: Counter, pair: tuple[str, str], replacement: str
    ) -> Counter:
        """Replace all occurrences of ``pair`` with ``replacement`` in every word.

        Operates on unique words (Counter) and returns an updated Counter.

        Parameters
        ----------
        word_counts : Counter[tuple[str, ...]]
            Unique words with their frequencies.
        pair : tuple[str, str]
            The pair ``(a, b)`` to merge.
        replacement : str
            The merged symbol (typically ``a + b``).

        Returns
        -------
        new_word_counts : Counter[tuple[str, ...]]
            Updated word counts with the pair merged.
        """
        new_counts: Counter = Counter()
        for word, count in word_counts.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_counts[tuple(new_word)] += count
        return new_counts

    def _build_base_vocab(self, word_counts: Counter) -> None:
        """Build base character vocabulary from corpus frequencies.

        ASCII characters are always kept. Non-ASCII characters appearing
        fewer than ``min_char_freq`` times are excluded so that BPE merge
        slots aren't wasted on corpus-specific rarities.
        """
        char_freqs: Counter = Counter()
        for word, count in word_counts.items():
            for c in word:
                char_freqs[c] += count

        base_chars = sorted(
            c
            for c in char_freqs
            if ord(c) < 128 or char_freqs[c] >= self._min_char_freq
        )
        self.vocab = {s: i for i, s in enumerate(self.special_tokens)}
        self.vocab.update(
            {c: i for i, c in enumerate(base_chars, start=len(self.special_tokens))}
        )
        self.id_to_token = {i: c for c, i in self.vocab.items()}

    def train(self, texts: list[str]) -> None:
        """Learn BPE merge rules from a corpus.

        Pipeline:
            1. Normalise each text (punctuation mapping → accent stripping)
            2. Pre-tokenize into words and build frequency tables
            3. Build base character vocabulary, dropping non-ASCII characters
               below ``min_char_freq``
            4. Iteratively find and merge the most frequent adjacent pair
            5. Record each merge with its rank

        Performance
        -----------
        Uses a Counter of unique words to avoid O(vocab_size × corpus) scans.
        Most words in natural language repeat, so this is ~10-100× faster than
        the naive per-token approach.

        Notes
        -----
        Normalization is applied at the start so that BPE merges are learned
        on the same character distribution that ``encode`` will see.
        """
        # 1. Normalize all texts
        texts = [normalize_text(t) for t in texts]

        # 2. Pre-tokenize and build unique-word counts (weighted)
        word_counts: Counter = Counter()
        for t in texts:
            for w in pre_tokenize(t, self.regex_pattern):
                word_counts[tuple(w)] += 1

        # 3. Build base character vocabulary (filters rare non-ASCII)
        self._build_base_vocab(word_counts)

        # Strip chars that didn't make the cut from word_counts
        kept_chars = set(self.vocab)
        word_counts = _filter_word_counts(word_counts, kept_chars)

        # 4. BPE merge loop
        while len(self.vocab) < self._target_vocab_size:
            pair_freqs = self._get_pair_freqs(word_counts)
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=lambda p: pair_freqs[p])
            if pair_freqs[best_pair] < 2:
                break

            merged = best_pair[0] + best_pair[1]
            word_counts = self._merge_pair(word_counts, best_pair, merged)

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
            merged = self.merges[best_pair]

            # Inline merge: build new symbol list, skipping the matched pair
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(self, text: str, **kwargs) -> list[int]:
        """Convert a text string to a list of token IDs.

        Pipeline:
            1. Normalize input text (matches training-time preprocessing)
            2. Pre-tokenize the text into words
            3. For each word, apply learned merges via ``_encode_word``
            4. Convert subword tokens to token IDs via ``self.vocab``

        Parameters
        ----------
        text : str
            Raw input text. Will be normalized the same way as the training
            corpus (punctuation mapping + accent stripping) before encoding.
        **kwargs
            Ignored extra keyword arguments for API compatibility with
            ``CharLevelDataset`` (which passes ``add_special_tokens=False``).

        Returns
        -------
        ids : list[int]
            Sequence of integer token IDs. Characters that were filtered out
            during training (very rare non-ASCII) become ``<UNK>``.
        """
        text = normalize_text(text)

        words = pre_tokenize(text, self.regex_pattern)
        unk_id = self.vocab.get("<UNK>")
        ids = []
        for word in words:
            for token in self._encode_word(word):
                ids.append(self.vocab.get(token, unk_id))
        return ids

    def save(self, path: str | os.PathLike) -> None:
        """Save tokenizer state to a file.

        Parameters
        ----------
        path : str | os.PathLike
            Destination path (typically a ``.pt`` file).
        """
        torch.save(
            {
                "vocab": self.vocab,
                "id_to_token": self.id_to_token,
                "merges": self.merges,
                "merge_ranks": self.merge_ranks,
                "special_tokens": self.special_tokens,
                "regex_pattern": self.regex_pattern,
            },
            path,
        )

    @classmethod
    def from_pretrained(
        cls,
        path: str | os.PathLike,
    ) -> BPETokenizer:
        """Load a previously saved BPETokenizer from a file.

        Parameters
        ----------
        path : str | os.PathLike
            Path to a ``.pt`` file previously written by ``.save()``.

        Returns
        -------
        tokenizer : BPETokenizer
            Restored tokenizer with full vocab, merges, and config.
        """
        data = torch.load(path, map_location="cpu", weights_only=True)
        tokenizer = cls(
            vocab_size=len(data["vocab"]),
            special_tokens=data["special_tokens"],
            regex_pattern=data["regex_pattern"],
        )
        tokenizer.vocab = data["vocab"]
        tokenizer.id_to_token = data["id_to_token"]
        tokenizer.merges = data["merges"]
        tokenizer.merge_ranks = data["merge_ranks"]
        return tokenizer

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
