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

import re
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

    @property
    def pad_id(self) -> int:
        return self.vocab.get("<PAD>", 0)

    @property
    def unk_id(self) -> int:
        return self.vocab.get("<UNK>", 1)

    @property
    def bos_id(self) -> int:
        return self.vocab.get("<BOS>", 2)

    @property
    def eos_id(self) -> int:
        return self.vocab.get("<EOS>", 3)

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
    vocab_size : int, optional (default=256)
        Target vocabulary size including base characters and special tokens.
        Must be > len(base_characters) + len(special_tokens).
    special_tokens : list[str] | None, optional
        Special tokens like ``["<PAD>", "<UNK>", "<BOS>", "<EOS>"]``.
        These occupy the first N IDs in the vocabulary.
    regex_pattern : str, optional (default=r'\\w+|\\s+|[^\\w\\s]')
        Regex pattern for pre-tokenization. The default matches word
        characters, whitespace, and individual punctuation as separate
        pre-tokens, preserving formatting for encode/decode roundtrips.

    Attributes
    ----------
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
        super().__init__(special_tokens)
        self._target_vocab_size = vocab_size
        self.regex_pattern = regex_pattern
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

    def train(self, texts: list[str]) -> None:
        """Learn BPE merge rules from a corpus.

        Pipeline:
            1. Pre-tokenize each text into words
            2. Split each word into characters (initial vocabulary)
            3. Iteratively find and merge the most frequent adjacent pair
            4. Record each merge with its rank
            5. Build final vocabulary (base chars + merged tokens + special tokens)

        Performance
        -----------
        Uses a Counter of unique words to avoid O(vocab_size × corpus) scans.
        Most words in natural language repeat, so this is ~10-100× faster than
        the naive per-token approach.
        """
        # Build unique-word counts from the corpus
        raw_words: list[tuple[str, ...]] = []
        for t in texts:
            for w in self._pre_tokenize(t):
                raw_words.append(tuple(w))
        word_counts = Counter(raw_words)

        # Initialise vocab with special tokens and base characters
        words_as_lists = [list(w) for w in word_counts]
        self.vocab = {s: i for i, s in enumerate(self.special_tokens)}
        self.vocab.update(self._get_char_vocab(words_as_lists))
        self.id_to_token = {i: c for c, i in self.vocab.items()}

        # Convert to tuple-based Counter for efficient weighted operations
        # (each word is a tuple of symbols/chars)
        word_counts = Counter({tuple(w): word_counts[tuple(w)] for w in words_as_lists})

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
            1. Pre-tokenize the text into words
            2. For each word, apply learned merges via ``_encode_word``
            3. Convert subword tokens to token IDs via ``self.vocab``

        Parameters
        ----------
        text : str
            Raw input text.
        **kwargs
            Ignored extra keyword arguments for API compatibility with
            ``CharLevelDataset`` (which passes ``add_special_tokens=False``).

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
