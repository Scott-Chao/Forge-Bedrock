"""
core/transformer/data.py — Corpus loading and dataset for language modeling.

Provides the data pipeline for training a decoder-only GPT:

    raw_text → [tokenizer] → 1D integer array → [CharLevelDataset]
    → (input_ids, target_ids) pairs → [DataLoader] → batched training

Works with any tokenizer that provides ``encode()`` / ``decode()``
(character-level, BPE, etc.). See ``core.transformer.embedding`` for
available tokenizer implementations.

Self-supervised formulation
---------------------------
For a sequence x₁, x₂, ..., x_N, training pairs are:

    input  = (x_t,     x_{t+1},   ..., x_{t+T-1})
    target = (x_{t+1}, x_{t+2},   ..., x_{t+T})

where T is the block size (context length). The model predicts the next
token at every position, using causal masking to ensure it can't cheat.

Corpora
-------
TinyShakespeare (~1 MB)
    - Shakespeare plays, pure ASCII text
    - ~65 unique characters → vocab_size ≈ 65
    - Public domain, no download auth needed

text8 (~100 MB)
    - Wikipedia dump, lowercase a-z + space only
    - 27 unique characters → vocab_size = 27
    - Not used by default; can be swapped in.

WikiText-2 (~2.5M tokens)
    - Wikipedia articles, suitable for subword (BPE) tokenization
    - Used by the ``train_gpt.ipynb`` notebook
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────────────────────────────
# Corpus loading
# ──────────────────────────────────────────────────────────────────────

CORPUS_SOURCES: dict[str, str] = {
    "tinyshakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/"
        "master/data/tinyshakespeare/input.txt"
    ),
}


def download_corpus(
    name: str = "tinyshakespeare",
    data_dir: str | os.PathLike = "assets/",
) -> str:
    """Download a known corpus if not already cached, return the text."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / f"{name}.txt"

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    url = CORPUS_SOURCES[name]
    urllib.request.urlretrieve(url, cache_path)
    return cache_path.read_text(encoding="utf-8")


def load_corpus_from_file(path: str | os.PathLike) -> str:
    """Load a corpus from a local text file."""
    return Path(path).read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Character-level dataset
# ──────────────────────────────────────────────────────────────────────


class CharLevelDataset(Dataset):
    """PyTorch Dataset for character-level language modeling.

    Takes the entire corpus as a single string, encodes it into a 1D
    integer array via a CharTokenizer, then stores it as one long
    sequence. Each call to __getitem__(i) extracts a contiguous chunk
    of length `block_size + 1` and returns:

        input  = chunk[:block_size]     (the first block_size tokens)
        target = chunk[1:block_size+1]  (shifted by one position)

    This way, for each input token the model learns to predict the
    *next* token (the target at the same index).

    Parameters
    ----------
    text : str
        Full corpus as a single string.
    tokenizer : Tokenizer
        A tokenizer with ``encode()`` / ``decode()`` methods (e.g.
        ``CharTokenizer`` or ``BPETokenizer``).
    block_size : int, optional (default=128)
        Context length (T). Each sample has shape (T,) for input and (T,) for target.
    device : torch.device | str | None, optional (default=None)
        If set, tensors are moved to this device on creation.
        If None, they stay on CPU (move to GPU in the training loop).
    """

    def __init__(
        self,
        text: str,
        tokenizer,
        block_size: int = 128,
        device: torch.device | str | None = None,
    ):
        super().__init__()

        self.block_size = block_size
        self.tokenizer = tokenizer
        ids = tokenizer.encode(text, add_special_tokens=False)
        self.data = torch.tensor(ids, dtype=torch.long)
        if device is not None:
            self.data = self.data.to(device)

    def __len__(self) -> int:
        """Return the total number of (input, target) pairs."""
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, target_ids) for sample at index idx."""
        chunk = self.data[idx : idx + self.block_size + 1]
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        return input_ids, target_ids


# ──────────────────────────────────────────────────────────────────────
# Convenience builder
# ──────────────────────────────────────────────────────────────────────


def create_dataloaders(
    corpus_name: str = "tinyshakespeare",
    tokenizer=None,
    block_size: int = 128,
    batch_size: int = 64,
    train_split: float = 0.9,
    data_dir: str | os.PathLike = "assets/",
    device: torch.device | str | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, CharLevelDataset, CharLevelDataset]:
    """One-stop convenience: load a text file, tokenize, split, and create DataLoaders.

    Parameters
    ----------
    corpus_name : str, optional (default="tinyshakespeare")
        Key in CORPUS_SOURCES.
    tokenizer : Tokenizer | None, optional
        If None, a ``CharTokenizer`` is built from the text by scanning
        all unique characters.
    block_size : int, optional (default=128)
        Context length.
    batch_size : int, optional (default=64)
        Batch size for both train and validation loaders.
    train_split : float, optional (default=0.9)
        Fraction of data for training (rest goes to validation).
    data_dir : str, optional (default="assets/")
        Where to cache the downloaded corpus file.
    device : torch.device | str | None, optional (default=None)
        Device for dataset tensors.
    num_workers : int, optional (default=0)
        DataLoader worker processes. 0 = load in main process.

    Returns
    -------
    train_loader : DataLoader
        Batched training data, shuffled.
    val_loader : DataLoader
        Batched validation data, NOT shuffled (evaluation order matters).
    train_dataset : CharLevelDataset
        Training split dataset (useful for debugging).
    val_dataset : CharLevelDataset
        Validation split dataset.
    """
    text = download_corpus(corpus_name, data_dir)

    if tokenizer is None:
        from core.transformer.embedding import CharTokenizer

        unique_chars = sorted(set(text))
        vocab = {c: i for i, c in enumerate(unique_chars)}
        tokenizer = CharTokenizer(vocab)

    split_idx = int(len(text) * train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_ds = CharLevelDataset(train_text, tokenizer, block_size, device)
    val_ds = CharLevelDataset(val_text, tokenizer, block_size, device)

    train_loader = DataLoader(
        train_ds, batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_ds, val_ds
