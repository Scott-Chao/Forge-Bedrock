"""
core/transformer/checkpoint.py — Model checkpoint loading.

Provides ``load_checkpoint`` to restore a trained GPT model and its
BPE tokenizer from disk (the files written by the ``train_gpt.ipynb``
save cell).
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_checkpoint(
    model_dir: str | Path = "models/",
    device: torch.device | str | None = None,
) -> tuple:
    """Load a trained GPT checkpoint with its BPE tokenizer.

    Loads the model configuration, tokenizer vocabulary, and trained
    weights from a directory previously saved by the training notebook.

    Parameters
    ----------
    model_dir : str | os.PathLike, optional (default="models/")
        Directory containing ``training_config.pt``, ``bpe_tokenizer.pt``,
        and ``gpt.pt``.
    device : torch.device | str | None, optional (default=None)
        Device to load the model onto. If None, stays on CPU.

    Returns
    -------
    (model, tokenizer, config) : tuple[GPT, BPETokenizer, dict]
        model : GPT with loaded weights, in eval mode.
        tokenizer : BPETokenizer with saved vocab and merges.
        config : dict of training configuration.
    """
    from core.transformer.embedding import BPETokenizer
    from core.transformer.transformer import GPT

    model_dir = Path(model_dir)

    config = torch.load(
        model_dir / "training_config.pt", map_location="cpu", weights_only=True
    )

    tokenizer = BPETokenizer.from_pretrained(model_dir / "bpe_tokenizer.pt")

    model = GPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        n_kv_heads=config.get("n_kv_heads"),
        max_seq_len=config["block_size"],
        d_ff=config["d_ff"],
        dropout=0.0,
        use_moe=config.get("use_moe", False),
        n_experts=config.get("n_experts", 8),
        moe_k=config.get("moe_k", 2),
    )
    model.load_state_dict(
        torch.load(
            model_dir / "gpt.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()

    if device is not None:
        model = model.to(device)

    return model, tokenizer, config
