"""
core/transformer/sampling.py — Token sampling strategies for autoregressive generation.

Given raw logits from the model, sampling strategies convert them into
the next token ID. Strategies range from deterministic (argmax) to
stochastic (temperature, top-k, top-p).

The typical pipeline is:

    logits → [Temperature scaling] → [Top-k filter] → [Top-p filter]
           → [softmax → sample] → next_token_id
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    past_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample a single token ID from logits with optional filtering.

    The pipeline is:

        logits → [repetition penalty] → [temperature scaling]
               → [top-k filter] → [top-p filter]
               → [softmax → multinomial] → next_token_id

    Parameters
    ----------
    logits : (vocab_size,) or (batch, vocab_size)
        Raw logits from the model's lm_head (for the last position).
    temperature : float, optional (default=1.0)
        Scaling factor applied before softmax.
        Lower = sharper (more greedy), higher = more random.
        Set to 0.0 for greedy argmax sampling.
    top_k : int | None, optional (default=None)
        If set, only the top-k highest logits are kept (others set to -inf).
    top_p : float | None, optional (default=None)
        If set, only the smallest set of tokens whose cumulative
        probability exceeds top_p are kept (nucleus sampling).
    repetition_penalty : float, optional (default=1.0)
        Penalty factor for already-generated tokens (from CTRL paper).
        Applied BEFORE temperature scaling.
        - 1.0 = disabled (no penalty)
        - > 1.0 = penalise repeated tokens (typical: 1.05 - 1.5)
        - < 1.0 = encourage repetition (not typically used)
    past_tokens : (batch, seq_len) or (seq_len,) | None, optional
        Token IDs that have already been generated. Tokens present in
        this sequence have their logits divided by ``repetition_penalty``.

    Returns
    -------
    token_id : (,) or (batch,)
        Sampled token ID(s).
    """
    # ── Repetition penalty: divide logits of seen tokens ────────────────
    if repetition_penalty != 1.0 and past_tokens is not None:
        if logits.dim() == 1:
            # Single sequence: (vocab_size,)
            unique_tok = torch.unique(past_tokens)
            logits[unique_tok] /= repetition_penalty
        else:
            # Batched: (batch, vocab_size)
            for b in range(logits.size(0)):
                unique_tok = torch.unique(past_tokens[b])
                logits[b, unique_tok] /= repetition_penalty

    if temperature == 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    if top_k is not None:
        values, _ = logits.topk(top_k, dim=-1)
        threshold = values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None:
        sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumsum = probs.cumsum(dim=-1)

        mask = cumsum - probs > top_p
        sorted_logits.masked_fill(mask, float("-inf"))

        logits = sorted_logits.scatter(dim=-1, index=sorted_indices, src=sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Greedy sampling: return the token with the highest logit.

    Parameters
    ----------
    logits : (vocab_size,) or (batch, vocab_size)

    Returns
    -------
    token_id : (,) or (batch,)
        Token with maximum logit value.
    """
    return sample(logits, temperature=0.0)


def sample_with_temperature(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Sample from a temperature-scaled softmax distribution.

    Parameters
    ----------
    logits : (vocab_size,) or (batch, vocab_size)
    temperature : float (default=1.0)
        Temperature scaling factor. Must be > 0.

    Returns
    -------
    token_id : (,) or (batch,)
    """
    return sample(logits, temperature=temperature)


def sample_top_k(logits: torch.Tensor, k: int = 50) -> torch.Tensor:
    """Top-k sampling: filter to top k logits, then sample.

    Parameters
    ----------
    logits : (vocab_size,) or (batch, vocab_size)
    k : int (default=50)
        Number of highest-probability tokens to keep.

    Returns
    -------
    token_id : (,) or (batch,)
    """
    return sample(logits, top_k=k)


def sample_top_p(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """Nucleus (top-p) sampling: filter to smallest set with cum prob > p.

    Parameters
    ----------
    logits : (vocab_size,) or (batch, vocab_size)
    p : float (default=0.9)
        Cumulative probability threshold.

    Returns
    -------
    token_id : (,) or (batch,)
    """
    return sample(logits, top_p=p)
