"""
core/transformer/transformer.py — Transformer building blocks (decoder-only GPT family).

Assembles RoPE positional encoding, feedforward network, decoder blocks,
and the full GPT model into a single module.

    tokens → [TokenEmbedding] → [GPTBlock × N] → [RMSNorm] → [lm_head] → logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
from core.transformer.attention import MultiHeadAttention, _create_causal_mask
from core.transformer.embedding import TokenEmbedding
from core.transformer.feedforward import FeedForward
from core.transformer.kv_cache import KVCache
from core.transformer.moe import MoEFFN
from core.transformer.normalization import RMSNorm
from core.transformer.rope import RotaryEmbedding
from core.transformer.sampling import sample

# ═══════════════════════════════════════════════════════════════════════
# GPT Decoder Block (Pre-Norm)
# ═══════════════════════════════════════════════════════════════════════


class GPTBlock(nn.Module):
    """GPT Decoder Block with Pre-Norm architecture.

    A single transformer decoder block consisting of:

        1. RMSNorm → Multi-Head Self-Attention → Residual
        2. RMSNorm → Position-wise FeedForward → Residual

    The attention uses a causal mask so each token can only attend to
    itself and previous tokens (autoregressive property).

    Parameters
    ----------
    d_model : int
        Feature dimension of the model.
    n_heads : int
        Number of attention heads.
    n_kv_heads : int | None, optional (default=None)
        Number of key/value heads for GQA. If None, defaults to n_heads (MHA).
    max_seq_len : int, optional (default=2048)
        Maximum sequence length for RoPE precomputation and causal mask.
    d_ff : int | None, optional (default=None)
        FeedForward hidden dimension. If None, defaults to 4 * d_model.
    dropout : float, optional (default=0.0)
        Dropout probability for attention weights.
    bias : bool, optional (default=True)
        Whether to use bias in linear projections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        max_seq_len: int = 2048,
        d_ff: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        # MoE parameters
        use_moe: bool = False,
        n_experts: int = 8,
        moe_k: int = 2,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.max_seq_len = max_seq_len
        self.use_moe = use_moe

        self.norm_1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model, n_heads, n_kv_heads=n_kv_heads, dropout=dropout, bias=bias
        )
        self.norm_2 = RMSNorm(d_model)

        d_ff = d_ff if d_ff is not None else 4 * d_model

        if use_moe:
            self.ff = MoEFFN(d_model, d_ff, n_experts, moe_k, bias)
        else:
            self.ff = FeedForward(d_model, d_ff, bias)

        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Process one transformer block.

        Always returns ``(output, (k, v))`` — the caller decides whether
        to use (k, v) for cache storage or discard them with ``_``.

        Parameters
        ----------
        x : (batch_size, seq_len, d_model)
            Input tensor.
        mask : (seq_len, seq_len) | None, optional
            Causal attention mask. If None, a causal mask is created
            automatically based on seq_len (unless past_kv is set).
        past_kv : tuple[torch.Tensor, torch.Tensor] | None, optional
            Cached (key, value) from previous decode steps for this layer.
            Passed directly to MultiHeadAttention.forward().

        Returns
        -------
        (out, (k, v)) : tuple
            out : (batch_size, seq_len, d_model) — block output.
            (k, v) : the key/value tensors in multi-head format,
                     returned by MultiHeadAttention.
        """
        if mask is None:
            if past_kv is None:
                # Normal forward (training / prefill): create a causal mask
                seq_len = x.size(1)
                mask = _create_causal_mask(seq_len, device=x.device)
            else:
                # Decode step: no mask needed because Q is only the latest
                # token (seq_len=1) and the cache only contains past +
                # current positions, so causality is automatic.
                pass

        residual = x
        x = self.norm_1(x)
        x, kv = self.attn(x, x, x, mask, self.rope, past_kv)
        x = residual + x

        residual = x
        x = self.norm_2(x)
        x = self.ff(x)
        x = residual + x

        return (x, kv)


# ═══════════════════════════════════════════════════════════════════════
# GPT Language Model
# ═══════════════════════════════════════════════════════════════════════


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
    n_kv_heads : int | None, optional (default=None)
        Number of key/value heads for GQA. If None, defaults to n_heads (MHA).
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
        n_kv_heads: int | None = None,
        max_seq_len: int = 2048,
        d_ff: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        tie_weights: bool = False,
        # MoE parameters
        use_moe: bool = False,
        n_experts: int = 8,
        moe_k: int = 2,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.max_seq_len = max_seq_len
        self.use_moe = use_moe

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                GPTBlock(
                    d_model,
                    n_heads,
                    n_kv_heads=n_kv_heads,
                    max_seq_len=max_seq_len,
                    d_ff=d_ff,
                    dropout=dropout,
                    bias=bias,
                    use_moe=use_moe,
                    n_experts=n_experts,
                    moe_k=moe_k,
                )
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
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Forward pass through the full GPT model.

        Parameters
        ----------
        tokens : (batch_size, seq_len)
            Input token IDs (integers).
        mask : (seq_len, seq_len) | None, optional
            Causal attention mask. If None, each GPTBlock will create
            its own causal mask based on seq_len.
        kv_cache : KVCache | None, optional
            The KV cache instance (for autoregressive decoding).

            * ``kv_cache is None`` — normal forward pass (training mode).
              All K, V computed and discarded.

            * ``kv_cache is not None`` — caching mode.
              Each layer retrieves ``past_kv`` from the cache, computes
              new K, V, and stores the updated (K, V) back.

              During prefill (first call): ``past_kv`` is ``None`` for every
              layer since nothing is cached yet; full K, V are computed and
              stored.

              During decode (subsequent calls with one token): ``past_kv``
              contains previously cached K, V; only the new token's K, V
              are computed, concatenated with the cache, and re-stored.

        Returns
        -------
        logits : (batch_size, seq_len, vocab_size)
            Unnormalised scores for each token at each position.
        """
        x = self.token_embedding(tokens)
        for i, block in enumerate(self.blocks):
            if kv_cache is not None:
                past_kv = kv_cache.get(i)  # None during prefill
                x, (k_out, v_out) = block(x, mask=mask, past_kv=past_kv)
                kv_cache.update(i, k_out, v_out)
            else:
                x, _ = block(x, mask=mask)
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

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.1,
        eos_token_id: int | None = None,
        use_kv_cache: bool = True,
    ) -> torch.LongTensor:
        """Autoregressively generate tokens from a prompt.

        Two modes — same loop structure, different forward strategy:

        **KV Cache mode** (``use_kv_cache=True``, default):
            Prefills the cache with the full prompt, then feeds one token
            at a time. Each decode step is O(n) instead of O(n²).

        **Naive mode** (``use_kv_cache=False``):
            Re-runs the full sequence through the model at every step.
            Useful as a correctness baseline.

        Parameters
        ----------
        prompt : (batch, prompt_len) or (prompt_len,)
            Initial token IDs to start generation from.
        max_new_tokens : int, optional (default=100)
            Maximum number of new tokens to generate.
        temperature : float, optional (default=1.0)
            Sampling temperature. 0 = argmax, 1 = scaled, higher = more random.
        top_k : int | None, optional (default=None)
            Top-k filtering threshold.
        top_p : float | None, optional (default=None)
            Nucleus (top-p) filtering threshold.
        repetition_penalty : float, optional (default=1.1)
            Penalty factor for already-generated tokens (CTRL paper style).
            1.0 = disabled. Typical range: 1.05 - 1.5.
            Higher values = stronger discouragement of repetition.
            Default 1.1 was empirically found optimal for this model size.
        eos_token_id : int | None, optional (default=None)
            If set, generation stops when this token is generated.
        use_kv_cache : bool, optional (default=True)
            If True, uses KV cache for efficient decoding.
            Falls back to full-sequence re-forward if False.

        Returns
        -------
        output : (batch, prompt_len + num_generated)
            Full sequence including prompt and generated tokens.
        """
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)

        cache = KVCache() if use_kv_cache else None

        output = prompt.clone()

        full_mask = _create_causal_mask(self.max_seq_len, device=output.device)

        logits = self.forward(
            output,
            mask=full_mask[: output.size(1), : output.size(1)],
            kv_cache=cache,
        )

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            next_token = sample(
                next_logits,
                temperature,
                top_k,
                top_p,
                repetition_penalty=repetition_penalty,
                past_tokens=output,
            )
            output = torch.cat([output, next_token.unsqueeze(-1)], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).any():
                break
            if output.size(1) >= self.max_seq_len:
                break

            if use_kv_cache:
                logits = self.forward(output[:, -1:], kv_cache=cache)
            else:
                seq_len = output.size(1)
                mask = full_mask[:seq_len, :seq_len]
                logits = self.forward(output, mask=mask)

        if cache is not None:
            cache.reset()

        return output

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

    @property
    def aux_loss(self) -> torch.Tensor:
        """Sum of auxiliary load-balancing losses from all MoE blocks.

        Returns a zero scalar when MoE is not enabled.
        """
        if not self.use_moe:
            return torch.tensor(0.0)
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.blocks:
            if hasattr(block.ff, "aux_loss"):
                total = total + block.ff.aux_loss
        return total
