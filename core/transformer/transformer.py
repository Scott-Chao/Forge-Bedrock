"""
core/transformer/transformer.py — Transformer building blocks (decoder-only GPT family).

Assembles attention mechanisms, RoPE positional encoding, feedforward network,
decoder blocks, and the full GPT model into a single module. Following the
organization of torch.nn.modules.transformer, all transformer-specific
components live in one file.

    tokens → [TokenEmbedding] → [GPTBlock × N] → [RMSNorm] → [lm_head] → logits
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.transformer.embedding import TokenEmbedding
from core.transformer.kv_cache import KVCache
from core.transformer.normalization import RMSNorm
from core.transformer.sampling import sample

# ═══════════════════════════════════════════════════════════════════════
# Scaled Dot-Product Attention
# ═══════════════════════════════════════════════════════════════════════


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Shapes
    ------
    query:  (batch_size, ..., seq_len, d_k)
    key:    (batch_size, ..., seq_len, d_k)
    value:  (batch_size, ..., seq_len, d_v)   — d_v is often == d_k
    mask:   (seq_len, seq_len) or broadcastable to (..., seq_len, seq_len)

    Returns
    -------
    output: (batch_size, ..., seq_len, d_v)
    """
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    return attention_weights @ value


def _create_causal_mask(
    seq_len: int, device: torch.device | None = None
) -> torch.Tensor:
    """Create a causal (upper-triangular) attention mask.

    Returns a boolean mask of shape (seq_len, seq_len) where:
        mask[i, j] = True   if i >= j  (allowed: attend to current & past)
        mask[i, j] = False  if i < j   (forbidden: attend to future)
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
    return ~mask


# ═══════════════════════════════════════════════════════════════════════
# Rotary Positional Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════


def precompute_freqs_cis(
    d_model: int, max_seq_len: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE.

    For each pair index i in [0, d_model/2):

        theta_i = base ** (-2 * i / d_model)

    For each position m in [0, max_seq_len):

        cos[m] = [cos(mθ₀), cos(mθ₁), ..., cos(mθ_{d/2-1}),
                  cos(mθ₀), cos(mθ₁), ..., cos(mθ_{d/2-1})]
        sin[m] = [sin(mθ₀), sin(mθ₁), ..., sin(mθ_{d/2-1}),
                  sin(mθ₀), sin(mθ₁), ..., sin(mθ_{d/2-1})]

    Parameters
    ----------
    d_model : int
        Feature dimension (must be even).
    max_seq_len : int
        Maximum sequence length to precompute.
    base : float
        Base frequency (default 10000.0, same as original Transformer).

    Returns
    -------
    cos : (max_seq_len, d_model)
    sin : (max_seq_len, d_model)
    """
    theta = base ** (-torch.arange(0, d_model, 2).float() / d_model)
    m = torch.arange(max_seq_len)
    angles = torch.outer(m, theta)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split-half swap: rotate_half(x)[i] = -x[i+d/2], rotate_half(x)[i+d/2] = x[i].

        rotate_half([a, b, c, d, e, f, g, h]) = [-e, -f, -g, -h, a, b, c, d]

    Used in the efficient formula: x_rotated = x * cos + rotate_half(x) * sin
    """
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE using the efficient rotate_half formula.

        x_rotated = x * cos + rotate_half(x) * sin

    Parameters
    ----------
    x : (..., seq_len, d_model)
        Q or K tensor.
    cos : (max_seq_len, d_model)
        Precomputed cosines.
    sin : (max_seq_len, d_model)
        Precomputed sines.

    Returns
    -------
    out : (..., seq_len, d_model)
        Rotated tensor, same shape as input.
    """
    seq_len = x.size(-2)
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(torch.nn.Module):
    """Rotary Positional Embedding — convenient nn.Module wrapper.

    Usage:
        rope = RotaryEmbedding(d_model)
        q_rope = rope(q)   # applies RoPE to Q
        k_rope = rope(k)   # applies RoPE to K
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        cos, sin = precompute_freqs_cis(d_model, max_seq_len, base)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        return apply_rotary_emb(x, self.cos[:seq_len], self.sin[:seq_len])


# ═══════════════════════════════════════════════════════════════════════
# Multi-Head Attention
# ═══════════════════════════════════════════════════════════════════════


class MultiHeadAttention(torch.nn.Module):
    """Multi-Head / Grouped-Query Attention.

    Supports both standard Multi-Head Attention (MHA) and Grouped-Query
    Attention (GQA, Ainslie et al., 2023).

    **MHA mode** (default, ``n_kv_heads=None``):
        Each query head has its own K, V projection. The KV cache stores
        one (K, V) pair per head.

            MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
            where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    **GQA mode** (``n_kv_heads < n_heads``):
        Multiple query heads **share** a single K, V pair. The KV cache
        stores only ``n_kv_heads`` pairs — a direct memory saving.

            n_groups = n_heads // n_kv_heads
            K = K.repeat_interleave(n_groups, dim=1)   # expand at runtime
            V = V.repeat_interleave(n_groups, dim=1)   # expand at runtime

    Important
    ---------
    In a decoder-only Transformer (GPT), Q = K = V = x (the same input).
    This is called **self-attention**: the tokens attending to themselves.
    The projections still exist; the input simply passes through three
    separate linear layers to produce Q, K, V.

    Parameters
    ----------
    d_model : int
        Input and output feature dimension.
    n_heads : int
        Number of parallel attention heads. Must divide d_model evenly.
    n_kv_heads : int | None, optional (default=None)
        Number of key/value heads (for GQA). If None, defaults to n_heads
        (standard MHA). Must divide n_heads evenly.
    dropout : float, optional (default=0.0)
        Dropout rate applied to attention weights. Not strictly necessary
        for a minimal model, but standard practice.
    bias : bool, optional (default=True)
        Whether to include bias terms in the linear projections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        if n_kv_heads is None:
            n_kv_heads = n_heads  # default: standard MHA

        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads  # dimension per head

        kv_dim = self.d_k * n_kv_heads
        self.w_q = torch.nn.Linear(d_model, d_model, bias=bias)
        self.w_k = torch.nn.Linear(d_model, kv_dim, bias=bias)
        self.w_v = torch.nn.Linear(d_model, kv_dim, bias=bias)
        self.w_o = torch.nn.Linear(d_model, d_model, bias=bias)

        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: RotaryEmbedding | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for multi-head attention, with optional KV cache.

        Always returns ``(output, (k_new, v_new))`` — the caller decides
        whether to use the new K, V (for cache storage) or discard them.

        Parameters
        ----------
        query : (batch_size, seq_len, d_model)
        key   : (batch_size, seq_len, d_model)
        value : (batch_size, seq_len, d_model)
        mask  : (seq_len, seq_len) or broadcastable — True = allowed
        rope  : RotaryEmbedding | None, optional
            If provided, applies rotary position encoding to Q and K
            after reshaping into multi-head form.
        past_kv : tuple[torch.Tensor, torch.Tensor] | None, optional
            Cached (key, value) from previous decode steps, each shaped
            (batch, n_heads, cached_len, d_k). When provided, the input
            ``key`` / ``value`` contain ONLY the new token(s); they are
            concatenated with the cached tensors along dim=-2 *before*
            the attention computation.

        Returns
        -------
        (output, (k_new, v_new)) : tuple
            output  : (batch_size, seq_len_q, d_model) — attention result.
            k_new   : (batch, n_heads, seq_len_new, d_k) — new key projection
                      (after RoPE if applicable), *without* cached content.
            v_new   : (batch, n_heads, seq_len_new, d_k) — new value projection.
            The caller (KVCache) is responsible for appending (k_new, v_new)
            to the cache — this avoids double-concatenation bugs.
        """
        batch = query.size(0)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = q.reshape(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.reshape(batch, -1, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = v.reshape(batch, -1, self.n_kv_heads, self.d_k).transpose(1, 2)

        # ── RoPE: rotate Q and K in their per-head space ──────────────
        if rope is not None:
            q = rope(q)
            k = rope(k)

        # ── KV Cache: keep new K, V reference before expansion ──
        k_new, v_new = k, v

        if past_kv is not None:
            k = torch.cat([past_kv[0], k_new], dim=-2)
            v = torch.cat([past_kv[1], v_new], dim=-2)

        # ── GQA: expand K, V to match Q head count by repeating groups ──
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        out = scaled_dot_product_attention(q, k, v, mask)

        if self.dropout is not None:
            out = self.dropout(out)

        out = out.transpose(1, 2).reshape(batch, -1, self.d_model)

        out = self.w_o(out)

        return (out, (k_new, v_new))


# ═══════════════════════════════════════════════════════════════════════
# Position-wise FeedForward Network
# ═══════════════════════════════════════════════════════════════════════


class FeedForward(nn.Module):
    """Position-wise FeedForward Network.

    A two-layer MLP with ReLU activation:

        x -> Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model)

    Applied independently to each position in the sequence
    (no communication between tokens here).

    Parameters
    ----------
    d_model : int
        Input and output feature dimension.
    d_ff : int
        Hidden (intermediate) dimension. Typically 4 * d_model.
    bias : bool, optional (default=True)
        Whether to use bias in both linear layers.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, bias: bool = True):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.d_model = d_model
        self.d_ff = d_ff

        self.w_1 = torch.nn.Linear(d_model, d_ff, bias=bias)
        self.relu = torch.nn.ReLU()
        self.w_2 = torch.nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feedforward network.

        Parameters
        ----------
        x : (batch_size, seq_len, d_model)
            Input tensor.

        Returns
        -------
        out : (batch_size, seq_len, d_model)
            Output after expansion, ReLU, and compression.
        """
        hidden = self.w_1(x)
        hidden = self.relu(hidden)
        out = self.w_2(hidden)
        return out


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
            from core.transformer.moe import MoEFFN  # lazy import avoids circular ref

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
                mask = self._create_attn_mask(seq_len, device=x.device)
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

    def _create_attn_mask(
        self, seq_len: int, device: torch.device | None = None
    ) -> torch.Tensor:
        """Create a causal attention mask for the given sequence length."""
        return _create_causal_mask(seq_len, device)


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
