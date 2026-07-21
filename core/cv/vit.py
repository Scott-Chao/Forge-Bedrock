"""
core/cv/vit.py — Vision Transformer (ViT) from pure PyTorch components.

ViT (Dosovitskiy et al., 2020) applies a standard Transformer directly
to image patches:

    image → [PatchEmbed] → [ViTBlock × N] → [LayerNorm] → [CLS] → [Linear]

The architecture differs from GPT in several ways:
    * No causal mask — bidirectional self-attention over all patches
    * No RoPE — learnable absolute position embeddings instead
    * LayerNorm (not RMSNorm)
    * GELU activation (not ReLU) in the FeedForward network

All components use PyTorch native modules rather than reusing the
transformer package, since the design constraints differ significantly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ============================================================================
# PatchEmbed — image → token sequence
# ============================================================================


class PatchEmbed(nn.Module):
    """Image → patch embedding with [CLS] token and position embedding.

    Three-step pipeline::

        1. Conv2d(kernel_size=P, stride=P) extracts non-overlapping
           patches and projects each to a D-dimensional vector.
        2. A learnable [CLS] token is prepended to the sequence.
        3. A learnable position embedding is added element-wise.

    Dropout is applied **after** adding the position embedding (but
    **not** to the [CLS] token alone — the whole sequence is dropped
    uniformly), following the ViT paper.

    Parameters
    ----------
    img_size : int, default=224
        Height and width of the input image (assumed square).
    patch_size : int, default=16
        Height and width of each patch.
    in_channels : int, default=3
        Number of input image channels.
    embed_dim : int, default=768
        Dimension of the patch embedding (D).
    dropout : float, default=0.0
        Dropout probability applied after position embedding.

    Shape
    -----
    Input:  (N, in_channels, img_size, img_size)
    Output: (N, num_patches + 1, embed_dim)

    where ``num_patches = (img_size // patch_size) ** 2``.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embedding sequence."""
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.drop(x)
        return x

    def extra_repr(self) -> str:
        """Return a formatted string for ``print(patch_embed)``."""
        return (
            f"img_size={self.img_size}, patch_size={self.patch_size}, "
            f"embed_dim={self.embed_dim}, num_patches={self.num_patches}"
        )


# ============================================================================
# ViTBlock — one encoder block
# ============================================================================


class ViTBlock(nn.Module):
    """ViT encoder block — bidirectional self-attention, no RoPE.

    Pre-Norm design::

        LayerNorm → Multi-Head Self-Attention → Residual
        → LayerNorm → FeedForward (GELU) → Residual

    Key differences from GPTBlock:
        * **No causal mask** — attention is fully bidirectional
        * **No RoPE** — position is handled by ``PatchEmbed.pos_embed``
        * **LayerNorm** instead of RMSNorm
        * **GELU** instead of ReLU in the FFN

    Parameters
    ----------
    d_model : int
        Feature dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int | None, default=None
        FeedForward hidden dimension.  If None, defaults to ``4 * d_model``.
    dropout : float, default=0.0
        Dropout probability — applied in three places: attention weights
        (inside ``nn.MultiheadAttention``), after GELU in the FFN, and
        after the position embedding in ``PatchEmbed``.
    bias : bool, default=True
        Whether to use bias in linear projections.

    Shape
    -----
    Input:  (N, seq_len, d_model)
    Output: (N, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.norm_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, bias=bias, batch_first=True
        )
        self.norm_2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one ViT encoder block."""
        residual = x
        x = self.norm_1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = residual + x

        residual = x
        x = self.norm_2(x)
        x = self.ff(x)
        x = residual + x
        return x


# ============================================================================
# ViT — full Vision Transformer for classification
# ============================================================================


class ViT(nn.Module):
    """Vision Transformer for image classification.

    Full pipeline::

        image → PatchEmbed → ViTBlock × N → LayerNorm → [CLS] → Linear(head)

    The [CLS] token's output (index 0 along the sequence dimension)
    serves as the global image representation fed to the classification
    head.

    Parameters
    ----------
    img_size : int, default=224
        Input image spatial size.
    patch_size : int, default=16
        Patch size.
    in_channels : int, default=3
        Input image channels.
    num_classes : int, default=1000
        Number of output classes.
    embed_dim : int, default=768
        Patch embedding dimension. (ViT-Base: 768, ViT-Large: 1024, ViT-Huge: 1280)
    depth : int, default=12
        Number of ViTBlocks. (ViT-Base: 12, ViT-Large: 24, ViT-Huge: 32)
    n_heads : int, default=12
        Number of attention heads. (ViT-Base: 12, ViT-Large: 16, ViT-Huge: 16)
    d_ff : int | None, default=None
        FeedForward hidden dimension.  If None, defaults to ``4 * embed_dim``.
    dropout : float, default=0.0
        Dropout probability — applied after position embedding, in
        attention weights, and in the FFN after GELU.
    bias : bool, default=True
        Whether to use bias in linear projections.

    Shape
    -----
    Input:  (N, in_channels, img_size, img_size)
    Output: (N, num_classes)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        n_heads: int = 12,
        d_ff: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_channels, embed_dim, dropout
        )
        self.blocks = nn.ModuleList(
            [ViTBlock(embed_dim, n_heads, d_ff, dropout, bias) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: image → patches → transformer → [CLS] → logits."""
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls = x[:, 0]
        logits = self.head(cls)
        return logits

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following ViT convention: N(0, 0.02)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def extra_repr(self) -> str:
        """Return a formatted string for ``print(vit)``."""
        total = sum(p.numel() for p in self.parameters())
        return (
            f"ViT-{self.depth}/embed_dim={self.embed_dim}, "
            f"num_classes={self.num_classes}, "
            f"params={total / 1e6:.1f}M"
        )
