"""
core/cv/vit.py — Vision Transformer (ViT) building blocks.

ViT (Dosovitskiy et al., 2020) applies a standard Transformer directly to
image patches, treating each patch as a "token" in a sequence.

    image → [PatchEmbed] → [GPTBlock × N] → [CLS] → [classification head]

PatchEmbed
    Splits an image into fixed-size patches, linearly projects each patch
    to a D-dimensional embedding, prepends a [CLS] token, and adds learnable
    position embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv2d import Conv2d


class PatchEmbed(nn.Module):
    """Image → patch embedding with [CLS] token and position embedding.

    The forward pass proceeds in three steps:

        1. **Patch + Project**: Apply a Conv2d with ``kernel_size=patch_size,
           stride=patch_size``.  This simultaneously extracts non-overlapping
           patches and linearly projects each to ``embed_dim`` channels.

        2. **[CLS] Token**: A learnable ``cls_token`` is prepended to the
           patch sequence (after flattening).  This token's output
           representation is used for classification.

        3. **Position Embedding**: A learnable ``pos_embed`` (one vector per
           sequence position) is added element-wise.

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

    Shape
    -----
    Input:  (N, in_channels, img_size, img_size)
    Output: (N, num_patches + 1, embed_dim)

    where ``num_patches = (img_size // patch_size) ** 2``.

    Example
    -------
    >>> embed = PatchEmbed(img_size=224, patch_size=16, embed_dim=768)
    >>> x = torch.randn(4, 3, 224, 224)
    >>> y = embed(x)
    >>> y.shape
    torch.Size([4, 197, 768])   # 196 patches + 1 [CLS] token
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embedding sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (N, in_channels, H, W).
            ``H`` and ``W`` must equal ``img_size``.

        Returns
        -------
        torch.Tensor
            Patch embedding of shape (N, num_patches + 1, embed_dim).
        """
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        return x

    def extra_repr(self) -> str:
        """Return a formatted string for ``print(patch_embed)``."""
        return (
            f"img_size={self.img_size}, patch_size={self.patch_size}, "
            f"embed_dim={self.embed_dim}, num_patches={self.num_patches}"
        )
