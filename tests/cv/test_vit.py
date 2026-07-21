"""tests/cv/test_vit.py — Tests for ViT building blocks."""

import pytest
import torch
from core.cv import PatchEmbed


class TestInit:
    def test_parameter_shapes(self):
        embed = PatchEmbed(img_size=224, patch_size=16, embed_dim=768)
        assert embed.num_patches == 196
        assert embed.proj.weight.shape == (768, 3, 16, 16)
        assert embed.cls_token.shape == (1, 1, 768)
        assert embed.pos_embed.shape == (1, 197, 768)
        assert len(list(embed.parameters())) == 4  # weight + bias + cls + pos

    def test_repr(self):
        r = repr(PatchEmbed(img_size=224, patch_size=16, embed_dim=512))
        assert "img_size=224" in r and "num_patches=196" in r and "embed_dim=512" in r


class TestForward:
    @pytest.mark.parametrize(
        "batch_size,img_size,patch_size,embed_dim,seq_len",
        [
            (4, 224, 16, 768, 197),  # ViT-Base
            (1, 224, 32, 768, 50),
            (2, 128, 16, 512, 65),
            (8, 64, 16, 256, 17),
        ],
    )
    def test_output_shape(self, batch_size, img_size, patch_size, embed_dim, seq_len):
        embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim
        )
        out = embed(torch.randn(batch_size, 3, img_size, img_size))
        assert out.shape == (batch_size, seq_len, embed_dim)

    def test_cls_token_is_first(self):
        """[CLS] token is shared across the batch (same learned parameter)."""
        out = PatchEmbed(img_size=64, patch_size=16, embed_dim=256)(
            torch.randn(4, 3, 64, 64)
        )
        assert torch.allclose(out[0, 0], out[1, 0], atol=1e-6)


class TestBackward:
    def test_all_params_get_grad_with_correct_shape(self):
        embed = PatchEmbed(img_size=64, patch_size=16, embed_dim=256)
        embed(torch.randn(2, 3, 64, 64)).sum().backward()
        for param in embed.parameters():
            assert param.grad is not None, f"{param.shape} has no grad"
            assert param.grad.shape == param.shape


class TestEdgeCases:
    @pytest.mark.parametrize(
        "img_size,patch_size,embed_dim,in_channels,seq_len",
        [
            (8, 4, 64, 3, 5),  # minimal
            (32, 8, 128, 1, 17),  # single channel
            (32, 16, 1, 3, 5),  # embed_dim=1
        ],
    )
    def test_nonstandard_configs(
        self, img_size, patch_size, embed_dim, in_channels, seq_len
    ):
        embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_channels=in_channels,
        )
        out = embed(torch.randn(2, in_channels, img_size, img_size))
        assert out.shape == (2, seq_len, embed_dim)
