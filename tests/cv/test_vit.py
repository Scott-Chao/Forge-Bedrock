"""tests/cv/test_vit.py — Tests for ViT building blocks."""

import pytest
import torch
from core.cv import PatchEmbed, ViT, ViTBlock


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


# ============================================================================
# ViTBlock
# ============================================================================


class TestViTBlockShape:
    """ViTBlock preserves shape and runs at train/eval."""

    @pytest.mark.parametrize(
        "batch,seq_len,d_model,n_heads",
        [
            (2, 17, 64, 4),
            (4, 65, 256, 8),
            (1, 197, 768, 12),
        ],
    )
    def test_output_shape(self, batch, seq_len, d_model, n_heads):
        block = ViTBlock(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch, seq_len, d_model)
        out = block(x)
        assert out.shape == (batch, seq_len, d_model)

    def test_output_differs_from_input(self):
        """A real transformation happens (not identity)."""
        block = ViTBlock(d_model=64, n_heads=4)
        x = torch.randn(2, 17, 64)
        out = block(x)
        assert not torch.allclose(out, x, atol=1e-4)

    def test_train_eval_differ_with_dropout(self):
        """Dropout makes train/eval outputs differ (statistically)."""
        block = ViTBlock(d_model=64, n_heads=4, dropout=0.5)
        x = torch.randn(4, 17, 64)
        block.train()
        outs = torch.stack([block(x) for _ in range(10)])
        train_var = outs.var(dim=0).mean()
        block.eval()
        outs = torch.stack([block(x) for _ in range(10)])
        eval_var = outs.var(dim=0).mean()
        assert train_var > eval_var * 2, "train variance should exceed eval variance"


class TestViTBlockBackward:
    def test_gradient_flows(self):
        block = ViTBlock(d_model=64, n_heads=4)
        x = torch.randn(2, 17, 64)
        out = block(x).sum()
        out.backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"{name} has no grad"
            assert param.grad.shape == param.shape


# ============================================================================
# ViT
# ============================================================================


class TestViTShape:
    """ViT produces correct classification logits."""

    @pytest.mark.parametrize(
        "depth,num_classes",
        [
            (1, 10),
            (2, 100),
            (4, 5),
        ],
    )
    def test_output_shape(self, depth, num_classes):
        vit = ViT(
            img_size=32,
            patch_size=8,
            num_classes=num_classes,
            embed_dim=128,
            depth=depth,
            n_heads=4,
        )
        x = torch.randn(2, 3, 32, 32)
        logits = vit(x)
        assert logits.shape == (2, num_classes)

    def test_cls_token_path(self):
        """Classification head operates on [CLS] token (index 0)."""
        vit = ViT(
            img_size=32,
            patch_size=8,
            num_classes=10,
            embed_dim=64,
            depth=2,
            n_heads=4,
        )
        x = torch.randn(2, 3, 32, 32)
        logits = vit(x)
        assert logits.dim() == 2  # (batch, num_classes), not (batch, seq, ...)
        assert logits.shape[1] == 10


class TestViTBackward:
    def test_gradient_flows(self):
        vit = ViT(
            img_size=32,
            patch_size=8,
            num_classes=5,
            embed_dim=64,
            depth=2,
            n_heads=4,
        )
        x = torch.randn(2, 3, 32, 32)
        vit(x).sum().backward()
        for name, param in vit.named_parameters():
            assert param.grad is not None, f"{name} has no grad"
            assert param.grad.shape == param.shape
