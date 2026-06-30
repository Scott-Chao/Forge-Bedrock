"""
tests/transformer/test_embedding.py — Tests for TokenEmbedding and CharTokenizer.
"""

import pytest
import torch
from core.transformer.embedding import (
    CharTokenizer,
    TokenEmbedding,
    _build_default_vocab,
)


class TestBuildDefaultVocab:
    """Tests for the default vocabulary builder."""

    def test_returns_dict(self):
        """_build_default_vocab should return a non-empty dict."""
        vocab = _build_default_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) > 0, "vocab should not be empty"

    def test_contains_special_tokens(self):
        """Vocabulary should have special tokens."""
        vocab = _build_default_vocab()
        for token in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            assert token in vocab, f"vocab missing {token}"

    def test_unique_indices(self):
        """All character indices should be unique."""
        vocab = _build_default_vocab()
        indices = list(vocab.values())
        assert len(indices) == len(set(indices)), "indices must be unique"

    def test_includes_basic_chars(self):
        """Should include lowercase letters and common punctuation."""
        vocab = _build_default_vocab()
        assert "a" in vocab
        assert "z" in vocab
        assert "A" in vocab
        assert "Z" in vocab
        assert " " in vocab, "space should be in vocab"
        assert "." in vocab or "!" in vocab, "punctuation should be in vocab"

    def test_size_around_70_to_90(self):
        """Vocab size should be in reasonable range for char-level."""
        vocab = _build_default_vocab()
        assert 60 <= len(vocab) <= 100, (
            f"vocab size {len(vocab)} outside expected range [60, 100]"
        )


class TestCharTokenizer:
    """Tests for the character-level tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        return CharTokenizer()

    def test_construction(self, tokenizer):
        """Tokenizer should have expected attributes."""
        assert hasattr(tokenizer, "vocab")
        assert hasattr(tokenizer, "itos")
        assert hasattr(tokenizer, "pad_id")
        assert hasattr(tokenizer, "unk_id")
        assert hasattr(tokenizer, "bos_id")
        assert hasattr(tokenizer, "eos_id")

    def test_vocab_size_property(self, tokenizer):
        """vocab_size should match the vocabulary length."""
        assert tokenizer.vocab_size == len(tokenizer.vocab)

    def test_encode_simple_string(self, tokenizer):
        """A simple known string should encode deterministically."""
        text = "hello"
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert isinstance(ids, list), "encode should return a list"
        assert len(ids) == len(text), f"expected {len(text)} tokens, got {len(ids)}"
        assert all(isinstance(i, int) for i in ids), "all IDs should be integers"

    def test_encode_with_special_tokens(self, tokenizer):
        """Encode with add_special_tokens should add BOS and EOS."""
        ids = tokenizer.encode("hi", add_special_tokens=True)
        assert ids[0] == tokenizer.bos_id, "first token should be <BOS>"
        assert ids[-1] == tokenizer.eos_id, "last token should be <EOS>"
        # The actual content should be between BOS and EOS
        content = ids[1:-1]
        assert len(content) == 2

    def test_encode_unknown_char(self, tokenizer):
        """Characters not in vocab should be replaced with <UNK>."""
        # Find a character NOT in the vocabulary
        vocab = tokenizer.vocab
        # Try some unusual unicode characters
        for char in ["你好", "α", "😊"]:
            ids = tokenizer.encode(char, add_special_tokens=False)
            # Each unknown char should map to unk_id
            for i, c in enumerate(char):
                if c not in vocab:
                    assert ids[i] == tokenizer.unk_id, (
                        f"unknown char '{c}' should map to <UNK>"
                    )

    def test_decode_roundtrip(self, tokenizer):
        """encode followed by decode should recover the original string."""
        original = "Hello, world!"
        ids = tokenizer.encode(original, add_special_tokens=True)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded == original, f"roundtrip failed: '{original}' -> '{decoded}'"

    def test_decode_with_and_without_special(self, tokenizer):
        """skip_special_tokens should control special token output."""
        text = "test"
        ids = tokenizer.encode(text, add_special_tokens=True)

        decoded_with = tokenizer.decode(ids, skip_special_tokens=False)
        decoded_without = tokenizer.decode(ids, skip_special_tokens=True)

        # With special tokens should be longer
        assert len(decoded_with) > len(decoded_without), (
            "skipping special tokens should produce shorter output"
        )

    def test_multiline_text(self, tokenizer):
        """Tokenizer should handle multiline and special characters."""
        text = "Line 1\nLine 2\tTab"
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        assert decoded == text, "multiline roundtrip failed"

    def test_repr(self, tokenizer):
        """repr should include vocab_size."""
        r = repr(tokenizer)
        assert "CharTokenizer" in r
        assert str(tokenizer.vocab_size) in r

    def test_empty_string(self, tokenizer):
        """Empty string should produce empty list (without special tokens)."""
        ids = tokenizer.encode("", add_special_tokens=False)
        assert ids == [], "empty string should encode to empty list"

    def test_encode_decode_empty_with_special(self, tokenizer):
        """Empty string with special tokens should be [BOS, EOS]."""
        ids = tokenizer.encode("", add_special_tokens=True)
        assert ids == [tokenizer.bos_id, tokenizer.eos_id]


class TestTokenEmbedding:
    """Tests for the token embedding layer."""

    def test_construction(self):
        """TokenEmbedding should create an embedding matrix."""
        vocab_size, d_model = 70, 32
        emb = TokenEmbedding(vocab_size, d_model)
        assert emb.vocab_size == vocab_size
        assert emb.d_model == d_model
        assert hasattr(emb, "embedding")
        assert emb.embedding.weight.shape == (vocab_size, d_model)

    def test_forward_shape(self):
        """Forward should convert (batch, seq) to (batch, seq, d_model)."""
        vocab_size, d_model = 70, 32
        emb = TokenEmbedding(vocab_size, d_model)
        tokens = torch.randint(0, vocab_size, (2, 8))  # (batch=2, seq=8)
        out = emb(tokens)
        assert out.shape == (2, 8, d_model), f"forward shape mismatch: {out.shape}"

    def test_output_dtype(self):
        """Output should be float32 (the default for nn.Embedding)."""
        emb = TokenEmbedding(70, 32)
        tokens = torch.randint(0, 70, (1, 4))
        out = emb(tokens)
        assert out.dtype == torch.float32, f"expected float32, got {out.dtype}"

    def test_different_tokens_different_embeddings(self):
        """Different token IDs should produce different vectors."""
        emb = TokenEmbedding(70, 16)
        tok_a = torch.tensor([[5]])
        tok_b = torch.tensor([[42]])
        vec_a = emb(tok_a)
        vec_b = emb(tok_b)
        assert not torch.allclose(vec_a, vec_b), (
            "different tokens should have different embeddings"
        )

    def test_same_token_same_embedding(self):
        """Same token ID should always produce the same vector."""
        emb = TokenEmbedding(70, 16)
        tok = torch.tensor([[13]])
        vec_1 = emb(tok)
        vec_2 = emb(tok)
        assert torch.allclose(vec_1, vec_2), (
            "same token should produce identical embeddings"
        )

    def test_padding_idx_zero_gradient(self):
        """With padding_idx set, gradient for that row should be zero."""
        vocab_size, d_model = 70, 16
        emb = TokenEmbedding(vocab_size, d_model, padding_idx=0)
        tokens = torch.tensor([[0, 1, 2]])  # includes pad=0
        out = emb(tokens)
        loss = out.sum()
        loss.backward()
        # Gradient for padding_idx row should be zero
        assert emb.embedding.weight.grad[0].abs().sum().item() == 0.0, (
            "padding_idx should have zero gradient"
        )
        # Gradient for non-padding rows should be non-zero
        assert emb.embedding.weight.grad[1].abs().sum().item() > 0, (
            "non-padding rows should have non-zero gradient"
        )

    def test_batched_input(self):
        """Should handle batched inputs of various shapes."""
        emb = TokenEmbedding(70, 32)
        shapes = [(1, 5), (2, 8), (4, 16), (8, 1)]
        for batch, seq in shapes:
            tokens = torch.randint(0, 70, (batch, seq))
            out = emb(tokens)
            assert out.shape == (batch, seq, 32), f"shape mismatch for ({batch}, {seq})"

    def test_embedding_is_parameter(self):
        """The embedding should be a trainable parameter."""
        emb = TokenEmbedding(70, 16)
        params = list(emb.parameters())
        assert len(params) == 1, "should have exactly one parameter"
        assert params[0].shape == (70, 16), "parameter should be the embedding matrix"
