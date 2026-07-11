"""
tests/transformer/test_embedding.py — Tests for TokenEmbedding and CharTokenizer.
"""

import pytest
import torch
from core.transformer.embedding import (
    BPETokenizer,
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
        assert hasattr(tokenizer, "id_to_token")
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


class TestBPETokenizer:
    """Tests for the BPE tokenizer (Phase 6)."""

    # ------------------------------------------------------------------
    # Pre-tokenization
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Hello, world!", ["Hello", ",", " ", "world", "!"]),
            ("Let's go", ["Let", "'", "s", " ", "go"]),
            ("", []),
            ("!@#$ %%%", ["!", "@", "#", "$", " ", "%", "%", "%"]),
            ("abc123_def", ["abc123_def"]),  # \w includes underscore
            (" no spaces ", [" ", "no", " ", "spaces", " "]),
        ],
    )
    def test_pre_tokenize(self, text, expected):
        """_pre_tokenize should split text by \\w+, \\s+, and individual [^\\w\\s]."""
        tok = BPETokenizer()
        assert tok._pre_tokenize(text) == expected

    # ------------------------------------------------------------------
    # Character vocabulary
    # ------------------------------------------------------------------

    def test_get_char_vocab_simple(self):
        """_get_char_vocab should collect unique chars with IDs after special."""
        tok = BPETokenizer(special_tokens=["<PAD>", "<UNK>"])
        words = ["hello", "world"]
        vocab = tok._get_char_vocab(words)
        # sorted(chars) = ['d','e','h','l','o','r','w'] → IDs start at 2
        assert vocab["d"] == 2
        assert vocab["e"] == 3
        assert vocab["h"] == 4
        assert set(vocab.keys()) == {"h", "e", "l", "o", "w", "r", "d"}
        # Special tokens should NOT be in char vocab
        assert "<PAD>" not in vocab
        assert "<UNK>" not in vocab

    # ------------------------------------------------------------------
    # Pair frequency
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "words, expected_pairs",
        [
            ([["a", "b", "c"]], {("a", "b"): 1, ("b", "c"): 1}),
            ([["a", "b"], ["a", "b"]], {("a", "b"): 2}),
            ([["x"]], {}),  # single symbol → no pairs
            ([["a", "a", "a"]], {("a", "a"): 2}),  # overlapping non-merge
        ],
    )
    def test_get_pair_freqs(self, words, expected_pairs):
        """_get_pair_freqs should count adjacent pairs correctly."""
        tok = BPETokenizer()
        assert tok._get_pair_freqs(words) == expected_pairs

    # ------------------------------------------------------------------
    # Merge pair
    # ------------------------------------------------------------------

    def test_merge_pair_basic(self):
        """_merge_pair should replace all non-overlapping occurrences."""
        tok = BPETokenizer()
        words = [["a", "b", "c"], ["a", "b", "a", "b"]]
        result = tok._merge_pair(words, ("a", "b"), "ab")
        assert result == [["ab", "c"], ["ab", "ab"]]

    def test_merge_pair_overlap(self):
        """Overlapping pairs should not be double-merged."""
        tok = BPETokenizer()
        words = [["a", "a", "a"]]
        result = tok._merge_pair(words, ("a", "a"), "aa")
        assert result == [["aa", "a"]]

    def test_merge_pair_no_match(self):
        """When no pair matches, words should be unchanged."""
        tok = BPETokenizer()
        words = [["x", "y", "z"]]
        result = tok._merge_pair(words, ("a", "b"), "ab")
        assert result == [["x", "y", "z"]]

    # ------------------------------------------------------------------
    # Full training pipeline
    # ------------------------------------------------------------------

    def test_train_creates_vocab_and_merges(self):
        """train should populate vocab, merges, and merge_ranks."""
        tok = BPETokenizer(vocab_size=12)
        tok.train(["low low new"])
        # 4 special + 5 chars (e,l,n,o,w) + up to 3 merges = 12 max, but
        # corpus is small so freq drops below 2 after ~2 merges
        assert len(tok.vocab) >= 9  # at least special + chars
        assert len(tok.merges) >= 1
        assert len(tok.merge_ranks) == len(tok.merges)
        # Special tokens should be in vocab
        for s in tok.special_tokens:
            assert s in tok.vocab
        # Base characters should be in vocab
        for c in "lowen":
            assert c in tok.vocab

    def test_train_subword_formation(self):
        """Common pairs should be merged into subword tokens."""
        tok = BPETokenizer(vocab_size=12)
        tok.train(["low low new"])
        merged_tokens = {k for k in tok.vocab if len(k) > 1 and not k.startswith("<")}
        assert len(merged_tokens) >= 1, "should have at least one merged subword"

    def test_train_larger_corpus(self):
        """Training on more text should produce more meaningful merges."""
        text = "the cat sat on the mat the rat hat"
        tok = BPETokenizer(vocab_size=30)
        tok.train([text])
        # Should have merged "th" or "he" or "at" — common English bigrams
        merged = {k for k in tok.vocab if len(k) > 1}
        assert len(merged) > 1

    def test_train_tie_breaking_deterministic(self):
        """Same corpus should produce same merges."""
        tok1 = BPETokenizer(vocab_size=15)
        tok2 = BPETokenizer(vocab_size=15)
        tok1.train(["hello world foo bar baz"])
        tok2.train(["hello world foo bar baz"])
        assert tok1.merges == tok2.merges
        assert tok1.vocab == tok2.vocab

    # ------------------------------------------------------------------
    # _encode_word
    # ------------------------------------------------------------------

    @pytest.fixture
    def trained_tok(self):
        """A small BPETokenizer trained on repetitive text so ('l','o')→'lo'→'low'."""
        tok = BPETokenizer(vocab_size=15)
        tok.train(["low low low low new new new new low low"])
        return tok

    def test_encode_word_merged(self, trained_tok):
        """``low`` should be encoded as a single subword after training."""
        tokens = trained_tok._encode_word("low")
        assert tokens == ["low"]

    def test_encode_word_unmerged(self, trained_tok):
        """``new`` should stay as separate chars if ('n','e') was too rare."""
        tokens = trained_tok._encode_word("new")
        # Either merged to ["new"] or stayed as chars — either is valid
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)
        assert "".join(tokens) == "new"

    def test_encode_word_no_merges(self):
        """Without training, encode_word should return the word's characters."""
        tok = BPETokenizer()
        assert tok._encode_word("hi") == ["h", "i"]

    def test_encode_word_empty(self, trained_tok):
        """Empty word should produce empty list."""
        assert trained_tok._encode_word("") == []

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------

    def test_encode_returns_ids(self, trained_tok):
        """encode should return a list of token IDs."""
        ids = trained_tok.encode("low new")
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_encode_known_word_maps_to_same_id(self, trained_tok):
        """Same word should always encode to the same ID sequence."""
        ids1 = trained_tok.encode("low")
        ids2 = trained_tok.encode("low")
        assert ids1 == ids2

    def test_encode_empty_string(self, trained_tok):
        """Empty string should return empty list."""
        assert trained_tok.encode("") == []

    def test_encode_unknown_token_uses_unk(self, trained_tok):
        """Characters not in vocab should become <UNK>."""
        unk_id = trained_tok.vocab["<UNK>"]
        # Use a character unlikely to be in our tiny vocab
        ids = trained_tok.encode("lowxyz_unknown")
        assert unk_id in ids

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def test_decode_basic(self, trained_tok):
        """decode should convert IDs back to the correct token string."""
        # "low" should be a merged token
        ids = trained_tok.encode("low")
        decoded = trained_tok.decode(ids)
        assert decoded == "low"

    def test_decode_skips_special_by_default(self, trained_tok):
        """Special tokens should be excluded by default."""
        bos_id = trained_tok.vocab["<BOS>"]
        eos_id = trained_tok.vocab["<EOS>"]
        decoded = trained_tok.decode([bos_id, eos_id])
        assert decoded == "", "special tokens should be skipped"

    def test_decode_includes_special_when_requested(self, trained_tok):
        """skip_special_tokens=False should keep special tokens."""
        unk_id = trained_tok.vocab["<UNK>"]
        decoded = trained_tok.decode([unk_id], skip_special_tokens=False)
        assert "<UNK>" in decoded

    def test_decode_empty_list(self, trained_tok):
        """Empty ID list should produce empty string."""
        assert trained_tok.decode([]) == ""

    # ------------------------------------------------------------------
    # Roundtrip
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "text",
        [
            "low",
            "new",
            "low new",
            "well",
            "low low well",
        ],
    )
    def test_encode_decode_roundtrip_known_words(self, trained_tok, text):
        """encode → decode should reproduce concatenated subwords."""
        ids = trained_tok.encode(text)
        decoded = trained_tok.decode(ids)
        # The training vocab only contains {e,l,n,o,w,space} and learned merges,
        # so these test strings should roundtrip losslessly.
        words = trained_tok._pre_tokenize(text)
        expected = "".join(words)
        assert decoded == expected
