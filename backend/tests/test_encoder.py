"""
Unit tests for text encoder.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from encoder import TextEncoder, encode, TARGET_DIM


@pytest.fixture(scope="module")
def encoder():
    """Create a shared encoder instance for tests."""
    return TextEncoder(device='cpu')


class TestEncoderShape:
    """Test encoder output shape."""

    def test_single_text_shape(self, encoder):
        """Test encoding a single text."""
        texts = ["This is a test message."]
        embeddings = encoder.encode(texts)

        assert embeddings.shape == (1, TARGET_DIM)
        assert embeddings.shape[1] == 512

    def test_multiple_texts_shape(self, encoder):
        """Test encoding multiple texts."""
        texts = [
            "First test message.",
            "Second test message.",
            "Third test message.",
        ]
        embeddings = encoder.encode(texts)

        assert embeddings.shape == (3, TARGET_DIM)
        assert embeddings.shape[1] == 512

    def test_empty_list(self, encoder):
        """Test encoding empty list."""
        texts = []
        embeddings = encoder.encode(texts)

        assert embeddings.shape == (0, TARGET_DIM)


class TestL2Normalization:
    """Test L2 normalization of embeddings."""

    def test_single_text_norm(self, encoder):
        """Test that single text embedding has L2 norm ≈ 1."""
        texts = ["This is a test message with some content."]
        embeddings = encoder.encode(texts)

        norm = np.linalg.norm(embeddings[0])
        assert 0.99 <= norm <= 1.01, f"Expected norm ≈ 1, got {norm}"

    def test_multiple_texts_norms(self, encoder):
        """Test that all embeddings have L2 norm ≈ 1."""
        texts = [
            "Short text.",
            "This is a medium length text message.",
            "This is a longer text message with more words and content to process.",
        ]
        embeddings = encoder.encode(texts)

        for i, emb in enumerate(embeddings):
            norm = np.linalg.norm(emb)
            assert 0.99 <= norm <= 1.01, f"Text {i}: Expected norm ≈ 1, got {norm}"

    def test_short_text_norm(self, encoder):
        """Test L2 norm for short text."""
        texts = ["Hi"]
        embeddings = encoder.encode(texts)

        norm = np.linalg.norm(embeddings[0])
        assert 0.99 <= norm <= 1.01, f"Expected norm ≈ 1, got {norm}"

    def test_long_text_norm(self, encoder):
        """Test L2 norm for longer text."""
        texts = [
            "This is a much longer text that contains many words and should still be "
            "properly normalized to have an L2 norm of approximately 1.0 regardless "
            "of the length of the input text that we are processing here today."
        ]
        embeddings = encoder.encode(texts)

        norm = np.linalg.norm(embeddings[0])
        assert 0.99 <= norm <= 1.01, f"Expected norm ≈ 1, got {norm}"


class TestEmbeddingProperties:
    """Test embedding properties."""

    def test_embedding_dtype(self, encoder):
        """Test that embeddings are float32."""
        texts = ["Test message"]
        embeddings = encoder.encode(texts)

        assert embeddings.dtype == np.float32

    def test_embeddings_are_different(self, encoder):
        """Test that different texts produce different embeddings."""
        texts = [
            "This is the first text.",
            "This is completely different content.",
        ]
        embeddings = encoder.encode(texts)

        # Embeddings should not be identical
        assert not np.allclose(embeddings[0], embeddings[1])

        # But they should both be valid
        assert np.all(np.isfinite(embeddings))

    def test_embeddings_are_deterministic(self, encoder):
        """Test that same text produces same embedding."""
        texts = ["This is a test message."]

        emb1 = encoder.encode(texts)
        emb2 = encoder.encode(texts)

        # Should be very close (allowing for tiny floating point differences)
        np.testing.assert_allclose(emb1, emb2, rtol=1e-5)


class TestMeanPooling:
    """Test mean pooling functionality."""

    def test_mean_pooling_basic(self, encoder):
        """Test basic mean pooling."""
        import torch

        # Create simple test tensors
        hidden_states = torch.randn(2, 5, 768)  # batch=2, seq=5, hidden=768
        attention_mask = torch.ones(2, 5)  # All tokens attended

        pooled = encoder.mean_pool(hidden_states, attention_mask)

        assert pooled.shape == (2, 768)
        assert torch.all(torch.isfinite(pooled))

    def test_mean_pooling_with_padding(self, encoder):
        """Test mean pooling with padding (masked tokens)."""
        import torch

        # Create test tensors with padding
        hidden_states = torch.randn(2, 5, 768)
        attention_mask = torch.tensor([
            [1, 1, 1, 0, 0],  # First sample has 3 real tokens
            [1, 1, 1, 1, 1],  # Second sample has 5 real tokens
        ])

        pooled = encoder.mean_pool(hidden_states, attention_mask)

        assert pooled.shape == (2, 768)
        assert torch.all(torch.isfinite(pooled))


class TestConvenienceFunction:
    """Test the convenience encode function."""

    def test_encode_function(self):
        """Test the global encode function."""
        texts = ["Test message for encoding."]
        embeddings = encode(texts)

        assert embeddings.shape == (1, TARGET_DIM)
        assert embeddings.dtype == np.float32

        norm = np.linalg.norm(embeddings[0])
        assert 0.99 <= norm <= 1.01


class TestEdgeCases:
    """Test edge cases."""

    def test_very_short_text(self, encoder):
        """Test with very short text."""
        texts = ["a"]
        embeddings = encoder.encode(texts)

        assert embeddings.shape == (1, TARGET_DIM)
        assert np.all(np.isfinite(embeddings))

        norm = np.linalg.norm(embeddings[0])
        assert 0.99 <= norm <= 1.01

    def test_text_with_special_chars(self, encoder):
        """Test with special characters."""
        texts = ["Hello! How are you? I'm fine, thanks. 😊"]
        embeddings = encoder.encode(texts)

        assert embeddings.shape == (1, TARGET_DIM)
        assert np.all(np.isfinite(embeddings))

    def test_multilingual_text(self, encoder):
        """Test with multilingual text (xlm-roberta supports it)."""
        texts = [
            "Hello world",
            "Bonjour le monde",
            "Hola mundo",
        ]
        embeddings = encoder.encode(texts)

        assert embeddings.shape == (3, TARGET_DIM)

        # All should have proper norms
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert 0.99 <= norm <= 1.01


class TestBatchProcessing:
    """Test batch processing."""

    def test_small_batch(self, encoder):
        """Test with small batch."""
        texts = ["Text one", "Text two", "Text three"]
        embeddings = encoder.encode(texts)

        assert embeddings.shape == (3, TARGET_DIM)

    def test_large_batch(self, encoder):
        """Test with larger batch."""
        texts = [f"This is test message number {i}" for i in range(20)]
        embeddings = encoder.encode(texts)

        assert embeddings.shape == (20, TARGET_DIM)

        # All should have proper norms
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert 0.99 <= norm <= 1.01


class TestSimilarity:
    """Test that similar texts have higher similarity."""

    def test_similar_texts_high_similarity(self, encoder):
        """Test that similar texts have high cosine similarity."""
        texts = [
            "The cat sat on the mat.",
            "The cat is sitting on the mat.",
            "Dogs are playing in the park.",
        ]
        embeddings = encoder.encode(texts)

        # Compute cosine similarities
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_12 = cosine_similarity(embeddings[0], embeddings[1])
        sim_13 = cosine_similarity(embeddings[0], embeddings[2])

        # Similar texts (cat on mat) should have higher similarity
        # than dissimilar texts (cat vs dogs)
        assert sim_12 > sim_13, f"sim(cat,cat)={sim_12} should be > sim(cat,dogs)={sim_13}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
