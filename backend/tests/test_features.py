"""
Unit tests for stylometry feature extraction.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import hashlib
from features import (
    extract_features,
    StyleFeatureExtractor,
    TOTAL_FEATURE_DIMS,
    CHAR_NGRAM_DIMS,
    FUNCTION_WORD_DIMS,
    STATISTICAL_DIMS,
    POS_TRIGRAM_DIMS,
)


class TestFeatureVectorShape:
    """Test that feature vectors have the correct shape."""

    def test_feature_vector_dimension(self):
        """Test that feature vector has exactly 512 dimensions."""
        text = "This is a test message with some words to analyze."
        tokens = text.split()

        vector, stats = extract_features(text, tokens)

        assert vector.shape == (TOTAL_FEATURE_DIMS,)
        assert len(vector) == 512

    def test_component_dimensions(self):
        """Test that all feature components add up correctly."""
        total = CHAR_NGRAM_DIMS + FUNCTION_WORD_DIMS + STATISTICAL_DIMS + POS_TRIGRAM_DIMS
        assert total == TOTAL_FEATURE_DIMS
        assert total == 512


class TestDeterminism:
    """Test that feature extraction is deterministic."""

    def test_same_input_same_output(self):
        """Test that same input produces same output."""
        text = "This is a test message with punctuation! And multiple sentences. What do you think?"
        tokens = text.split()

        vector1, stats1 = extract_features(text, tokens)
        vector2, stats2 = extract_features(text, tokens)

        # Vectors should be identical
        np.testing.assert_array_equal(vector1, vector2)

        # Stats should be identical
        assert stats1 == stats2

    def test_deterministic_checksum(self):
        """Test that known input produces deterministic checksum."""
        text = "The quick brown fox jumps over the lazy dog."
        tokens = text.split()

        vector, stats = extract_features(text, tokens)

        # Round to fixed precision for stable checksum
        vector_rounded = np.round(vector, decimals=6)

        # Calculate MD5 checksum
        checksum = hashlib.md5(vector_rounded.tobytes()).hexdigest()

        # This checksum should be consistent across runs
        assert isinstance(checksum, str)
        assert len(checksum) == 32  # MD5 hex digest length

        # Verify it's the same on second run
        vector2, _ = extract_features(text, tokens)
        vector2_rounded = np.round(vector2, decimals=6)
        checksum2 = hashlib.md5(vector2_rounded.tobytes()).hexdigest()

        assert checksum == checksum2

    def test_different_input_different_output(self):
        """Test that different inputs produce different outputs."""
        text1 = "This is the first text sample."
        text2 = "This is a completely different text."

        vector1, _ = extract_features(text1, text1.split())
        vector2, _ = extract_features(text2, text2.split())

        # Vectors should be different
        assert not np.array_equal(vector1, vector2)


class TestCharacterNgrams:
    """Test character n-gram features."""

    def test_char_ngram_extraction(self):
        """Test that character n-grams are extracted."""
        extractor = StyleFeatureExtractor()
        text = "hello world"

        features = extractor.extract_char_ngrams(text)

        assert features.shape == (CHAR_NGRAM_DIMS,)
        assert features.sum() > 0  # Should have some features
        assert np.all(features >= 0)  # All values should be non-negative

    def test_char_ngram_normalized(self):
        """Test that character n-gram features are normalized."""
        extractor = StyleFeatureExtractor()
        text = "test message"

        features = extractor.extract_char_ngrams(text)

        # Should sum to approximately 1 (normalized)
        assert 0.99 <= features.sum() <= 1.01


class TestFunctionWords:
    """Test function word features."""

    def test_function_word_extraction(self):
        """Test function word frequency extraction."""
        extractor = StyleFeatureExtractor()
        tokens = ["the", "quick", "brown", "fox", "the", "dog"]

        features = extractor.extract_function_word_features(tokens)

        assert features.shape == (FUNCTION_WORD_DIMS,)
        assert features.sum() > 0  # Should have some features

    def test_function_word_frequency(self):
        """Test that function word frequencies are calculated correctly."""
        extractor = StyleFeatureExtractor()
        # "the" appears twice in 6 tokens = 2/6 = 0.333...
        tokens = ["the", "quick", "brown", "fox", "the", "dog"]

        features = extractor.extract_function_word_features(tokens)

        # "the" should be in the function word list
        if "the" in extractor.function_words[:FUNCTION_WORD_DIMS]:
            idx = extractor.function_words.index("the")
            expected_freq = 2 / 6
            assert abs(features[idx] - expected_freq) < 0.01


class TestSentenceStats:
    """Test sentence statistics."""

    def test_sentence_stats_extraction(self):
        """Test sentence statistics extraction."""
        extractor = StyleFeatureExtractor()
        text = "This is sentence one. This is sentence two! Is this sentence three?"

        stats = extractor.extract_sentence_stats(text)

        assert 'mean_sentence_len' in stats
        assert 'std_sentence_len' in stats
        assert 'num_sentences' in stats
        assert stats['num_sentences'] == 3

    def test_single_sentence(self):
        """Test statistics for single sentence."""
        extractor = StyleFeatureExtractor()
        text = "This is a single sentence."

        stats = extractor.extract_sentence_stats(text)

        assert stats['num_sentences'] == 1
        assert stats['mean_sentence_len'] == 5  # 5 words


class TestPunctuationFeatures:
    """Test punctuation features."""

    def test_punctuation_extraction(self):
        """Test punctuation feature extraction."""
        extractor = StyleFeatureExtractor()
        text = "Hello! How are you? I'm fine, thanks."
        tokens = text.split()

        stats = extractor.extract_punctuation_features(text, tokens)

        assert 'punct_rate_per_1k' in stats
        assert 'comma_rate' in stats
        assert 'period_rate' in stats
        assert 'question_rate' in stats
        assert 'exclamation_rate' in stats

    def test_punctuation_counts(self):
        """Test that punctuation is counted correctly."""
        extractor = StyleFeatureExtractor()
        text = "Hello, world!"
        tokens = ["Hello,", "world!"]

        stats = extractor.extract_punctuation_features(text, tokens)

        # Should detect comma and exclamation
        assert stats['comma_rate'] > 0
        assert stats['exclamation_rate'] > 0


class TestContractionFeatures:
    """Test contraction features."""

    def test_contraction_extraction(self):
        """Test contraction detection."""
        extractor = StyleFeatureExtractor()
        text = "I'm happy. You're welcome. We'll see."
        tokens = text.split()

        stats = extractor.extract_contraction_features(text, tokens)

        assert 'contraction_rate' in stats
        assert stats['contraction_rate'] > 0


class TestEntropyFeatures:
    """Test entropy features."""

    def test_entropy_extraction(self):
        """Test entropy calculation."""
        extractor = StyleFeatureExtractor()
        tokens = ["the", "the", "cat", "sat"]

        stats = extractor.extract_entropy_features(tokens)

        assert 'token_entropy' in stats
        assert 'unique_token_ratio' in stats
        assert stats['token_entropy'] > 0
        assert 0 < stats['unique_token_ratio'] <= 1

    def test_unique_token_ratio(self):
        """Test unique token ratio calculation."""
        extractor = StyleFeatureExtractor()
        # 3 unique tokens out of 4 total = 0.75
        tokens = ["the", "the", "cat", "sat"]

        stats = extractor.extract_entropy_features(tokens)

        expected_ratio = 3 / 4
        assert abs(stats['unique_token_ratio'] - expected_ratio) < 0.01


class TestWordLengthFeatures:
    """Test word length features."""

    def test_word_length_extraction(self):
        """Test word length statistics."""
        extractor = StyleFeatureExtractor()
        tokens = ["a", "test", "message"]

        stats = extractor.extract_word_length_features(tokens)

        assert 'mean_word_len' in stats
        assert 'std_word_len' in stats
        assert 'max_word_len' in stats
        assert stats['max_word_len'] == 7  # "message" has 7 chars


class TestPOSTrigramFeatures:
    """Test POS trigram features."""

    def test_pos_trigram_extraction(self):
        """Test POS trigram feature extraction."""
        extractor = StyleFeatureExtractor()
        tokens = ["The", "quick", "brown", "fox", "jumps"]

        features = extractor.extract_pos_trigram_features(tokens)

        assert features.shape == (POS_TRIGRAM_DIMS,)
        assert features.sum() > 0 or len(tokens) < 3  # Should have features if enough tokens


class TestStatsSummary:
    """Test that stats summary includes expected keys and values."""

    def test_stats_keys(self):
        """Test that stats dict includes all expected keys."""
        text = "This is a test! How are you? I'm fine, thanks. Testing multiple sentences here."
        tokens = text.split()

        vector, stats = extract_features(text, tokens)

        # Check for expected keys
        expected_keys = [
            'mean_sentence_len', 'std_sentence_len', 'num_sentences',
            'punct_rate_per_1k', 'comma_rate', 'period_rate',
            'contraction_rate', 'token_entropy', 'unique_token_ratio',
            'mean_word_len', 'std_word_len', 'max_word_len',
        ]

        for key in expected_keys:
            assert key in stats, f"Expected key '{key}' not found in stats"

    def test_stats_value_ranges(self):
        """Test that stats values are in expected ranges."""
        text = "This is a test message with several words and punctuation marks! How exciting?"
        tokens = text.split()

        vector, stats = extract_features(text, tokens)

        # Rates should be between 0 and 1
        assert 0 <= stats['comma_rate'] <= 1
        assert 0 <= stats['period_rate'] <= 1
        assert 0 <= stats['contraction_rate'] <= 1
        assert 0 <= stats['unique_token_ratio'] <= 1

        # Counts should be non-negative
        assert stats['num_sentences'] >= 0
        assert stats['mean_sentence_len'] >= 0
        assert stats['mean_word_len'] >= 0

        # Entropy should be non-negative
        assert stats['token_entropy'] >= 0


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_text(self):
        """Test with empty text."""
        text = ""
        tokens = []

        vector, stats = extract_features(text, tokens)

        assert vector.shape == (TOTAL_FEATURE_DIMS,)
        assert isinstance(stats, dict)

    def test_short_text(self):
        """Test with very short text."""
        text = "Hi"
        tokens = ["Hi"]

        vector, stats = extract_features(text, tokens)

        assert vector.shape == (TOTAL_FEATURE_DIMS,)
        assert stats['num_sentences'] >= 0

    def test_long_text(self):
        """Test with longer text."""
        text = " ".join(["word"] * 1000)
        tokens = text.split()

        vector, stats = extract_features(text, tokens)

        assert vector.shape == (TOTAL_FEATURE_DIMS,)
        assert stats['mean_word_len'] == 4  # "word" has 4 chars


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test full feature extraction pipeline."""
        text = """
        Hello! This is a test message. I'm testing the feature extraction system.
        It should handle multiple sentences correctly, with punctuation and contractions.
        The quick brown fox jumps over the lazy dog. How are you today?
        """
        tokens = text.split()

        vector, stats = extract_features(text, tokens)

        # Check vector properties
        assert vector.shape == (TOTAL_FEATURE_DIMS,)
        assert vector.dtype == np.float32
        assert np.all(np.isfinite(vector))  # No NaN or inf values

        # Check stats properties
        assert isinstance(stats, dict)
        assert len(stats) > 0
        assert all(isinstance(v, (int, float, np.number)) for v in stats.values())

    def test_realistic_text(self):
        """Test with realistic text sample."""
        text = """
        I wanted to reach out regarding the project timeline. We've made significant
        progress this week, but there are still a few outstanding issues that need
        attention. Could we schedule a meeting to discuss the next steps? I'm available
        tomorrow afternoon or Thursday morning. Let me know what works best for you!
        """
        tokens = text.split()

        vector, stats = extract_features(text, tokens)

        # Should have reasonable statistics
        assert stats['num_sentences'] > 0
        assert stats['mean_sentence_len'] > 0
        assert stats['token_entropy'] > 0
        assert 0 < stats['unique_token_ratio'] <= 1

        # Vector should be normalized and finite
        assert np.all(np.isfinite(vector))
        assert vector.sum() > 0


class TestConsistency:
    """Test consistency across multiple calls."""

    def test_multiple_extractions_same_result(self):
        """Test that multiple extractions produce same result."""
        text = "The quick brown fox jumps over the lazy dog."
        tokens = text.split()

        checksums = []
        for _ in range(5):
            vector, _ = extract_features(text, tokens)
            vector_rounded = np.round(vector, decimals=6)
            checksum = hashlib.md5(vector_rounded.tobytes()).hexdigest()
            checksums.append(checksum)

        # All checksums should be identical
        assert len(set(checksums)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
