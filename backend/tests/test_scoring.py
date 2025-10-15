"""
Unit tests for scoring and policy modules.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from scoring import (
    cosine_similarity,
    mahalanobis_similarity,
    compute_semantic_similarity,
    compute_stylometry_similarity,
    detect_llm_likeness,
    score_sample,
    sigmoid
)
from policy import (
    decide,
    Decision,
    get_default_thresholds,
    validate_thresholds,
    adjust_thresholds_for_risk,
    MIN_WORDS_VERIFY,
    MIN_WORDS_ENROLL
)


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        v = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity ~0."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        sim = cosine_similarity(v1, v2)
        assert abs(sim) < 1e-6

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([-1.0, -2.0, -3.0])
        sim = cosine_similarity(v1, v2)
        assert abs(sim - (-1.0)) < 1e-6

    def test_zero_vector(self):
        """Zero vector should return 0 similarity."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([0.0, 0.0, 0.0])
        sim = cosine_similarity(v1, v2)
        assert sim == 0.0


class TestMahalanobisSimilarity:
    """Test Mahalanobis similarity computation."""

    def test_identical_vectors(self):
        """Identical vectors should have high similarity."""
        v = np.array([1.0, 2.0, 3.0])
        centroid = np.array([1.0, 2.0, 3.0])
        sim = mahalanobis_similarity(v, centroid)
        assert sim > 0.95

    def test_distant_vectors(self):
        """Distant vectors should have lower similarity."""
        v = np.array([1.0, 2.0, 3.0])
        centroid = np.array([10.0, 20.0, 30.0])
        sim = mahalanobis_similarity(v, centroid)
        assert sim < 0.5

    def test_with_covariance(self):
        """Test with diagonal covariance."""
        v = np.array([1.0, 2.0, 3.0])
        centroid = np.array([1.1, 2.1, 3.1])
        cov_diag = np.array([0.1, 0.1, 0.1])
        sim = mahalanobis_similarity(v, centroid, cov_diag)
        assert 0.0 <= sim <= 1.0


class TestSemanticSimilarity:
    """Test semantic similarity computation."""

    def test_high_similarity(self):
        """Test high similarity case."""
        emb = np.array([1.0, 0.0, 0.0])
        centroid = np.array([0.9, 0.1, 0.0])
        sim = compute_semantic_similarity(emb, centroid, use_mahalanobis=False)
        assert sim > 0.8

    def test_low_similarity(self):
        """Test low similarity case."""
        emb = np.array([1.0, 0.0, 0.0])
        centroid = np.array([0.0, 1.0, 0.0])
        sim = compute_semantic_similarity(emb, centroid, use_mahalanobis=False)
        assert sim < 0.7


class TestStylometrySimilarity:
    """Test stylometry similarity computation."""

    def test_identical_style(self):
        """Identical style should have high similarity."""
        style = np.array([1.0, 2.0, 3.0, 4.0])
        profile_mean = np.array([1.0, 2.0, 3.0, 4.0])
        sim = compute_stylometry_similarity(style, profile_mean)
        assert sim > 0.4  # Should be above 0.4 for identical style

    def test_different_style(self):
        """Different style should have lower similarity."""
        style = np.array([1.0, 2.0, 3.0, 4.0])
        profile_mean = np.array([5.0, 6.0, 7.0, 8.0])
        sim = compute_stylometry_similarity(style, profile_mean)
        assert sim < 0.7


class TestLLMDetection:
    """Test LLM-likeness detection."""

    def test_short_text(self):
        """Short text should return no penalty."""
        text = "This is a short text."
        penalty, llm_like = detect_llm_likeness(text)
        assert penalty == 0.0
        assert llm_like is False

    def test_human_like_text(self):
        """Human-like text with varied sentence lengths."""
        text = """Hi there! This is a test.
        I'm writing something with different sentence lengths.
        Some are short. Others are much longer and contain more information.
        This variety is typical of human writing."""
        penalty, llm_like = detect_llm_likeness(text)
        # Should have low penalty for varied human text
        assert penalty < 0.5

    def test_consistent_text(self):
        """Very consistent text might trigger LLM detection."""
        # Create text with very uniform sentence lengths
        sentences = [
            "This is a sentence with exactly ten words in it right here.",
            "Here is another sentence that has exactly ten words too.",
            "Yet another sentence with exactly ten words is right here.",
            "And one more sentence that contains exactly ten words total.",
        ]
        text = " ".join(sentences)
        penalty, llm_like = detect_llm_likeness(text)
        # May or may not trigger depending on thresholds
        assert 0.0 <= penalty <= 1.0


class TestScoreSample:
    """Test the main score_sample function."""

    def test_perfect_match(self):
        """Test scoring with perfect match to profile."""
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        style = np.random.randn(512).astype(np.float32)

        profile = {
            'centroid': embedding.copy(),
            'style_mean': style.copy(),
            'style_std': np.ones(512, dtype=np.float32) * 0.1,
        }

        text = "This is a test message with sufficient length to avoid being too short for the system."

        result = score_sample(profile, text, embedding, style)

        assert 'final_score' in result
        assert 'semantic_score' in result
        assert 'stylometry_score' in result
        assert 'llm_penalty' in result
        assert 0.0 <= result['final_score'] <= 1.0
        assert result['final_score'] > 0.7  # Should be high for perfect match

    def test_poor_match(self):
        """Test scoring with poor match to profile."""
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # Create very different embedding
        embedding2 = -embedding  # Opposite direction

        style = np.random.randn(512).astype(np.float32)
        style2 = np.random.randn(512).astype(np.float32) * 5  # Very different

        profile = {
            'centroid': embedding,
            'style_mean': style,
            'style_std': np.ones(512, dtype=np.float32) * 0.1,
        }

        text = "This is a test message with sufficient length to avoid being too short."

        result = score_sample(profile, text, embedding2, style2)

        assert 'final_score' in result
        assert 0.0 <= result['final_score'] <= 1.0
        assert result['final_score'] < 0.7  # Should be low for poor match


class TestPolicyDecision:
    """Test policy decision logic."""

    def test_high_confidence_allow(self):
        """Score at or above high threshold should ALLOW."""
        thresholds = get_default_thresholds()

        result = decide(
            score=0.84,
            threshold_high=thresholds['high'],
            threshold_med=thresholds['med'],
            text="This is a test message with enough words " * 10,
            word_count=100,
            llm_like=False,
            flow="verify"
        )

        assert result['decision'] == Decision.ALLOW
        assert 'HIGH_CONFIDENCE' in result['reasons']

    def test_high_threshold_boundary(self):
        """Score exactly at high threshold should ALLOW."""
        result = decide(
            score=0.84,
            threshold_high=0.84,
            threshold_med=0.72,
            text="Test " * 60,
            word_count=60,
            llm_like=False
        )

        assert result['decision'] == Decision.ALLOW

    def test_medium_confidence_challenge(self):
        """Score between med and high should CHALLENGE."""
        result = decide(
            score=0.78,
            threshold_high=0.84,
            threshold_med=0.72,
            text="Test " * 60,
            word_count=60,
            llm_like=False
        )

        assert result['decision'] == Decision.CHALLENGE
        assert 'MED_CONFIDENCE' in result['reasons']

    def test_low_confidence_step_up(self):
        """Score below med threshold should STEP_UP."""
        result = decide(
            score=0.65,
            threshold_high=0.84,
            threshold_med=0.72,
            text="Test " * 60,
            word_count=60,
            llm_like=False
        )

        assert result['decision'] == Decision.STEP_UP
        assert 'LOW_CONFIDENCE' in result['reasons']

    def test_short_text_challenge(self):
        """Text below minimum word count should CHALLENGE."""
        result = decide(
            score=0.90,  # Even with high score
            threshold_high=0.84,
            threshold_med=0.72,
            text="Short text",
            word_count=2,
            llm_like=False,
            flow="verify"
        )

        assert result['decision'] == Decision.CHALLENGE
        assert 'SHORT_LEN' in result['reasons']

    def test_llm_like_challenge(self):
        """LLM-like text with medium score should CHALLENGE."""
        result = decide(
            score=0.78,
            threshold_high=0.84,
            threshold_med=0.72,
            text="Test " * 60,
            word_count=60,
            llm_like=True
        )

        assert result['decision'] == Decision.CHALLENGE
        assert 'LLM_LIKE' in result['reasons']

    def test_llm_like_low_score_step_up(self):
        """LLM-like text with low score should STEP_UP."""
        result = decide(
            score=0.60,
            threshold_high=0.84,
            threshold_med=0.72,
            text="Test " * 60,
            word_count=60,
            llm_like=True
        )

        assert result['decision'] == Decision.STEP_UP
        assert 'LLM_LIKE' in result['reasons']

    def test_enroll_flow_word_count(self):
        """Enroll flow should use higher word count minimum."""
        # Text with 60 words should be OK for verify but not enroll
        result_verify = decide(
            score=0.90,
            threshold_high=0.84,
            threshold_med=0.72,
            text="Test " * 60,
            word_count=60,
            llm_like=False,
            flow="verify"
        )

        result_enroll = decide(
            score=0.90,
            threshold_high=0.84,
            threshold_med=0.72,
            text="Test " * 60,
            word_count=60,
            llm_like=False,
            flow="enroll"
        )

        assert result_verify['decision'] == Decision.ALLOW
        assert result_enroll['decision'] == Decision.CHALLENGE
        assert 'SHORT_LEN' in result_enroll['reasons']


class TestThresholds:
    """Test threshold validation and adjustment."""

    def test_valid_thresholds(self):
        """Valid thresholds should pass validation."""
        assert validate_thresholds(0.84, 0.72) is True
        assert validate_thresholds(0.90, 0.80) is True

    def test_invalid_order(self):
        """Med >= high should be invalid."""
        assert validate_thresholds(0.72, 0.84) is False
        assert validate_thresholds(0.80, 0.80) is False

    def test_out_of_range(self):
        """Thresholds outside [0, 1] should be invalid."""
        assert validate_thresholds(1.5, 0.72) is False
        assert validate_thresholds(0.84, -0.1) is False

    def test_risk_adjustment_high(self):
        """High risk should increase thresholds."""
        base = get_default_thresholds()
        adjusted = adjust_thresholds_for_risk(base, risk_level="high")

        assert adjusted['high'] > base['high']
        assert adjusted['med'] > base['med']

    def test_risk_adjustment_low(self):
        """Low risk should decrease thresholds."""
        base = get_default_thresholds()
        adjusted = adjust_thresholds_for_risk(base, risk_level="low")

        assert adjusted['high'] < base['high']
        assert adjusted['med'] < base['med']


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_score_at_boundaries(self):
        """Test decisions at exact threshold boundaries."""
        # Exactly at high threshold
        result = decide(0.84, 0.84, 0.72, "test " * 60, 60)
        assert result['decision'] == Decision.ALLOW

        # Just below high threshold
        result = decide(0.839, 0.84, 0.72, "test " * 60, 60)
        assert result['decision'] == Decision.CHALLENGE

        # Exactly at med threshold
        result = decide(0.72, 0.84, 0.72, "test " * 60, 60)
        assert result['decision'] == Decision.CHALLENGE

        # Just below med threshold
        result = decide(0.719, 0.84, 0.72, "test " * 60, 60)
        assert result['decision'] == Decision.STEP_UP

    def test_extreme_scores(self):
        """Test with extreme score values."""
        # Score = 1.0 (perfect)
        result = decide(1.0, 0.84, 0.72, "test " * 60, 60)
        assert result['decision'] == Decision.ALLOW

        # Score = 0.0 (worst)
        result = decide(0.0, 0.84, 0.72, "test " * 60, 60)
        assert result['decision'] == Decision.STEP_UP


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
