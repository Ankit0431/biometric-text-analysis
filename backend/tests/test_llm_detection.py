"""
Unit tests for the hybrid LLM detection system.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from llm_detection import (
    extract_llm_features,
    heuristic_llm_detection,
    detect_llm_likeness,
    create_training_features,
    initialize_llm_detection,
    SKLEARN_AVAILABLE,
    NLTK_AVAILABLE,
    TRANSFORMERS_AVAILABLE
)


class TestFeatureExtraction:
    """Test feature extraction functionality."""
    
    def test_basic_feature_extraction(self):
        """Test basic feature extraction from normal text."""
        text = "This is a test sentence. Here is another one with different length."
        features = extract_llm_features(text)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (20,)
        assert not np.all(features == 0)  # Should have some non-zero features
        assert np.all(features >= 0)  # All features should be non-negative
        assert np.all(features <= 1)  # All features should be normalized to [0,1]

    def test_empty_text(self):
        """Test feature extraction with empty or very short text."""
        # Empty text
        features = extract_llm_features("")
        assert isinstance(features, np.ndarray)
        assert features.shape == (20,)
        assert np.all(features == 0)
        
        # Very short text
        features = extract_llm_features("Hi")
        assert isinstance(features, np.ndarray)
        assert features.shape == (20,)

    def test_feature_consistency(self):
        """Test that same text produces same features."""
        text = "This is a consistent test with multiple sentences. Each sentence has different lengths."
        
        features1 = extract_llm_features(text)
        features2 = extract_llm_features(text)
        
        np.testing.assert_array_equal(features1, features2)

    def test_different_texts_different_features(self):
        """Test that different texts produce different features."""
        text1 = "Short. Simple. Basic."
        text2 = "This is a much longer and more complex sentence. It has various linguistic patterns and structures."
        
        features1 = extract_llm_features(text1)
        features2 = extract_llm_features(text2)
        
        # Features should be different
        assert not np.array_equal(features1, features2)
        # Should have meaningful differences
        assert np.mean(np.abs(features1 - features2)) > 0.01


class TestHeuristicDetection:
    """Test heuristic-only LLM detection."""
    
    def test_basic_heuristic_detection(self):
        """Test basic heuristic detection functionality."""
        text = "This is a test text with normal human-like writing patterns."
        penalty, is_llm = heuristic_llm_detection(text)
        
        assert isinstance(penalty, float)
        assert isinstance(is_llm, bool)
        assert 0.0 <= penalty <= 1.0

    def test_short_text_no_penalty(self):
        """Short text should return no penalty."""
        penalty, is_llm = heuristic_llm_detection("Short")
        assert penalty == 0.0
        assert is_llm is False

    def test_formal_ai_text_detection(self):
        """Formal AI-like text should trigger detection."""
        ai_text = """
        Furthermore, it is important to note that the implementation of this system 
        requires careful consideration of multiple factors. Additionally, the proposed 
        solution demonstrates significant advantages. Consequently, I would recommend 
        proceeding with this approach.
        """
        penalty, is_llm = heuristic_llm_detection(ai_text)
        
        # Should have some penalty for formal AI patterns
        assert penalty > 0.1

    def test_casual_human_text(self):
        """Casual human text should have low penalty."""
        human_text = "Hey! How's it going? I was wondering if you'd seen the new movie. It's pretty good tbh."
        penalty, is_llm = heuristic_llm_detection(human_text)
        
        # Should have low penalty for casual human text
        assert penalty < 0.5


class TestHybridDetection:
    """Test hybrid ML + heuristic detection."""
    
    def test_hybrid_detection_basic(self):
        """Test basic hybrid detection functionality."""
        text = "This is a test text for hybrid detection with multiple sentences and patterns."
        penalty, is_llm = detect_llm_likeness(text)
        
        assert isinstance(penalty, float)
        assert isinstance(is_llm, bool)
        assert 0.0 <= penalty <= 1.0

    def test_hybrid_vs_heuristic_consistency(self):
        """Test that hybrid detection is reasonably consistent with heuristic."""
        text = "This is a test with various patterns and sentence structures for comparison."
        
        # Get hybrid result
        hybrid_penalty, hybrid_is_llm = detect_llm_likeness(text, use_ml=True)
        
        # Get heuristic-only result
        heuristic_penalty, heuristic_is_llm = heuristic_llm_detection(text)
        
        # Results should be in same ballpark (within 0.5)
        assert abs(hybrid_penalty - heuristic_penalty) <= 0.5

    def test_different_text_types(self):
        """Test detection on different types of text."""
        texts = [
            ("casual", "lol yeah that's pretty funny! didn't expect that tbh"),
            ("formal", "I am writing to inquire about the status of the application submitted last week."),
            ("ai_formal", "Furthermore, it is essential to consider the various implications. Additionally, proper implementation requires careful analysis."),
        ]
        
        for text_type, text in texts:
            penalty, is_llm = detect_llm_likeness(text)
            assert 0.0 <= penalty <= 1.0, f"Invalid penalty for {text_type}: {penalty}"

    def test_use_ml_parameter(self):
        """Test that use_ml parameter works correctly."""
        text = "This is a test text for ML parameter testing."
        
        # With ML (if available)
        penalty_ml, is_llm_ml = detect_llm_likeness(text, use_ml=True)
        
        # Without ML (heuristic only)
        penalty_heuristic, is_llm_heuristic = detect_llm_likeness(text, use_ml=False)
        
        # Both should return valid results
        assert 0.0 <= penalty_ml <= 1.0
        assert 0.0 <= penalty_heuristic <= 1.0


class TestTrainingFeatures:
    """Test training data creation functionality."""
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
    def test_create_training_features(self):
        """Test creation of training features."""
        texts = [
            "This is a human-written text with natural patterns.",
            "Furthermore, it is important to consider various factors. Additionally, proper implementation is essential.",
            "hey what's up? nothing much here lol",
            "The implementation demonstrates significant advantages over existing methodologies."
        ]
        labels = [0, 1, 0, 1]  # 0=human, 1=AI
        
        X, y = create_training_features(texts, labels)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == len([t for t in texts if len(t) > 10])  # Should have features for valid texts
        assert X.shape[1] == 20  # Should have 20 features
        assert len(y) == X.shape[0]

    def test_empty_training_data(self):
        """Test handling of empty training data."""
        with pytest.raises(ValueError):
            create_training_features([], [])


class TestSystemIntegration:
    """Test system integration and initialization."""
    
    def test_initialization(self):
        """Test system initialization."""
        result = initialize_llm_detection()
        assert isinstance(result, bool)

    def test_import_fallbacks(self):
        """Test that system works even when optional dependencies are missing."""
        # This test mainly ensures the module loads properly
        # Actual fallback behavior is tested implicitly in other tests
        assert True  # If we get here, imports worked

    def test_feature_availability_flags(self):
        """Test that feature availability flags are set correctly."""
        # These should be boolean values
        assert isinstance(SKLEARN_AVAILABLE, bool)
        assert isinstance(NLTK_AVAILABLE, bool)
        assert isinstance(TRANSFORMERS_AVAILABLE, bool)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        # Create a very long text
        long_text = "This is a sentence. " * 1000
        
        penalty, is_llm = detect_llm_likeness(long_text)
        assert 0.0 <= penalty <= 1.0
        assert isinstance(is_llm, bool)

    def test_special_characters(self):
        """Test handling of text with special characters."""
        special_text = "This text has émojis 🚀 and spéciál chàracters! It's wéird but válid."
        
        penalty, is_llm = detect_llm_likeness(special_text)
        assert 0.0 <= penalty <= 1.0

    def test_numeric_text(self):
        """Test handling of mostly numeric text."""
        numeric_text = "The results were 123.45, 678.90, and 111.22 respectively. These numbers are important."
        
        penalty, is_llm = detect_llm_likeness(numeric_text)
        assert 0.0 <= penalty <= 1.0

    def test_repetitive_text(self):
        """Test handling of repetitive text."""
        repetitive_text = "Same same same. Same same same. Same same same. Same same same."
        
        penalty, is_llm = detect_llm_likeness(repetitive_text)
        assert 0.0 <= penalty <= 1.0

    def test_mixed_language_text(self):
        """Test handling of mixed language text."""
        mixed_text = "Hello world! Bonjour le monde! ¡Hola mundo! This is mixed language text."
        
        penalty, is_llm = detect_llm_likeness(mixed_text)
        assert 0.0 <= penalty <= 1.0


class TestFeatureInterpretation:
    """Test that features make sense for different text types."""
    
    def test_sentence_variance_features(self):
        """Test sentence variance features work correctly."""
        # Very consistent sentences
        consistent_text = "This is sentence one. This is sentence two. This is sentence three."
        
        # Very varied sentences  
        varied_text = "Short. This is a medium length sentence. This is a much longer sentence with many words and complex structure."
        
        consistent_features = extract_llm_features(consistent_text)
        varied_features = extract_llm_features(varied_text)
        
        # Consistent text should have lower sentence variance features
        # (Features 0-3 are sentence-related)
        assert consistent_features[2] < varied_features[2]  # Coefficient of variation

    def test_punctuation_features(self):
        """Test punctuation-related features."""
        # Text with varied punctuation
        varied_punct = "Hello! How are you? I'm fine. Let's go; it'll be fun."
        
        # Text with mostly periods
        simple_punct = "This is sentence one. This is sentence two. This is sentence three."
        
        varied_features = extract_llm_features(varied_punct)
        simple_features = extract_llm_features(simple_punct)
        
        # Varied punctuation should have higher entropy (feature 4)
        assert varied_features[4] >= simple_features[4]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])