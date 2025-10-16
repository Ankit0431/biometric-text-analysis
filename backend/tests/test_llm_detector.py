"""
Unit tests for LLM-likeness detector.

Tests the heuristic-based detector with various text samples.
"""

import pytest
from llm_detector import (
    detect_llm_likeness,
    get_penalty,
    is_llm_generated,
    _extract_features,
)


class TestLLMDetector:
    """Test suite for LLM detector functionality."""

    def test_empty_text(self):
        """Empty text should return zero penalty."""
        penalty, llm_like, details = detect_llm_likeness("")
        assert penalty == 0.0
        assert llm_like is False
        assert details['reason'] == "text_too_short"

    def test_very_short_text(self):
        """Very short text should return zero penalty."""
        penalty, llm_like, details = detect_llm_likeness("Hello world")
        assert penalty == 0.0
        assert llm_like is False

    def test_typical_human_text(self):
        """Typical casual human text should have low penalty."""
        text = """
        Hey! So I was thinking about what you said yesterday, and honestly?
        I'm not sure I agree. Like, it's a good point and all, but there's
        so many other things to consider. You know what I mean? Anyway,
        let me know what you think when you get a chance. Can't wait to hear
        your thoughts on this!
        """
        penalty, llm_like, details = detect_llm_likeness(text)
        assert penalty < 0.10  # Should be below LLM threshold
        assert llm_like is False

    def test_llm_like_formal_text(self):
        """Formal, polished text should trigger higher penalty."""
        text = """
        In conclusion, the analysis demonstrates several important points.
        First, the methodology employed ensures robust results. Second,
        the data supports the hypothesis. Third, further research is needed.
        It is important to note that these findings are preliminary.
        However, they provide a strong foundation for future work.
        """
        penalty, llm_like, details = detect_llm_likeness(text)
        assert penalty > 0  # Should have some penalty
        assert "uniform_sentences" in details['flags'] or "typical_llm_length" in details['flags']

    def test_explicit_llm_phrases(self):
        """Text with explicit LLM phrases should have high penalty."""
        text = """
        As an AI language model, I don't have personal opinions. However,
        I can provide you with information on this topic. It's important to
        note that my knowledge cutoff is in 2021. If you have any further
        questions, feel free to ask!
        """
        penalty, llm_like, details = detect_llm_likeness(text)
        assert penalty >= 0.10  # Should be high enough to classify as LLM
        assert llm_like is True
        assert details['features']['llm_phrase_count'] > 0
        assert any('llm_phrases' in flag for flag in details['flags'])

    def test_penalty_capped_at_0_2(self):
        """Penalty should never exceed 0.2."""
        # Very LLM-like text with multiple triggers
        text = """
        As an AI, I cannot provide personal opinions. My knowledge cutoff is 2021.
        In summary, it is important to note several key points. First, the analysis
        demonstrates clear patterns. Second, the methodology ensures robust results.
        Third, further investigation is warranted. In conclusion, please feel free
        to ask if you have any further questions or need additional clarification.
        """
        penalty, llm_like, details = detect_llm_likeness(text)
        assert penalty <= 0.2
        assert llm_like is True

    def test_natural_conversational_text(self):
        """Natural, varied conversational text should have low penalty."""
        text = """
        Ugh, so today was weird. Got to work and my boss was like "we need to talk"
        and I'm thinking oh no what did I do?? But turns out it's nothing bad, just
        some new project stuff. Anyway, then later I'm getting coffee and I run into
        Sarah - haven't seen her in forever! We chatted for a bit, caught up on
        life and whatever. She's doing well I think? Hard to tell sometimes lol.
        Anyway that's my day. How's yours going?
        """
        penalty, llm_like, details = detect_llm_likeness(text)
        assert penalty <= 0.08  # Should be very low (boundary inclusive)
        assert llm_like is False

    def test_technical_writing_human(self):
        """Technical but human-written text should have moderate penalty."""
        text = """
        So I've been debugging this issue for hours - turns out it was a race
        condition in the async handler. SMH. The fix is pretty straightforward
        though: just add a lock around the critical section. I tested it locally
        and it seems to work, but I'm gonna run the full test suite tomorrow
        morning to be sure. Let me know if you spot any issues with my PR!
        """
        penalty, llm_like, details = detect_llm_likeness(text)
        assert penalty < 0.10
        assert llm_like is False


class TestFeatureExtraction:
    """Test feature extraction functions."""

    def test_feature_extraction_basic(self):
        """Test basic feature extraction."""
        text = "Hello world. This is a test. How are you?"
        features = _extract_features(text)

        assert 'sentence_length_variance' in features
        assert 'punctuation_entropy' in features
        assert 'llm_phrase_count' in features
        assert 'vocabulary_diversity' in features
        assert 'avg_sentence_length' in features
        assert 'punctuation_rate' in features
        assert 'contractions_rate' in features

    def test_sentence_length_variance_uniform(self):
        """Uniform sentence lengths should have low variance."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        features = _extract_features(text)
        assert features['sentence_length_variance'] < 1.0  # Very uniform

    def test_sentence_length_variance_varied(self):
        """Varied sentence lengths should have high variance."""
        text = "Short. This is a medium length sentence here. This is a much longer sentence with many more words in it that goes on and on."
        features = _extract_features(text)
        assert features['sentence_length_variance'] > 10  # Highly varied

    def test_llm_phrase_detection(self):
        """Should detect common LLM phrases."""
        text_with_llm = "As an AI, I cannot provide personal opinions."
        text_without_llm = "I think this is a great idea!"

        features_with = _extract_features(text_with_llm)
        features_without = _extract_features(text_without_llm)

        assert features_with['llm_phrase_count'] > 0
        assert features_without['llm_phrase_count'] == 0

    def test_vocabulary_diversity_high(self):
        """Text with varied vocabulary should have high diversity."""
        text = "The quick brown fox jumps over lazy dog near sparkling river."
        features = _extract_features(text)
        assert features['vocabulary_diversity'] > 0.8  # Most words unique

    def test_vocabulary_diversity_low(self):
        """Repetitive text should have low diversity."""
        text = "I think I think I think I think I think I think."
        features = _extract_features(text)
        assert features['vocabulary_diversity'] < 0.5  # Many repeats

    def test_contractions_rate(self):
        """Should correctly count contractions."""
        text_with = "I don't think it's a good idea. We can't do that."
        text_without = "I do not think it is a good idea. We cannot do that."

        features_with = _extract_features(text_with)
        features_without = _extract_features(text_without)

        assert features_with['contractions_rate'] > 0.1
        assert features_without['contractions_rate'] == 0.0

    def test_punctuation_rate(self):
        """Should calculate punctuation rate correctly."""
        text = "Hello! How are you? I'm fine, thanks. What about you?"
        features = _extract_features(text)
        assert features['punctuation_rate'] > 0
        assert features['punctuation_rate'] < 0.2  # Reasonable range


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    def test_get_penalty_function(self):
        """get_penalty should return just the penalty value."""
        text = "This is a test sentence with some content."
        penalty = get_penalty(text)
        assert isinstance(penalty, float)
        assert 0 <= penalty <= 0.2

    def test_is_llm_generated_default_threshold(self):
        """is_llm_generated should use default threshold of 0.10."""
        human_text = "Hey what's up? Just wanted to check in, see how you're doing!"
        llm_text = "As an AI language model, I must inform you that I cannot provide personal opinions."

        assert not is_llm_generated(human_text)
        assert is_llm_generated(llm_text)

    def test_is_llm_generated_custom_threshold(self):
        """is_llm_generated should respect custom threshold."""
        text = "This is a moderately formal text with some structure."

        # With low threshold, might classify as LLM
        # With high threshold, likely won't
        result_low = is_llm_generated(text, threshold=0.01)
        result_high = is_llm_generated(text, threshold=0.50)

        # At least one should be different unless penalty is exactly at boundary
        # This test verifies the threshold parameter works
        penalty = get_penalty(text)
        assert is_llm_generated(text, threshold=penalty - 0.01) is True
        assert is_llm_generated(text, threshold=penalty + 0.01) is False


class TestDeterminism:
    """Test deterministic behavior."""

    def test_deterministic_penalty(self):
        """Same input should always produce same penalty."""
        text = "This is a test sentence with consistent content for testing."

        penalty1, llm1, details1 = detect_llm_likeness(text)
        penalty2, llm2, details2 = detect_llm_likeness(text)
        penalty3, llm3, details3 = detect_llm_likeness(text)

        assert penalty1 == penalty2 == penalty3
        assert llm1 == llm2 == llm3

    def test_deterministic_features(self):
        """Same input should produce same features."""
        text = "Testing deterministic feature extraction behavior here."

        features1 = _extract_features(text)
        features2 = _extract_features(text)

        assert features1 == features2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_only_punctuation(self):
        """Text with only punctuation should handle gracefully."""
        text = "!!! ??? ... --- ,,, ;;;"
        penalty, llm_like, details = detect_llm_likeness(text)
        # Should not crash, penalty should be reasonable
        assert 0 <= penalty <= 0.2

    def test_single_long_sentence(self):
        """Single very long sentence should handle gracefully."""
        text = "This is one extremely long sentence that goes on and on " * 20
        penalty, llm_like, details = detect_llm_likeness(text)
        assert 0 <= penalty <= 0.2

    def test_no_punctuation(self):
        """Text without punctuation should handle gracefully."""
        text = "this is text without any punctuation at all just words"
        penalty, llm_like, details = detect_llm_likeness(text)
        assert 0 <= penalty <= 0.2

    def test_unicode_and_special_chars(self):
        """Text with unicode should handle gracefully."""
        text = "Hello 👋 こんにちは! This is a tëst with spëcial çharacters."
        penalty, llm_like, details = detect_llm_likeness(text)
        assert 0 <= penalty <= 0.2

    def test_all_caps(self):
        """All caps text should handle gracefully."""
        text = "THIS IS ALL CAPS TEXT. IT SHOULD STILL WORK FINE. TESTING NOW."
        penalty, llm_like, details = detect_llm_likeness(text)
        assert 0 <= penalty <= 0.2


class TestRealWorldExamples:
    """Test with realistic text samples."""

    def test_email_casual(self):
        """Casual email should have low penalty."""
        text = """
        Hi Sarah,

        Thanks for getting back to me! Yeah, I'd love to grab coffee next week.
        How's Tuesday afternoon work for you? Maybe around 3pm?

        BTW - did you see the news about the project? Pretty exciting stuff!

        Let me know what works for you.

        Cheers,
        Alex
        """
        penalty, llm_like, _ = detect_llm_likeness(text)
        assert penalty < 0.10
        assert llm_like is False

    def test_chatgpt_style_response(self):
        """ChatGPT-style response should have high penalty."""
        text = """
        I understand your concern. Let me provide a comprehensive explanation.

        First, it's important to note that there are several factors to consider.
        The primary consideration is the scope of the analysis. Additionally,
        we must account for various edge cases.

        In summary, the approach involves three key steps:
        1. Initial assessment
        2. Detailed analysis
        3. Final recommendations

        Please let me know if you need any further clarification on these points.
        """
        penalty, llm_like, _ = detect_llm_likeness(text)
        assert penalty >= 0.08  # Should be relatively high

    def test_blog_post_human(self):
        """Human blog post should have low to moderate penalty."""
        text = """
        Okay, so I've been playing around with this new framework and WOW.
        Just... wow. It's so much better than what I was using before!

        Don't get me wrong - the old way worked fine. But this? This is
        on another level. Setup took like 10 minutes (seriously) and
        I was already building features.

        The docs could be better tbh, but the community is super helpful.
        Already found answers to most of my questions on their Discord.

        10/10 would recommend checking it out if you're in this space.
        """
        penalty, llm_like, _ = detect_llm_likeness(text)
        assert penalty < 0.10
        assert llm_like is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
