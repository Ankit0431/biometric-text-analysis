"""
Unit tests for text normalizer.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from normalizer import (
    normalize,
    TextNormalizer,
    NormalizedText,
    RejectionReason,
    MIN_WORDS_VERIFY,
    MIN_WORDS_ENROLL,
)


class TestPIIMasking:
    """Test PII masking functionality."""

    def test_mask_email_addresses(self):
        """Test that email addresses are masked."""
        normalizer = TextNormalizer()

        text = "Contact me at john.doe@example.com for details."
        result = normalizer.mask_pii(text)

        assert "<EMAIL>" in result
        assert "john.doe@example.com" not in result
        assert "Contact me at" in result  # Preserved surrounding text
        assert "for details." in result

    def test_mask_multiple_emails(self):
        """Test masking multiple email addresses."""
        normalizer = TextNormalizer()

        text = "Send to alice@test.com and bob@example.org"
        result = normalizer.mask_pii(text)

        assert result.count("<EMAIL>") == 2
        assert "alice@test.com" not in result
        assert "bob@example.org" not in result

    def test_mask_phone_numbers(self):
        """Test that phone numbers are masked."""
        normalizer = TextNormalizer()

        text = "Call me at 555-123-4567 or (555) 987-6543"
        result = normalizer.mask_pii(text)

        assert "<NUM>" in result
        assert "555-123-4567" not in result
        assert "987-6543" not in result

    def test_mask_numeric_sequences(self):
        """Test that long numeric sequences are masked."""
        normalizer = TextNormalizer()

        text = "The order number is 123456789"
        result = normalizer.mask_pii(text)

        assert "<NUM>" in result
        assert "123456789" not in result

    def test_mask_person_names(self):
        """Test that person names are masked."""
        normalizer = TextNormalizer()

        text = "John Smith and Jane Doe will attend."
        result = normalizer.mask_pii(text)

        assert "<PER>" in result
        # Names should be masked
        assert "John Smith" not in result or "<PER>" in result

    def test_mask_organizations(self):
        """Test that organization names are masked."""
        normalizer = TextNormalizer()

        text = "I work at Acme Corp and Google Inc."
        result = normalizer.mask_pii(text)

        assert "<ORG>" in result
        assert "Acme Corp" not in result or "<ORG>" in result


class TestStructurePreservation:
    """Test that normalization preserves text structure."""

    def test_preserves_punctuation(self):
        """Test that punctuation is preserved."""
        normalizer = TextNormalizer()

        text = "Hello! How are you? I'm fine, thanks. What about you?"
        result = normalizer.mask_pii(text)

        assert "!" in result
        assert "?" in result
        assert "," in result
        assert "." in result

    def test_preserves_casing(self):
        """Test that casing is preserved in non-PII parts."""
        normalizer = TextNormalizer()

        text = "This is a TEST message with VARIOUS CaSiNg"
        result = normalizer.mask_pii(text)

        # Should preserve original casing structure
        assert "TEST" in result
        assert "VARIOUS" in result
        assert "This is a" in result

    def test_mixed_case_and_punctuation(self):
        """Test sample with punctuation and mixed case returns same structure preserved."""
        text = "Hello! This is a TEST message. It has VARIOUS punctuation: commas, periods, and exclamation marks!"

        result = normalize(text, flow_type='verify')

        # Check structure is preserved
        assert "!" in result.text
        assert "." in result.text
        assert ":" in result.text
        assert "," in result.text
        assert "TEST" in result.text
        assert "VARIOUS" in result.text
        assert "Hello" in result.text


class TestEmailArtifacts:
    """Test removal of email quoted text and signatures."""

    def test_strip_quoted_reply(self):
        """Test removal of quoted email replies."""
        normalizer = TextNormalizer()

        text = """This is my reply.

On Monday, John wrote:
> His original message
> More quoted text"""

        result = normalizer.strip_email_artifacts(text)

        assert "This is my reply." in result
        assert "On Monday, John wrote:" not in result
        assert "His original message" not in result

    def test_strip_email_signature(self):
        """Test removal of email signatures."""
        normalizer = TextNormalizer()

        text = """Here is the actual content.

--
John Doe
Senior Engineer
john@example.com"""

        result = normalizer.strip_email_artifacts(text)

        assert "Here is the actual content." in result
        assert "Senior Engineer" not in result
        assert "john@example.com" not in result

    def test_strip_signature_variants(self):
        """Test removal of various signature formats."""
        normalizer = TextNormalizer()

        variants = [
            "Content here\n\nBest regards,\nJohn",
            "Content here\n\nSincerely,\nJohn",
            "Content here\n\nThanks,\nJohn",
            "Content here\n\nRegards,\nJohn",
        ]

        for text in variants:
            result = normalizer.strip_email_artifacts(text)
            assert "Content here" in result
            assert "John" not in result or result.count("John") == 0


class TestLengthValidation:
    """Test text length validation and rejection."""

    def test_reject_short_text_verify(self):
        """Test that text with <50 words is rejected for verify flow."""
        # Create text with exactly 40 words
        text = " ".join(["word"] * 40)

        result = normalize(text, flow_type='verify')

        assert result.word_count == 40
        assert len(result.rejected_reasons) > 0
        assert RejectionReason.SHORT_LEN in result.rejected_reasons
        assert RejectionReason.TOO_SHORT_VERIFY in result.rejected_reasons

    def test_accept_sufficient_text_verify(self):
        """Test that text with >=50 words is accepted for verify flow."""
        # Create text with exactly 50 words
        text = " ".join(["word"] * 50)

        result = normalize(text, flow_type='verify')

        assert result.word_count == 50
        assert len(result.rejected_reasons) == 0

    def test_reject_short_text_enroll(self):
        """Test that text with <70 words is rejected for enroll flow."""
        # Create text with exactly 60 words
        text = " ".join(["word"] * 60)

        result = normalize(text, flow_type='enroll')

        assert result.word_count == 60
        assert len(result.rejected_reasons) > 0
        assert RejectionReason.SHORT_LEN in result.rejected_reasons
        assert RejectionReason.TOO_SHORT_ENROLL in result.rejected_reasons

    def test_accept_sufficient_text_enroll(self):
        """Test that text with >=70 words is accepted for enroll flow."""
        # Create text with exactly 70 words
        text = " ".join(["word"] * 70)

        result = normalize(text, flow_type='enroll')

        assert result.word_count == 70
        assert len(result.rejected_reasons) == 0

    def test_exact_threshold_verify(self):
        """Test exact threshold for verify flow (50 words)."""
        text = " ".join(["word"] * MIN_WORDS_VERIFY)
        result = normalize(text, flow_type='verify')

        assert result.word_count == MIN_WORDS_VERIFY
        assert len(result.rejected_reasons) == 0

    def test_exact_threshold_enroll(self):
        """Test exact threshold for enroll flow (70 words)."""
        text = " ".join(["word"] * MIN_WORDS_ENROLL)
        result = normalize(text, flow_type='enroll')

        assert result.word_count == MIN_WORDS_ENROLL
        assert len(result.rejected_reasons) == 0


class TestLanguageDetection:
    """Test language detection."""

    def test_detect_english(self):
        """Test detection of English text."""
        normalizer = TextNormalizer()

        text = "This is a sample text in English with common words."
        lang = normalizer.detect_language(text)

        assert lang == "en"

    def test_default_language(self):
        """Test that default language is returned for ambiguous text."""
        normalizer = TextNormalizer()

        text = "xyz abc def"  # Ambiguous text
        lang = normalizer.detect_language(text)

        assert lang == "en"  # Default


class TestTokenization:
    """Test tokenization."""

    def test_basic_tokenization(self):
        """Test basic word tokenization."""
        normalizer = TextNormalizer()

        text = "Hello world this is a test"
        tokens = normalizer.tokenize(text)

        assert len(tokens) == 6
        assert tokens == ["Hello", "world", "this", "is", "a", "test"]

    def test_tokenization_with_punctuation(self):
        """Test tokenization preserves punctuation with words."""
        normalizer = TextNormalizer()

        text = "Hello, world! How are you?"
        tokens = normalizer.tokenize(text)

        assert len(tokens) > 0
        assert any("," in token or token == "Hello," for token in tokens)


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_normalization_with_pii(self):
        """Test complete normalization with PII masking."""
        text = """Hi, my name is John Smith and my email is john@example.com.
        My phone number is 555-1234 and I work at Acme Corp.
        Please contact me to discuss the proposal with more details and information.
        I think we can work together on this project and make it successful.
        Looking forward to hearing from you soon about the next steps."""

        result = normalize(text, flow_type='verify')

        # Should mask PII
        assert "<EMAIL>" in result.text or "john@example.com" not in result.text
        assert "<NUM>" in result.text or "555-1234" not in result.text

        # Should have tokens
        assert len(result.tokens) > 0
        assert result.word_count > 0

        # Should have language
        assert result.lang == "en"

        # Should not be rejected (has enough words)
        assert len(result.rejected_reasons) == 0

    def test_normalization_exact_match(self):
        """Test known input with PII produces expected masked output."""
        text = "Please email me at test@example.com with the number 12345"

        result = normalize(text, flow_type='verify')

        # Expected output should mask email and number
        assert "test@example.com" not in result.text
        assert "<EMAIL>" in result.text
        assert "12345" not in result.text
        assert "<NUM>" in result.text
        assert "Please email me at" in result.text
        assert "with the number" in result.text

    def test_short_sample_rejection(self):
        """Test that sample with <50 words includes rejection reason SHORT_LEN."""
        text = "This is a very short text sample."  # Only 7 words

        result = normalize(text, flow_type='verify')

        assert result.word_count < 50
        assert RejectionReason.SHORT_LEN in result.rejected_reasons
        assert len(result.rejected_reasons) > 0


class TestNormalizedTextStructure:
    """Test the NormalizedText dataclass."""

    def test_normalized_text_fields(self):
        """Test that NormalizedText has all required fields."""
        text = " ".join(["word"] * 60)
        result = normalize(text, flow_type='verify')

        assert hasattr(result, 'text')
        assert hasattr(result, 'tokens')
        assert hasattr(result, 'word_count')
        assert hasattr(result, 'rejected_reasons')
        assert hasattr(result, 'lang')

        assert isinstance(result.text, str)
        assert isinstance(result.tokens, list)
        assert isinstance(result.word_count, int)
        assert isinstance(result.rejected_reasons, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
