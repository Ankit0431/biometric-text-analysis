"""
Text normalizer for biometric text analysis.

This module handles:
- PII masking (emails, phone numbers, names, organizations)
- Email quoted text and signature removal
- Language detection
- Text quality checks and rejection
"""
import re
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum


class RejectionReason(str, Enum):
    """Reasons why a text sample might be rejected."""
    SHORT_LEN = "SHORT_LEN"  # Too few words
    TOO_SHORT_VERIFY = "TOO_SHORT_VERIFY"  # < 50 words for verify
    TOO_SHORT_ENROLL = "TOO_SHORT_ENROLL"  # < 70 words for enroll


# Configurable thresholds
MIN_WORDS_VERIFY = 50
MIN_WORDS_ENROLL = 70


@dataclass
class NormalizedText:
    """Result of text normalization."""
    text: str  # Normalized text with PII masked
    tokens: List[str]  # Tokenized words
    word_count: int  # Number of words
    rejected_reasons: List[str]  # Reasons for rejection (empty if accepted)
    lang: Optional[str] = None  # Detected language


class TextNormalizer:
    """Handles text normalization and PII masking."""

    def __init__(self):
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )

        # Phone number patterns (various formats)
        self.phone_pattern = re.compile(
            r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|'
            r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b|'
            r'\b\d{10,}\b'
        )

        # Numeric sequences (longer than 3 digits)
        self.number_pattern = re.compile(r'\b\d{4,}\b')

        # Email quoted text patterns
        self.email_quote_patterns = [
            re.compile(r'^On .+wrote:.*$', re.MULTILINE | re.DOTALL),
            re.compile(r'^From:.+?(?=\n\n|\Z)', re.MULTILINE | re.DOTALL),
            re.compile(r'^[-_]{2,}.*?(?=\n\n|\Z)', re.MULTILINE | re.DOTALL),
            re.compile(r'^>+.*?$', re.MULTILINE),
        ]

        # Email signature patterns
        self.signature_patterns = [
            re.compile(r'\n--\s*\n.*$', re.DOTALL),
            re.compile(r'\nBest regards,.*$', re.DOTALL | re.IGNORECASE),
            re.compile(r'\nSincerely,.*$', re.DOTALL | re.IGNORECASE),
            re.compile(r'\nThanks,.*$', re.DOTALL | re.IGNORECASE),
            re.compile(r'\nRegards,.*$', re.DOTALL | re.IGNORECASE),
        ]

        # Common person name patterns (simple heuristic)
        # Matches capitalized words that could be names
        self.name_pattern = re.compile(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        )

        # Organization patterns (simplified)
        self.org_pattern = re.compile(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Company|Co|Group|Associates)\.?))\b'
        )

    def strip_email_artifacts(self, text: str) -> str:
        """
        Remove email quoted text and signatures.

        Args:
            text: Input text that may contain email artifacts

        Returns:
            Text with email artifacts removed
        """
        result = text

        # Remove quoted text
        for pattern in self.email_quote_patterns:
            result = pattern.sub('', result)

        # Remove signatures
        for pattern in self.signature_patterns:
            result = pattern.sub('', result)

        return result.strip()

    def mask_pii(self, text: str) -> str:
        """
        Mask PII while preserving punctuation and casing structure.

        Args:
            text: Input text with potential PII

        Returns:
            Text with PII masked
        """
        result = text

        # Mask emails first (before numbers to avoid partial matches)
        result = self.email_pattern.sub('<EMAIL>', result)

        # Mask phone numbers
        result = self.phone_pattern.sub('<NUM>', result)

        # Mask organizations (before names to catch company names)
        result = self.org_pattern.sub('<ORG>', result)

        # Mask person names (simple heuristic - capitalized words)
        # This is a simplified approach; production would use NER
        result = self.name_pattern.sub('<PER>', result)

        # Mask remaining numeric sequences
        result = self.number_pattern.sub('<NUM>', result)

        return result

    def detect_language(self, text: str) -> str:
        """
        Detect language of text.

        For prototype, returns a simple heuristic.
        In production, use cld3 or similar library.

        Args:
            text: Input text

        Returns:
            Language code (e.g., 'en')
        """
        # Simple heuristic: check for common English words
        english_indicators = ['the', 'and', 'is', 'to', 'in', 'a', 'of']
        text_lower = text.lower()

        english_count = sum(1 for word in english_indicators if f' {word} ' in f' {text_lower} ')

        # If we find multiple English indicators, assume English
        if english_count >= 2:
            return 'en'

        # Default to English for prototype
        return 'en'

    def tokenize(self, text: str) -> List[str]:
        """
        Simple word tokenization.

        Args:
            text: Input text

        Returns:
            List of word tokens
        """
        # Simple whitespace tokenization preserving basic structure
        # Split on whitespace and filter out empty strings
        tokens = [token for token in re.split(r'\s+', text) if token.strip()]
        return tokens

    def normalize(self, text: str, flow_type: str = 'verify') -> NormalizedText:
        """
        Normalize text: strip artifacts, mask PII, detect language, check quality.

        Args:
            text: Raw input text
            flow_type: 'verify' or 'enroll' (affects length threshold)

        Returns:
            NormalizedText object with normalized text and metadata
        """
        # Strip email artifacts
        cleaned = self.strip_email_artifacts(text)

        # Mask PII while preserving structure
        masked = self.mask_pii(cleaned)

        # Tokenize
        tokens = self.tokenize(masked)
        word_count = len(tokens)

        # Detect language
        lang = self.detect_language(cleaned)

        # Check rejection criteria
        rejected_reasons = []

        if flow_type == 'verify' and word_count < MIN_WORDS_VERIFY:
            rejected_reasons.append(RejectionReason.TOO_SHORT_VERIFY)
            rejected_reasons.append(RejectionReason.SHORT_LEN)
        elif flow_type == 'enroll' and word_count < MIN_WORDS_ENROLL:
            rejected_reasons.append(RejectionReason.TOO_SHORT_ENROLL)
            rejected_reasons.append(RejectionReason.SHORT_LEN)

        return NormalizedText(
            text=masked,
            tokens=tokens,
            word_count=word_count,
            rejected_reasons=rejected_reasons,
            lang=lang,
        )


# Global normalizer instance
normalizer = TextNormalizer()


def normalize(text: str, flow_type: str = 'verify') -> NormalizedText:
    """
    Convenience function for text normalization.

    Args:
        text: Raw input text
        flow_type: 'verify' or 'enroll' (affects length threshold)

    Returns:
        NormalizedText object
    """
    return normalizer.normalize(text, flow_type)
