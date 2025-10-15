"""
Classical stylometry feature extraction.

This module computes deterministic stylometric features from normalized text:
- Character n-grams (3-5 grams)
- Function word frequencies
- Sentence statistics
- Punctuation rates
- Contraction rates
- Emoji/URL rates
- Entropy measures
- POS trigram features (simplified)
"""
import re
import json
import os
import hashlib
from typing import Dict, List, Tuple
from collections import Counter
from pathlib import Path
import numpy as np
import math


# Feature vector configuration
CHAR_NGRAM_DIMS = 256  # Character n-gram features via hashing
FUNCTION_WORD_DIMS = 100  # Top function words
STATISTICAL_DIMS = 50  # Sentence stats, punctuation, etc.
POS_TRIGRAM_DIMS = 106  # POS trigram features (simplified)
TOTAL_FEATURE_DIMS = 512  # Total feature vector dimension


def load_function_words(lang: str = "en") -> List[str]:
    """
    Load function words for a given language.

    Args:
        lang: Language code (e.g., 'en')

    Returns:
        List of function words
    """
    data_dir = Path(__file__).parent / "data"
    filepath = data_dir / f"function_words.{lang}.json"

    if not filepath.exists():
        # Fallback to a minimal set
        return ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i"]

    with open(filepath, 'r') as f:
        data = json.load(f)
        return data.get(lang, [])


class StyleFeatureExtractor:
    """Extract classical stylometry features from text."""

    def __init__(self, lang: str = "en"):
        """
        Initialize the feature extractor.

        Args:
            lang: Language code
        """
        self.lang = lang
        self.function_words = load_function_words(lang)[:FUNCTION_WORD_DIMS]

        # Punctuation characters to track
        self.punctuation = set('.,;:!?-—–()[]{}"\'"''""')

        # Common contractions
        self.contractions = {
            "n't", "'t", "'re", "'ve", "'ll", "'d", "'m", "'s",
            "won't", "can't", "don't", "isn't", "aren't", "wasn't", "weren't",
            "haven't", "hasn't", "hadn't", "won't", "wouldn't", "don't", "doesn't", "didn't",
            "I'm", "you're", "he's", "she's", "it's", "we're", "they're",
            "I've", "you've", "we've", "they've", "I'll", "you'll", "he'll",
            "she'll", "it'll", "we'll", "they'll", "I'd", "you'd", "he'd", "she'd"
        }

    def extract_char_ngrams(self, text: str, n_min: int = 3, n_max: int = 5) -> np.ndarray:
        """
        Extract character n-gram features using hashing trick.

        Args:
            text: Input text
            n_min: Minimum n-gram size
            n_max: Maximum n-gram size

        Returns:
            Feature vector of character n-gram counts
        """
        # Initialize feature vector
        features = np.zeros(CHAR_NGRAM_DIMS, dtype=np.float32)

        # Clean text (lowercase for char n-grams)
        text_clean = text.lower()

        # Extract n-grams
        for n in range(n_min, n_max + 1):
            for i in range(len(text_clean) - n + 1):
                ngram = text_clean[i:i+n]
                # Hash to feature index
                hash_val = hashlib.md5(ngram.encode()).hexdigest()
                idx = int(hash_val, 16) % CHAR_NGRAM_DIMS
                features[idx] += 1

        # Normalize by total n-grams
        total = features.sum()
        if total > 0:
            features = features / total

        return features

    def extract_function_word_features(self, tokens: List[str]) -> np.ndarray:
        """
        Extract function word frequency features.

        Args:
            tokens: List of word tokens

        Returns:
            Feature vector of function word frequencies
        """
        features = np.zeros(FUNCTION_WORD_DIMS, dtype=np.float32)

        # Count tokens
        token_counts = Counter(token.lower() for token in tokens)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return features

        # Extract function word frequencies
        for i, word in enumerate(self.function_words):
            if i >= FUNCTION_WORD_DIMS:
                break
            features[i] = token_counts.get(word, 0) / total_tokens

        return features

    def extract_sentence_stats(self, text: str) -> Dict[str, float]:
        """
        Extract sentence-level statistics.

        Args:
            text: Input text

        Returns:
            Dictionary of sentence statistics
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {
                'mean_sentence_len': 0.0,
                'std_sentence_len': 0.0,
                'max_sentence_len': 0.0,
                'min_sentence_len': 0.0,
                'num_sentences': 0,
            }

        # Calculate sentence lengths in words
        sentence_lens = [len(s.split()) for s in sentences]

        return {
            'mean_sentence_len': np.mean(sentence_lens),
            'std_sentence_len': np.std(sentence_lens),
            'max_sentence_len': max(sentence_lens),
            'min_sentence_len': min(sentence_lens),
            'num_sentences': len(sentences),
        }

    def extract_punctuation_features(self, text: str, tokens: List[str]) -> Dict[str, float]:
        """
        Extract punctuation-based features.

        Args:
            text: Input text
            tokens: List of tokens

        Returns:
            Dictionary of punctuation features
        """
        total_chars = len(text)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return {
                'punct_rate_per_1k': 0.0,
                'comma_rate': 0.0,
                'period_rate': 0.0,
                'question_rate': 0.0,
                'exclamation_rate': 0.0,
            }

        # Count punctuation
        punct_counts = Counter(c for c in text if c in self.punctuation)
        total_punct = sum(punct_counts.values())

        return {
            'punct_rate_per_1k': (total_punct / total_tokens) * 1000 if total_tokens > 0 else 0.0,
            'comma_rate': punct_counts.get(',', 0) / total_tokens,
            'period_rate': punct_counts.get('.', 0) / total_tokens,
            'question_rate': punct_counts.get('?', 0) / total_tokens,
            'exclamation_rate': punct_counts.get('!', 0) / total_tokens,
        }

    def extract_contraction_features(self, text: str, tokens: List[str]) -> Dict[str, float]:
        """
        Extract contraction-based features.

        Args:
            text: Input text
            tokens: List of tokens

        Returns:
            Dictionary of contraction features
        """
        total_tokens = len(tokens)

        if total_tokens == 0:
            return {'contraction_rate': 0.0}

        # Count contractions
        contraction_count = 0
        for token in tokens:
            if any(contr in token for contr in self.contractions):
                contraction_count += 1

        return {
            'contraction_rate': contraction_count / total_tokens,
        }

    def extract_special_char_features(self, text: str, tokens: List[str]) -> Dict[str, float]:
        """
        Extract emoji and URL features.

        Args:
            text: Input text
            tokens: List of tokens

        Returns:
            Dictionary of special character features
        """
        total_tokens = len(tokens)

        if total_tokens == 0:
            return {
                'emoji_rate': 0.0,
                'url_rate': 0.0,
                'number_rate': 0.0,
            }

        # Count emojis (simplified - check for common emoji ranges)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE
        )
        emoji_count = len(emoji_pattern.findall(text))

        # Count URLs
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        url_count = len(url_pattern.findall(text))

        # Count numeric tokens
        number_count = sum(1 for token in tokens if re.match(r'^\d+$', token))

        return {
            'emoji_rate': emoji_count / total_tokens,
            'url_rate': url_count / total_tokens,
            'number_rate': number_count / total_tokens,
        }

    def extract_entropy_features(self, tokens: List[str]) -> Dict[str, float]:
        """
        Extract entropy-based features.

        Args:
            tokens: List of tokens

        Returns:
            Dictionary of entropy features
        """
        if not tokens:
            return {
                'token_entropy': 0.0,
                'unique_token_ratio': 0.0,
            }

        # Token frequency distribution
        token_counts = Counter(token.lower() for token in tokens)
        total_tokens = len(tokens)

        # Calculate Shannon entropy
        entropy = 0.0
        for count in token_counts.values():
            p = count / total_tokens
            entropy -= p * math.log2(p)

        # Unique token ratio (vocabulary richness)
        unique_ratio = len(token_counts) / total_tokens

        return {
            'token_entropy': entropy,
            'unique_token_ratio': unique_ratio,
        }

    def extract_word_length_features(self, tokens: List[str]) -> Dict[str, float]:
        """
        Extract word length statistics.

        Args:
            tokens: List of tokens

        Returns:
            Dictionary of word length features
        """
        if not tokens:
            return {
                'mean_word_len': 0.0,
                'std_word_len': 0.0,
                'max_word_len': 0.0,
            }

        word_lens = [len(token) for token in tokens]

        return {
            'mean_word_len': np.mean(word_lens),
            'std_word_len': np.std(word_lens),
            'max_word_len': max(word_lens),
        }

    def extract_pos_trigram_features(self, tokens: List[str]) -> np.ndarray:
        """
        Extract POS (Part-of-Speech) trigram features.

        This is a simplified version using heuristic POS tagging.
        In production, use spaCy or similar NLP library.

        Args:
            tokens: List of tokens

        Returns:
            Feature vector of POS trigram counts
        """
        features = np.zeros(POS_TRIGRAM_DIMS, dtype=np.float32)

        # Simple heuristic POS tagging
        def simple_pos(token: str) -> str:
            """Simple POS tagger based on heuristics."""
            token_lower = token.lower()

            # Function words
            if token_lower in self.function_words:
                return 'FUNC'
            # Punctuation
            elif all(c in self.punctuation for c in token):
                return 'PUNCT'
            # Numbers
            elif re.match(r'^\d+$', token):
                return 'NUM'
            # Capitalized (potential proper noun)
            elif token[0].isupper() and len(token) > 1:
                return 'PROP'
            # Ends with -ing (potential verb)
            elif token_lower.endswith('ing'):
                return 'VERB'
            # Ends with -ly (potential adverb)
            elif token_lower.endswith('ly'):
                return 'ADV'
            # Ends with -ed (potential past tense)
            elif token_lower.endswith('ed'):
                return 'VERB'
            # Default: noun
            else:
                return 'NOUN'

        # Tag tokens
        pos_tags = [simple_pos(token) for token in tokens]

        # Extract trigrams
        for i in range(len(pos_tags) - 2):
            trigram = f"{pos_tags[i]}_{pos_tags[i+1]}_{pos_tags[i+2]}"
            # Hash to feature index
            hash_val = hashlib.md5(trigram.encode()).hexdigest()
            idx = int(hash_val, 16) % POS_TRIGRAM_DIMS
            features[idx] += 1

        # Normalize
        total = features.sum()
        if total > 0:
            features = features / total

        return features

    def extract_statistical_features(self, text: str, tokens: List[str]) -> np.ndarray:
        """
        Combine all statistical features into a single vector.

        Args:
            text: Input text
            tokens: List of tokens

        Returns:
            Feature vector of statistical features
        """
        # Collect all statistics
        stats = {}
        stats.update(self.extract_sentence_stats(text))
        stats.update(self.extract_punctuation_features(text, tokens))
        stats.update(self.extract_contraction_features(text, tokens))
        stats.update(self.extract_special_char_features(text, tokens))
        stats.update(self.extract_entropy_features(tokens))
        stats.update(self.extract_word_length_features(tokens))

        # Convert to vector (fixed order)
        feature_keys = [
            'mean_sentence_len', 'std_sentence_len', 'max_sentence_len', 'min_sentence_len', 'num_sentences',
            'punct_rate_per_1k', 'comma_rate', 'period_rate', 'question_rate', 'exclamation_rate',
            'contraction_rate', 'emoji_rate', 'url_rate', 'number_rate',
            'token_entropy', 'unique_token_ratio', 'mean_word_len', 'std_word_len', 'max_word_len',
        ]

        features = np.zeros(STATISTICAL_DIMS, dtype=np.float32)
        for i, key in enumerate(feature_keys):
            if i >= STATISTICAL_DIMS:
                break
            features[i] = stats.get(key, 0.0)

        return features, stats

    def extract_features(self, text: str, tokens: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Extract all stylometry features.

        Args:
            text: Input text
            tokens: List of word tokens

        Returns:
            Tuple of (feature_vector, stats_dict)
        """
        # Extract all feature components
        char_ngram_features = self.extract_char_ngrams(text)
        function_word_features = self.extract_function_word_features(tokens)
        statistical_features, stats = self.extract_statistical_features(text, tokens)
        pos_trigram_features = self.extract_pos_trigram_features(tokens)

        # Concatenate into single vector
        feature_vector = np.concatenate([
            char_ngram_features,
            function_word_features,
            statistical_features,
            pos_trigram_features,
        ])

        # Ensure correct dimensions
        assert len(feature_vector) == TOTAL_FEATURE_DIMS, \
            f"Expected {TOTAL_FEATURE_DIMS} dims, got {len(feature_vector)}"

        return feature_vector.astype(np.float32), stats


# Global extractor instance
_extractor = None


def get_extractor(lang: str = "en") -> StyleFeatureExtractor:
    """Get or create global feature extractor."""
    global _extractor
    if _extractor is None or _extractor.lang != lang:
        _extractor = StyleFeatureExtractor(lang)
    return _extractor


def extract_features(text: str, tokens: List[str], lang: str = "en") -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to extract stylometry features.

    Args:
        text: Input text
        tokens: List of word tokens
        lang: Language code

    Returns:
        Tuple of (feature_vector, stats_dict)
    """
    extractor = get_extractor(lang)
    return extractor.extract_features(text, tokens)
