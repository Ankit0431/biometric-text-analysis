"""
Hybrid LLM Detection System

This module implements a hybrid approach to detect LLM-generated text by combining:
1. Machine Learning classification using GradientBoostingClassifier
2. Heuristic-based detection rules
3. Advanced linguistic features (sentence variance, POS entropy, perplexity, etc.)

The system provides more accurate and robust LLM detection compared to heuristics alone.
"""
import numpy as np
import re
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import logging

# Optional imports - will use fallbacks if not available
try:
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    joblib = None
    GradientBoostingClassifier = None
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    nltk = None
    word_tokenize = None
    sent_tokenize = None
    pos_tag = None
    NLTK_AVAILABLE = False

try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    GPT2LMHeadModel = None
    GPT2TokenizerFast = None
    torch = None
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global model variables
_llm_model = None
_perplexity_model = None
_perplexity_tokenizer = None


def _ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    if not NLTK_AVAILABLE:
        return False
    
    try:
        import nltk
        # Try to use the data, download if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        
        return True
    except Exception as e:
        logger.warning(f"Failed to ensure NLTK data: {e}")
        return False


def _load_perplexity_model():
    """Load the perplexity model (DistilGPT-2) for text evaluation."""
    global _perplexity_model, _perplexity_tokenizer
    
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers not available, perplexity features will be disabled")
        return False
    
    if _perplexity_model is not None:
        return True
    
    try:
        # Use DistilGPT-2 for faster inference
        model_name = "distilgpt2"
        _perplexity_tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        _perplexity_model = GPT2LMHeadModel.from_pretrained(model_name)
        _perplexity_model.eval()
        
        # Add padding token
        _perplexity_tokenizer.pad_token = _perplexity_tokenizer.eos_token
        
        logger.info("Perplexity model loaded successfully")
        return True
    except Exception as e:
        logger.warning(f"Failed to load perplexity model: {e}")
        return False


def _compute_perplexity(text: str, max_length: int = 512) -> float:
    """
    Compute perplexity of text using DistilGPT-2.
    
    Args:
        text: Input text
        max_length: Maximum sequence length
        
    Returns:
        Perplexity score (lower = more likely to be human-written)
    """
    if not _load_perplexity_model() or not text.strip():
        return 50.0  # Default moderate perplexity
    
    try:
        # Tokenize and truncate
        inputs = _perplexity_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length,
            padding=True
        )
        
        with torch.no_grad():
            outputs = _perplexity_model(**inputs)
            logits = outputs.logits
            
            # Shift for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            
            # Compute cross entropy loss
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            perplexity = torch.exp(loss).item()
            
            # Clamp to reasonable range
            return float(np.clip(perplexity, 1.0, 1000.0))
            
    except Exception as e:
        logger.warning(f"Perplexity computation failed: {e}")
        return 50.0


def extract_llm_features(text: str) -> np.ndarray:
    """
    Extract comprehensive features for LLM detection.
    
    Features include:
    - Sentence length statistics (mean, std, cv, entropy)
    - Punctuation patterns (entropy, ratios)
    - Word-level statistics (length variance, burstiness)
    - Function word ratios and patterns
    - POS tag entropy (if NLTK available)
    - Perplexity (if transformers available)
    - Syntactic complexity measures
    - Lexical diversity measures
    
    Args:
        text: Input text to analyze
        
    Returns:
        Feature vector of shape (20,) containing normalized features
    """
    features = np.zeros(20, dtype=np.float32)
    
    if not text or len(text.strip()) < 10:
        return features
    
    text = text.strip()
    
    # Basic tokenization (fallback if NLTK not available)
    if NLTK_AVAILABLE and _ensure_nltk_data():
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
        except:
            # Fallback to simple tokenization
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            words = re.findall(r'\b\w+\b', text.lower())
    else:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = re.findall(r'\b\w+\b', text.lower())
    
    if len(sentences) < 2 or len(words) < 5:
        return features
    
    # Feature 0-3: Sentence length statistics
    sentence_lengths = [len(s.split()) for s in sentences]
    if sentence_lengths:
        mean_sent_len = np.mean(sentence_lengths)
        std_sent_len = np.std(sentence_lengths)
        features[0] = min(mean_sent_len / 50.0, 1.0)  # Normalize by max reasonable length
        features[1] = min(std_sent_len / 20.0, 1.0)   # Normalize standard deviation
        
        # Coefficient of variation (consistency measure)
        if mean_sent_len > 0:
            features[2] = min((std_sent_len / mean_sent_len), 2.0) / 2.0
        
        # Sentence length entropy
        if len(set(sentence_lengths)) > 1:
            sent_counts = Counter(sentence_lengths)
            total = sum(sent_counts.values())
            probs = [count / total for count in sent_counts.values()]
            features[3] = min(-sum(p * math.log2(p) for p in probs if p > 0) / 5.0, 1.0)
    
    # Feature 4-6: Punctuation patterns
    punctuation_chars = [c for c in text if c in '.,;:!?-()[]{}"\'-']
    if punctuation_chars:
        punct_counts = Counter(punctuation_chars)
        total_punct = len(punctuation_chars)
        
        # Punctuation entropy
        probs = [count / total_punct for count in punct_counts.values()]
        punct_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        features[4] = min(punct_entropy / 4.0, 1.0)
        
        # Comma ratio (LLMs often overuse commas)
        features[5] = min(punct_counts.get(',', 0) / total_punct, 1.0)
        
        # Punctuation density
        features[6] = min(len(punctuation_chars) / len(text), 0.2) / 0.2
    
    # Feature 7-9: Word-level statistics
    word_lengths = [len(w) for w in words if w.isalpha()]
    if word_lengths:
        mean_word_len = np.mean(word_lengths)
        std_word_len = np.std(word_lengths)
        features[7] = min(mean_word_len / 10.0, 1.0)
        features[8] = min(std_word_len / 5.0, 1.0)
        
        # Word length coefficient of variation
        if mean_word_len > 0:
            features[9] = min((std_word_len / mean_word_len), 2.0) / 2.0
    
    # Feature 10-12: Function word analysis
    function_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
        'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
        'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
        'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would',
        'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about',
        'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can',
        'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people',
        'into', 'year', 'your', 'good', 'some', 'could', 'them',
        'see', 'other', 'than', 'then', 'now', 'look', 'only',
        'come', 'its', 'over', 'think', 'also', 'back', 'after',
        'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
        'even', 'new', 'want', 'because', 'any', 'these', 'give',
        'day', 'most', 'us'
    }
    
    function_word_count = sum(1 for w in words if w in function_words)
    total_words = len(words)
    if total_words > 0:
        features[10] = function_word_count / total_words
        
        # Unique function word ratio
        unique_function_words = len(set(w for w in words if w in function_words))
        features[11] = unique_function_words / len(function_words)
        
        # Word repetition (burstiness)
        word_counts = Counter(words)
        max_word_freq = max(word_counts.values()) if word_counts else 1
        features[12] = min(max_word_freq / total_words, 0.5) / 0.5
    
    # Feature 13-15: POS tag analysis (if NLTK available)
    if NLTK_AVAILABLE and _ensure_nltk_data():
        try:
            # Get POS tags for a sample of words (to avoid performance issues)
            sample_words = words[:100] if len(words) > 100 else words
            pos_tags = pos_tag(sample_words)
            
            if pos_tags:
                # POS tag entropy
                pos_counts = Counter(tag for word, tag in pos_tags)
                total_pos = len(pos_tags)
                pos_probs = [count / total_pos for count in pos_counts.values()]
                pos_entropy = -sum(p * math.log2(p) for p in pos_probs if p > 0)
                features[13] = min(pos_entropy / 4.0, 1.0)
                
                # Noun ratio (LLMs sometimes use fewer nouns)
                noun_count = sum(1 for word, tag in pos_tags if tag.startswith('N'))
                features[14] = noun_count / total_pos if total_pos > 0 else 0
                
                # Verb ratio
                verb_count = sum(1 for word, tag in pos_tags if tag.startswith('V'))
                features[15] = verb_count / total_pos if total_pos > 0 else 0
        except Exception as e:
            logger.debug(f"POS tagging failed: {e}")
    
    # Feature 16: Perplexity (if transformers available)
    if TRANSFORMERS_AVAILABLE:
        perplexity = _compute_perplexity(text)
        # Normalize perplexity (lower = more likely AI, higher = more likely human)
        # Typical range: 10-100, with AI often having lower perplexity
        features[16] = min(perplexity / 100.0, 1.0)
    else:
        features[16] = 0.5  # Neutral if unavailable
    
    # Feature 17-19: Additional complexity measures
    # Lexical diversity (Type-Token Ratio)
    unique_words = len(set(words))
    features[17] = unique_words / total_words if total_words > 0 else 0
    
    # Average words per sentence
    avg_words_per_sent = total_words / len(sentences) if sentences else 0
    features[18] = min(avg_words_per_sent / 30.0, 1.0)
    
    # Character-level entropy
    char_counts = Counter(text.lower())
    total_chars = len(text)
    if total_chars > 0:
        char_probs = [count / total_chars for count in char_counts.values()]
        char_entropy = -sum(p * math.log2(p) for p in char_probs if p > 0)
        features[19] = min(char_entropy / 6.0, 1.0)  # Normalize by typical max entropy
    
    return features


def heuristic_llm_detection(text: str) -> Tuple[float, bool]:
    """
    Fast heuristic-based LLM detection as fallback.
    
    This is the original heuristic system adapted for the hybrid approach.
    Uses statistical patterns that are quick to compute.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Tuple of (penalty_score, is_llm_like)
        - penalty_score: Float in [0, 1] indicating LLM-likeness
        - is_llm_like: Boolean flag for binary classification
    """
    if not text or len(text.strip()) < 20:
        return 0.0, False
    
    # Split into sentences
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    
    if len(sentences) < 2:
        return 0.0, False
    
    penalty = 0.0
    llm_signals = []
    
    # 1. Sentence length variance
    sentence_lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(sentence_lengths)
    std_len = np.std(sentence_lengths)
    coefficient_of_variation = std_len / (mean_len + 1e-6)
    
    if coefficient_of_variation < 0.30 and len(sentences) >= 3:
        penalty += 0.25
        llm_signals.append(f"low_sent_variance={coefficient_of_variation:.3f}")
    
    # 2. Punctuation consistency
    punctuation_chars = [c for c in text if c in '.,;:!?-']
    if len(punctuation_chars) > 5:
        punct_counts = Counter(punctuation_chars)
        total = sum(punct_counts.values())
        punct_probs = [count / total for count in punct_counts.values()]
        punct_entropy = -sum(p * math.log2(p) for p in punct_probs if p > 0)
        
        if punct_entropy < 1.5:
            penalty += 0.20
            llm_signals.append(f"low_punct_entropy={punct_entropy:.3f}")
    
    # 3. Formal LLM phrases
    llm_indicators = [
        'furthermore', 'moreover', 'additionally', 'in conclusion',
        'it is important to note', 'it should be noted', 'as mentioned',
        'in summary', 'to summarize', 'in other words', 'that being said',
        'on the other hand', 'in contrast', 'similarly', 'likewise',
        'consequently', 'therefore', 'thus', 'hence', 'accordingly'
    ]
    text_lower = text.lower()
    llm_phrase_count = sum(1 for phrase in llm_indicators if phrase in text_lower)
    llm_phrase_ratio = llm_phrase_count / max(len(sentences), 1)
    
    if llm_phrase_ratio > 0.15:
        penalty += 0.30
        llm_signals.append(f"llm_phrases={llm_phrase_ratio:.3f}")
    
    # 4. Perfect grammar indicators
    words = text.split()
    contractions = ["don't", "can't", "won't", "didn't", "wouldn't", "shouldn't", "isn't", "aren't"]
    has_contractions = any(c in text.lower() for c in contractions)
    
    if not has_contractions and len(words) > 30:
        penalty += 0.15
        llm_signals.append("no_contractions")
    
    # 5. Sophisticated vocabulary
    long_words = [w for w in words if len(w.strip('.,!?;:')) > 8]
    long_word_ratio = len(long_words) / max(len(words), 1)
    
    if long_word_ratio > 0.12:
        penalty += 0.20
        llm_signals.append(f"sophisticated_vocab={long_word_ratio:.3f}")
    
    penalty = min(penalty, 1.0)
    is_llm_like = penalty > 0.35
    
    if llm_signals:
        logger.debug(f"Heuristic LLM detection: penalty={penalty:.3f}, signals={llm_signals}")
    
    return penalty, is_llm_like


def load_llm_model(model_path: str = "models/llm_detector.pkl") -> bool:
    """
    Load the trained LLM detection model.
    
    Args:
        model_path: Path to the trained model file
        
    Returns:
        True if model loaded successfully, False otherwise
    """
    global _llm_model
    
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available, using heuristic-only detection")
        return False
    
    try:
        _llm_model = joblib.load(model_path)
        logger.info(f"LLM detection model loaded from {model_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to load LLM model from {model_path}: {e}")
        return False


def detect_llm_likeness(
    text: str, 
    use_ml: bool = True,
    ml_weight: float = 0.6,
    heuristic_weight: float = 0.4
) -> Tuple[float, bool]:
    """
    Hybrid LLM detection combining ML classification and heuristic rules.
    
    This is the main detection function that combines:
    1. Machine learning prediction (if model available)
    2. Heuristic rule-based detection
    3. Weighted combination of both approaches
    
    Args:
        text: Input text to analyze
        use_ml: Whether to use ML model (if available)
        ml_weight: Weight for ML prediction (default 0.6)
        heuristic_weight: Weight for heuristic prediction (default 0.4)
        
    Returns:
        Tuple of (combined_penalty, is_llm_like)
        - combined_penalty: Float in [0, 1] indicating LLM-likeness
        - is_llm_like: Boolean flag (True if combined_penalty > 0.4)
    """
    if not text or len(text.strip()) < 10:
        return 0.0, False
    
    # Always compute heuristic as fallback
    heuristic_penalty, heuristic_flag = heuristic_llm_detection(text)
    
    # Try ML prediction if available and requested
    ml_penalty = 0.5  # Default neutral score
    ml_available = False
    
    if use_ml and SKLEARN_AVAILABLE and _llm_model is not None:
        try:
            features = extract_llm_features(text)
            # Predict probability of being LLM-generated
            ml_prob = _llm_model.predict_proba([features])[0][1]
            ml_penalty = float(ml_prob)
            ml_available = True
            
            logger.debug(f"ML LLM detection: probability={ml_penalty:.4f}")
            
        except Exception as e:
            logger.warning(f"ML LLM detection failed, using heuristic only: {e}")
    
    # Combine predictions
    if ml_available:
        # Weighted combination of ML and heuristic
        combined_penalty = ml_weight * ml_penalty + heuristic_weight * heuristic_penalty
        approach = "hybrid"
    else:
        # Fallback to heuristic only
        combined_penalty = heuristic_penalty
        approach = "heuristic_only"
    
    # Ensure in valid range
    combined_penalty = float(np.clip(combined_penalty, 0.0, 1.0))
    
    # Binary classification threshold
    is_llm_like = combined_penalty > 0.4
    
    ml_str = f"{ml_penalty:.3f}" if ml_available else "N/A"
    logger.debug(f"LLM detection ({approach}): penalty={combined_penalty:.4f}, "
                f"is_llm={is_llm_like}, heuristic={heuristic_penalty:.3f}, "
                f"ml={ml_str}")
    
    return combined_penalty, is_llm_like


def create_training_features(texts: List[str], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training features from a list of texts and labels.
    
    This function is useful for training the ML model.
    
    Args:
        texts: List of text samples
        labels: List of binary labels (0=human, 1=LLM)
        
    Returns:
        Tuple of (feature_matrix, label_array)
    """
    features = []
    valid_labels = []
    
    for text, label in zip(texts, labels):
        try:
            feat = extract_llm_features(text)
            if not np.all(feat == 0):  # Skip if all features are zero
                features.append(feat)
                valid_labels.append(label)
        except Exception as e:
            logger.warning(f"Failed to extract features for text: {e}")
            continue
    
    if not features:
        raise ValueError("No valid features extracted from input texts")
    
    return np.array(features), np.array(valid_labels)


def train_llm_detector(
    texts: List[str], 
    labels: List[int],
    model_path: str = "models/llm_detector.pkl",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Train a new LLM detection model.
    
    Args:
        texts: List of training texts
        labels: List of binary labels (0=human, 1=LLM)
        model_path: Path to save the trained model
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with training metrics
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for training")
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import os
    
    # Extract features
    X, y = create_training_features(texts, labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'n_samples': len(X),
        'n_features': X.shape[1]
    }
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    # Load into global variable
    global _llm_model
    _llm_model = model
    
    logger.info(f"LLM detector trained and saved to {model_path}")
    logger.info(f"Training metrics: {metrics}")
    
    return metrics


# Initialize the system
def initialize_llm_detection(model_path: str = "models/llm_detector.pkl") -> bool:
    """
    Initialize the LLM detection system.
    
    Args:
        model_path: Path to the trained model (optional)
        
    Returns:
        True if initialization successful
    """
    success = True
    
    # Try to load ML model
    if SKLEARN_AVAILABLE:
        try:
            success &= load_llm_model(model_path)
        except:
            logger.info("ML model not found, will use heuristic-only detection")
    else:
        logger.info("scikit-learn not available, using heuristic-only detection")
    
    # Initialize other components
    _ensure_nltk_data()
    
    logger.info("LLM detection system initialized")
    return success


# Auto-initialize on import
try:
    initialize_llm_detection()
except Exception as e:
    logger.warning(f"Auto-initialization failed: {e}")