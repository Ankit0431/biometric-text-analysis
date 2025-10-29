"""
Scoring and similarity computation for biometric text authentication.

This module implements:
- Stylometry similarity (cohort-normalized z-score → sigmoid mapping)
- Semantic similarity (E_s): cosine similarity to centroid with Mahalanobis normalization
- Keystroke timing similarity (typing rhythm patterns)
- LLM-likeness detection (heuristic-based stub)
- Fusion scoring combining all signals with hybrid feature normalization and adaptive weighting
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import math
import logging
from keystroke_features import extract_keystroke_features, compute_keystroke_similarity

# Try to import new hybrid LLM detection, fallback to old if not available
try:
    from llm_detection import detect_llm_likeness as hybrid_detect_llm_likeness
    HYBRID_LLM_AVAILABLE = True
except ImportError:
    hybrid_detect_llm_likeness = None
    HYBRID_LLM_AVAILABLE = False

# Configure logging for scoring pipeline
logger = logging.getLogger(__name__)


# Default scoring weights
DEFAULT_WEIGHTS = {
    'semantic': 0.55,      # Weight for semantic embedding similarity
    'stylometry': 0.15,    # Weight for stylometry features
    'keystroke': 0.20,     # Weight for keystroke timing
    'llm_penalty': 0.10,   # Weight for LLM-likeness penalty
}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in range [-1, 1]
    """
    if len(a.shape) > 1:
        a = a.flatten()
    if len(b.shape) > 1:
        b = b.flatten()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def mahalanobis_similarity(
    vector: np.ndarray,
    centroid: np.ndarray,
    cov_diag: Optional[np.ndarray] = None,
    epsilon: float = 1e-6
) -> float:
    """
    Compute normalized Mahalanobis distance-based similarity.

    Uses diagonal covariance for computational efficiency.
    Returns a similarity score in [0, 1] range.

    Args:
        vector: Query vector
        centroid: Profile centroid
        cov_diag: Diagonal of covariance matrix (variance per dimension)
        epsilon: Small constant to prevent division by zero

    Returns:
        Similarity score (higher is more similar)
    """
    if len(vector.shape) > 1:
        vector = vector.flatten()
    if len(centroid.shape) > 1:
        centroid = centroid.flatten()

    diff = vector - centroid

    if cov_diag is not None:
        # Mahalanobis with diagonal covariance
        if len(cov_diag.shape) > 1:
            cov_diag = cov_diag.flatten()
        # Add epsilon to prevent division by zero
        inv_cov = 1.0 / (cov_diag + epsilon)
        distance_sq = np.sum(diff * diff * inv_cov)
    else:
        # Euclidean distance if no covariance provided
        distance_sq = np.sum(diff * diff)

    distance = np.sqrt(distance_sq)

    # Convert distance to similarity using exponential decay
    # Smaller distance = higher similarity
    similarity = np.exp(-distance / 10.0)

    return float(np.clip(similarity, 0.0, 1.0))


def compute_semantic_similarity(
    embedding: np.ndarray,
    centroid: np.ndarray,
    cov_diag: Optional[np.ndarray] = None,
    use_mahalanobis: bool = True
) -> float:
    """
    Compute semantic similarity (E_s) using cosine similarity and optionally
    Mahalanobis distance.

    Args:
        embedding: Current sample embedding
        centroid: Profile centroid
        cov_diag: Diagonal covariance (optional)
        use_mahalanobis: Whether to use Mahalanobis in addition to cosine

    Returns:
        Semantic similarity score in [0, 1]
    """
    # Cosine similarity (already in [-1, 1], convert to [0, 1])
    cos_sim = cosine_similarity(embedding, centroid)
    cos_sim_normalized = (cos_sim + 1.0) / 2.0  # Map to [0, 1]

    if use_mahalanobis and cov_diag is not None:
        # Combine cosine and Mahalanobis
        mahal_sim = mahalanobis_similarity(embedding, centroid, cov_diag)
        # Weighted combination (favor cosine slightly)
        semantic_score = 0.7 * cos_sim_normalized + 0.3 * mahal_sim
    else:
        semantic_score = cos_sim_normalized

    return float(np.clip(semantic_score, 0.0, 1.0))


def sigmoid(x: float, scale: float = 1.0, offset: float = 0.0) -> float:
    """
    Sigmoid function for smooth mapping with overflow protection.

    Args:
        x: Input value
        scale: Scale parameter
        offset: Offset parameter

    Returns:
        Sigmoid output in [0, 1]
    """
    z = -scale * (x - offset)
    # Clip to prevent overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + math.exp(z))


def compute_stylometry_similarity(
    style_vector: np.ndarray,
    profile_style_mean: np.ndarray,
    profile_style_std: Optional[np.ndarray] = None,
    cohort_mean: Optional[np.ndarray] = None,
    cohort_std: Optional[np.ndarray] = None,
    use_pca: bool = True
) -> float:
    """
    Compute stylometry similarity using enhanced PCA-based pipeline or legacy methods.

    Enhanced Process (use_pca=True):
    1. Check if PCA models are available
    2. If available, use PCA-transformed features for similarity computation
    3. Otherwise, fall back to legacy method

    Legacy Process (use_pca=False):
    1. Compute normalized distance relative to profile mean/std
    2. Use robust scaling to handle variability in writing styles
    3. Map through sigmoid to [0, 1]

    Args:
        style_vector: Current sample's style features
        profile_style_mean: Mean of user's style features
        profile_style_std: Std dev of user's style features
        cohort_mean: Cohort mean for normalization (legacy)
        cohort_std: Cohort std dev for normalization (legacy)
        use_pca: Whether to use PCA-enhanced pipeline

    Returns:
        Stylometry similarity score in [0, 1]
    """
    # Try PCA-enhanced method first if requested
    if use_pca:
        try:
            from stylometry_pca import get_enhanced_pipeline
            pipeline = get_enhanced_pipeline()
            
            if pipeline.is_fitted:
                # Check dimension compatibility before using PCA method
                style_dim = style_vector.shape[0] if len(style_vector.shape) == 1 else style_vector.shape[1]
                profile_dim = profile_style_mean.shape[0] if len(profile_style_mean.shape) == 1 else profile_style_mean.shape[1]
                
                if style_dim == profile_dim:
                    # Dimensions match, use PCA-enhanced similarity computation
                    return pipeline.compute_enhanced_similarity(
                        style_vector,
                        profile_style_mean,
                        profile_style_std
                    )
                else:
                    logger.warning(f"Dimension mismatch: style_vector has {style_dim}D, profile has {profile_dim}D. Using fallback similarity computation.")
                    
                    # When dimensions don't match, we need to use a fallback approach
                    # For now, we'll disable PCA and fall back to legacy method
                    # In the future, profiles should be re-enrolled with PCA features
                    
                    # Fall back to legacy method
            else:
                logger.info("PCA pipeline not fitted, falling back to legacy method")
        except ImportError:
            logger.warning("PCA stylometry module not available, using legacy method")
        except Exception as e:
            logger.warning(f"PCA stylometry failed: {e}, falling back to legacy method")
    
    # Legacy method
    if len(style_vector.shape) > 1:
        style_vector = style_vector.flatten()
    if len(profile_style_mean.shape) > 1:
        profile_style_mean = profile_style_mean.flatten()

    # Check for dimension compatibility in legacy method
    style_dim = len(style_vector) if isinstance(style_vector, np.ndarray) else style_vector.shape[0]
    profile_dim = len(profile_style_mean) if isinstance(profile_style_mean, np.ndarray) else profile_style_mean.shape[0]
    
    if style_dim != profile_dim:
        logger.warning(f"Cannot compute stylometry similarity: style_vector ({style_dim}D) != profile_style_mean ({profile_dim}D)")
        logger.warning("Profile needs to be re-enrolled with PCA-enhanced features or PCA pipeline needs retraining")
        # Return a neutral score to avoid complete failure
        return 0.5

    # Compute absolute difference
    diff = style_vector - profile_style_mean
    
    # Use cosine similarity as primary metric (more robust to scale differences)
    cos_sim = cosine_similarity(style_vector, profile_style_mean)
    cos_sim_normalized = (cos_sim + 1.0) / 2.0  # Map from [-1,1] to [0,1]

    # Compute normalized distance as secondary metric
    if profile_style_std is not None:
        if len(profile_style_std.shape) > 1:
            profile_style_std = profile_style_std.flatten()
        
        # Use adaptive epsilon based on std values to prevent extreme z-scores
        # For features with very low variance, use larger epsilon
        epsilon = np.maximum(0.3, profile_style_std * 0.3)
        z_scores = diff / (profile_style_std + epsilon)
        
        # Use median absolute deviation instead of L2 norm (more robust to outliers)
        mad = np.median(np.abs(z_scores))
        distance = mad * 1.4826  # Scale to approximate std dev
    else:
        # If no std, just use normalized L2 distance
        distance = np.linalg.norm(diff) / (np.linalg.norm(profile_style_mean) + 1e-6)

    # Cohort normalization (if available)
    if cohort_mean is not None and cohort_std is not None:
        cohort_normalized = (distance - cohort_mean.mean()) / (cohort_std.mean() + 1e-6)
    else:
        cohort_normalized = distance

    # Map distance to similarity using very lenient sigmoid
    # The sigmoid should be gentle: distance 0 → ~0.95, distance 2 → ~0.7, distance 5 → ~0.5
    distance_similarity = sigmoid(float(-cohort_normalized), scale=0.15, offset=1.5)

    # Combine cosine similarity (70%) and distance similarity (30%)
    # Cosine is more reliable for style matching as it's scale-invariant
    combined_similarity = 0.7 * cos_sim_normalized + 0.3 * distance_similarity

    return float(np.clip(combined_similarity, 0.0, 1.0))


def detect_llm_likeness(
    text: str,
    stats: Optional[Dict[str, Any]] = None,
    use_hybrid: bool = True
) -> Tuple[float, bool]:
    """
    Detect LLM-likeness using hybrid ML + heuristic approach.

    This function now supports two detection modes:
    1. Hybrid (default): Uses ML classification + heuristic rules when available
    2. Heuristic-only: Falls back to original statistical patterns

    LLM-generated text typically exhibits:
    - Very consistent sentence lengths (low variance)
    - Perfect grammar and punctuation
    - Overly formal or structured writing
    - High vocabulary diversity but formulaic patterns
    - Lack of natural typing errors or hesitations
    - Lower perplexity scores
    - Distinctive POS tag patterns

    Args:
        text: Input text
        stats: Optional pre-computed statistics (legacy parameter)
        use_hybrid: Whether to use hybrid ML+heuristic detection (default True)

    Returns:
        Tuple of (llm_penalty, is_llm_like)
        - llm_penalty: Penalty to apply to score (0 = no penalty, 1 = maximum penalty)
        - is_llm_like: Boolean flag indicating if text appears LLM-generated
    """
    # Try hybrid detection first if available and requested
    if use_hybrid and HYBRID_LLM_AVAILABLE:
        try:
            penalty, is_llm = hybrid_detect_llm_likeness(text)
            logger.debug(f"Hybrid LLM detection: penalty={penalty:.4f}, is_llm={is_llm}")
            return penalty, is_llm
        except Exception as e:
            logger.warning(f"Hybrid LLM detection failed, falling back to heuristic: {e}")
    
    # Fallback to original heuristic-only detection
    return _heuristic_llm_detection(text, stats)


def _heuristic_llm_detection(
    text: str,
    stats: Optional[Dict[str, Any]] = None
) -> Tuple[float, bool]:
    """
    Original heuristic-only LLM detection (kept as fallback).
    
    Args:
        text: Input text
        stats: Optional pre-computed statistics

    Returns:
        Tuple of (llm_penalty, is_llm_like)
    """
    if stats is None:
        stats = {}

    # Split into sentences
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

    if len(sentences) < 2:
        # Too short to determine reliably
        return 0.0, False

    # 1. Sentence length variance (LLMs are too consistent)
    sentence_lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(sentence_lengths)
    std_len = np.std(sentence_lengths)
    coefficient_of_variation = std_len / (mean_len + 1e-6)

    # 2. Punctuation patterns (LLMs use punctuation very consistently)
    punctuation_chars = [c for c in text if c in '.,;:!?-']
    if len(punctuation_chars) > 5:
        punct_counts = Counter(punctuation_chars)
        total = sum(punct_counts.values())
        punct_probs = [count / total for count in punct_counts.values()]
        punct_entropy = -sum(p * math.log2(p) for p in punct_probs if p > 0)
    else:
        punct_entropy = 1.0

    # 3. Word length consistency (LLMs use consistent vocabulary)
    words = text.split()
    word_lengths = [len(w.strip('.,!?;:')) for w in words if w.strip('.,!?;:')]
    if len(word_lengths) > 10:
        word_len_std = np.std(word_lengths)
        word_len_mean = np.mean(word_lengths)
        word_len_cv = word_len_std / (word_len_mean + 1e-6)
    else:
        word_len_cv = 0.5

    # 4. Sentence starter patterns (LLMs often start sentences similarly)
    sentence_starters = [s.split()[0].lower() if s.split() else '' for s in sentences]
    starter_counts = Counter(sentence_starters)
    if len(sentence_starters) > 0:
        max_starter_freq = max(starter_counts.values()) / len(sentence_starters)
    else:
        max_starter_freq = 0.0

    # 5. Paragraph structure (LLMs often use perfect paragraph lengths)
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if len(paragraphs) > 1:
        para_lengths = [len(p.split()) for p in paragraphs]
        para_cv = np.std(para_lengths) / (np.mean(para_lengths) + 1e-6)
    else:
        para_cv = 0.5

    # 6. Check for common LLM phrases and patterns
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

    # Calculate penalty based on multiple factors
    penalty = 0.0
    llm_signals = []

    # === ROBOTIC LLM SIGNALS (TOO CONSISTENT) ===
    
    # Signal 1: Extremely consistent sentence lengths
    if coefficient_of_variation < 0.30 and len(sentences) >= 3:
        penalty += 0.30
        llm_signals.append(f"low_sent_variance={coefficient_of_variation:.3f}")

    # Signal 2: Too perfect punctuation distribution
    if punct_entropy < 1.5 and len(punctuation_chars) > 8:
        penalty += 0.30
        llm_signals.append(f"low_punct_entropy={punct_entropy:.3f}")

    # Signal 3: Very consistent word lengths
    if word_len_cv < 0.50 and len(word_lengths) > 15:
        penalty += 0.20
        llm_signals.append(f"consistent_words={word_len_cv:.3f}")

    # Signal 4: Repetitive sentence starters
    if max_starter_freq > 0.35 and len(sentences) >= 3:
        penalty += 0.20
        llm_signals.append(f"repetitive_starters={max_starter_freq:.3f}")

    # Signal 5: Formal LLM phrases
    if llm_phrase_ratio > 0.2:
        penalty += 0.30
        llm_signals.append(f"llm_phrases={llm_phrase_ratio:.3f}")

    # Signal 6: Perfect paragraph structure
    if para_cv < 0.20 and len(paragraphs) >= 2:
        penalty += 0.15
        llm_signals.append(f"uniform_paragraphs={para_cv:.3f}")
    
    # === NATURAL-SOUNDING LLM SIGNALS (TOO PERFECT) ===
    
    # Signal 7: Suspiciously sophisticated vocabulary
    # Count words >10 letters and complex patterns
    long_words = [w for w in words if len(w.strip('.,!?;:')) > 10]
    long_word_ratio = len(long_words) / max(len(words), 1)
    if long_word_ratio > 0.15:  # >15% very long words is suspicious
        penalty += 0.25
        llm_signals.append(f"sophisticated_vocab={long_word_ratio:.3f}")
    
    # Signal 8: Perfect grammar indicators (no informal contractions, typos)
    contractions = ["dont", "cant", "wont", "didnt", "wouldnt", "shouldnt", "isnt", "arent"]
    text_nospace = text.lower().replace(" ", "").replace("'", "")
    has_contractions = any(c in text_nospace for c in contractions)
    # LLMs rarely use contractions; humans often do
    if not has_contractions and len(words) > 50:
        penalty += 0.15
        llm_signals.append("no_contractions")
    
    # Signal 9: Suspiciously high sentence variety WITHOUT informal markers
    # If variance is HIGH but no casual language → likely LLM trying to look natural
    informal_markers = ["lol", "btw", "tbh", "imo", "haha", "yeah", "nah", "gonna", "wanna"]
    has_informal = any(marker in text.lower() for marker in informal_markers)
    if coefficient_of_variation > 0.55 and not has_informal and len(sentences) >= 5:
        penalty += 0.20
        llm_signals.append(f"unnatural_variance={coefficient_of_variation:.3f}")
    
    # Signal 10: Flawless punctuation usage (no missing spaces, double spaces, etc.)
    punct_errors = text.count('  ') + text.count('.,') + text.count(',.') + text.count('..') + text.count('!!')
    if punct_errors == 0 and len(text) > 200:
        penalty += 0.15
        llm_signals.append("perfect_punctuation")

    penalty = min(penalty, 1.0)

    # LOWER threshold since we have 10 signals now (catches "too perfect" text)
    is_llm_like = penalty > 0.35

    # Log all checks, not just when detected
    print(f"DEBUG LLM DETECTION: penalty={penalty:.3f}, is_llm={is_llm_like}, signals={llm_signals}")
    print(f"  sent_cv={coefficient_of_variation:.3f}, punct_entropy={punct_entropy:.3f}")
    print(f"  word_cv={word_len_cv:.3f}, starter_freq={max_starter_freq:.3f}")
    print(f"  llm_phrases={llm_phrase_count}/{len(sentences)}, para_cv={para_cv:.3f}")
    print(f"  long_words={long_word_ratio:.3f}, informal={has_informal}, punct_errors={punct_errors}")

    return penalty, is_llm_like


def compute_final_score(
    semantic_score: float, 
    stylometry_score: float, 
    keystroke_score: Optional[float] = None, 
    llm_penalty: float = 0.0
) -> float:
    """
    Compute final score with hybrid feature fusion and proper normalization.
    
    This implements the core scoring algorithm with:
    - Individual score normalization and clipping
    - Dynamic weighting based on availability and confidence
    - Z-score normalization for consistent scaling
    - Adaptive weight adjustment based on component availability
    
    Args:
        semantic_score: Semantic similarity score [0, 1]
        stylometry_score: Stylometry similarity score [0, 1]
        keystroke_score: Keystroke timing similarity score [0, 1] (optional)
        llm_penalty: LLM-likeness penalty [0, 1]
        
    Returns:
        Final normalized score [0, 1]
    """
    # Use provided keystroke score or default to neutral
    keystroke_value = keystroke_score if keystroke_score is not None else 0.0
    
    # Clip individual scores
    scores = np.array([semantic_score, stylometry_score, keystroke_value])
    scores = np.clip(scores, 0, 1)
    
    # Check if we need normalization (only if scores vary significantly)
    scores_std = scores.std()
    if scores_std > 0.1:  # Only normalize if there's meaningful variance
        # Z-score normalization: (x - mean) / std
        scores_mean = scores.mean()
        scores = (scores - scores_mean) / scores_std
        
        # Min-max normalization to [0, 1]
        scores_min = scores.min()
        scores_max = scores.max()
        if scores_max - scores_min > 1e-6:
            scores = (scores - scores_min) / (scores_max - scores_min)
        else:
            scores = np.full_like(scores, 0.5)
    else:
        # Scores are already similar, use them directly (no normalization needed)
        # This preserves high scores when all components agree
        print(f"SCORING: Scores similar (std={scores_std:.3f}), skipping normalization")
    
    # Default weights for hybrid fusion
    weights = {
        'semantic': 0.45,
        'stylometry': 0.35, 
        'keystroke': 0.15,
        'llm_penalty': 0.05
    }
    
    # Adaptive weight tuning based on availability and confidence
    if keystroke_score is None:
        # No keystroke data available - redistribute weight
        weights['keystroke'] = 0.05  # Minimal weight for default neutral score
        weights['semantic'] += 0.05   # Boost semantic component
        print(f"SCORING: Keystroke unavailable, redistributing weights: semantic={weights['semantic']:.2f}")
    
    if llm_penalty > 0.4:
        # High LLM penalty - increase its impact
        weights['llm_penalty'] = 0.15
        # Reduce other weights proportionally
        remaining_weight = 1.0 - weights['llm_penalty']
        scale_factor = remaining_weight / (weights['semantic'] + weights['stylometry'] + weights['keystroke'])
        weights['semantic'] *= scale_factor
        weights['stylometry'] *= scale_factor  
        weights['keystroke'] *= scale_factor
        print(f"SCORING: High LLM penalty ({llm_penalty:.3f}), increasing penalty weight to {weights['llm_penalty']:.2f}")
    
    # Compute weighted fusion score
    weighted_score = (
        weights['semantic'] * scores[0] +
        weights['stylometry'] * scores[1] + 
        weights['keystroke'] * scores[2]
    ) - weights['llm_penalty'] * llm_penalty
    
    # Final clipping to [0, 1]
    final_score = float(np.clip(weighted_score, 0, 1))
    
    # Detailed logging for debugging
    print(f"SCORING FUSION: final={final_score:.4f}")
    print(f"  Raw scores: semantic={semantic_score:.4f}, stylometry={stylometry_score:.4f}, keystroke={keystroke_score or 0.0:.4f}")
    print(f"  Normalized: semantic={scores[0]:.4f}, stylometry={scores[1]:.4f}, keystroke={scores[2]:.4f}")
    print(f"  Weights: semantic={weights['semantic']:.3f}, stylometry={weights['stylometry']:.3f}, keystroke={weights['keystroke']:.3f}, llm_penalty={weights['llm_penalty']:.3f}")
    print(f"  LLM penalty: {llm_penalty:.4f}")
    print(f"  Component contributions: semantic={weights['semantic'] * scores[0]:.4f}, stylometry={weights['stylometry'] * scores[1]:.4f}, keystroke={weights['keystroke'] * scores[2]:.4f}, penalty=-{weights['llm_penalty'] * llm_penalty:.4f}")
    
    return final_score


def score_sample(
    user_profile: Dict[str, Any],
    text: str,
    embedding: np.ndarray,
    style_features: np.ndarray,
    timings: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Score a text sample against a user profile using refactored hybrid fusion pipeline.

    This implements the fusion scoring function combining:
    - Semantic embedding similarity (E_s)
    - Stylometry feature similarity
    - Keystroke timing similarity
    - LLM-likeness penalty

    Args:
        user_profile: User's biometric profile containing:
            - centroid: Semantic embedding centroid
            - cov_diag: Covariance diagonal (optional)
            - style_mean: Mean stylometry features
            - style_std: Std dev of stylometry features
            - keystroke_mean: Mean keystroke features (optional)
            - keystroke_std: Std keystroke features (optional)
            - cohort_style_mean: Cohort mean (optional)
            - cohort_style_std: Cohort std (optional)
        text: Input text
        embedding: Semantic embedding of the text
        style_features: Stylometry feature vector
        timings: Keystroke timing data (histogram, mean_iki, std_iki, total_events)
        context: Additional context (optional)
        weights: Scoring weights (optional, uses defaults)

    Returns:
        Dictionary containing:
            - final_score: Overall similarity score [0, 1]
            - semantic_score: Semantic similarity component
            - stylometry_score: Stylometry similarity component
            - keystroke_score: Keystroke timing similarity component
            - llm_penalty: LLM-likeness penalty
            - llm_like: Boolean flag
            - components: Detailed breakdown
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Extract profile data
    centroid = user_profile.get('centroid')
    cov_diag = user_profile.get('cov_diag')
    style_mean = user_profile.get('style_mean')
    style_std = user_profile.get('style_std')
    keystroke_mean = user_profile.get('keystroke_mean')
    keystroke_std = user_profile.get('keystroke_std')
    cohort_style_mean = user_profile.get('cohort_style_mean')
    cohort_style_std = user_profile.get('cohort_style_std')

    if centroid is None or style_mean is None:
        raise ValueError("Profile must contain at least 'centroid' and 'style_mean'")

    # Convert to numpy arrays if needed
    if not isinstance(centroid, np.ndarray):
        centroid = np.array(centroid, dtype=np.float32)
    if not isinstance(style_mean, np.ndarray):
        style_mean = np.array(style_mean, dtype=np.float32)
    if cov_diag is not None and not isinstance(cov_diag, np.ndarray):
        cov_diag = np.array(cov_diag, dtype=np.float32)
    if style_std is not None and not isinstance(style_std, np.ndarray):
        style_std = np.array(style_std, dtype=np.float32)
    if keystroke_mean is not None and not isinstance(keystroke_mean, np.ndarray):
        keystroke_mean = np.array(keystroke_mean, dtype=np.float32)

    # 1. Compute semantic similarity
    semantic_score = compute_semantic_similarity(
        embedding, centroid, cov_diag, use_mahalanobis=True
    )

    # 2. Compute stylometry similarity
    stylometry_score = compute_stylometry_similarity(
        style_features,
        style_mean,
        style_std,
        cohort_style_mean,
        cohort_style_std
    )
    
    # Debug logging for stylometry
    if stylometry_score < 0.5:
        print(f"DEBUG STYLOMETRY: score={stylometry_score:.4f}")
        print(f"  style_features shape: {style_features.shape}, mean={style_features.mean():.2f}, std={style_features.std():.2f}")
        print(f"  profile style_mean: mean={style_mean.mean():.2f}, std={style_mean.std():.2f}")
        if style_std is not None:
            print(f"  profile style_std: mean={style_std.mean():.4f}, min={style_std.min():.4f}, max={style_std.max():.4f}")
        # Calculate cosine similarity for debugging
        cos_sim = cosine_similarity(style_features, style_mean)
        print(f"  cosine similarity: {cos_sim:.4f}")

    # 3. Compute keystroke timing similarity
    keystroke_score = None  # Default to None to trigger adaptive weighting
    if timings and keystroke_mean is not None:
        # Check if there's sufficient keystroke data
        total_events = timings.get('total_events', 0)
        text_length = len(text)
        
        # Calculate keystroke ratio
        # Normal typing: ~1 event per character (keydown only, since we count those)
        # Mixed (typing + some paste): 0.5-0.9 events per character
        # Mostly copy-pasted: <0.3 events per character
        keystroke_ratio = total_events / text_length if text_length > 0 else 0
        
        if keystroke_ratio < 0.15:  # Less than 15% - almost entirely copy-pasted
            # Severely penalize - this is clearly not typed
            keystroke_score = 0.05
            print(f"DEBUG KEYSTROKE: SEVERE COPY-PASTE - events={total_events}, text_len={text_length}, ratio={keystroke_ratio:.3f}")
        elif keystroke_ratio < 0.40:  # 15-40% - significant copy-pasting
            # Moderate penalty - mix of typing and pasting
            keystroke_score = 0.3
            print(f"DEBUG KEYSTROKE: MIXED INPUT DETECTED - events={total_events}, text_len={text_length}, ratio={keystroke_ratio:.3f}")
        else:
            # Sufficient keystrokes - compute normal similarity
            # Extract keystroke features from timing data
            current_keystroke_features = extract_keystroke_features(timings)
            # Compare with enrolled keystroke features
            keystroke_score = compute_keystroke_similarity(
                current_keystroke_features,
                keystroke_mean
            )
            print(f"DEBUG KEYSTROKE: TYPED INPUT - events={total_events}, text_len={text_length}, ratio={keystroke_ratio:.3f}, score={keystroke_score:.4f}")

    # 4. Detect LLM-likeness
    llm_penalty, llm_like = detect_llm_likeness(text)

    # 5. NEW: Use refactored fusion scoring with proper normalization
    final_score = compute_final_score(
        semantic_score=semantic_score,
        stylometry_score=stylometry_score, 
        keystroke_score=keystroke_score,
        llm_penalty=llm_penalty
    )

    return {
        'final_score': final_score,
        'semantic_score': float(semantic_score),
        'stylometry_score': float(stylometry_score),
        'keystroke_score': float(keystroke_score) if keystroke_score is not None else None,
        'llm_penalty': float(llm_penalty),
        'llm_like': llm_like,
        'components': {
            'semantic': float(semantic_score),
            'stylometry': float(stylometry_score),
            'keystroke': float(keystroke_score) if keystroke_score is not None else None,
            'llm_penalty': float(llm_penalty),
        },
        'weights': weights  # Note: These are the original weights, actual fusion uses adaptive weights
    }
