"""
Scoring and similarity computation for biometric text authentication.

This module implements:
- Stylometry similarity (cohort-normalized z-score → sigmoid mapping)
- Semantic similarity (E_s): cosine similarity to centroid with Mahalanobis normalization
- LLM-likeness detection (heuristic-based stub)
- Fusion scoring combining all signals
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import math


# Default scoring weights
DEFAULT_WEIGHTS = {
    'semantic': 0.6,      # Weight for semantic embedding similarity
    'stylometry': 0.3,    # Weight for stylometry features
    'llm_penalty': 0.1,   # Weight for LLM-likeness penalty
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
    cohort_std: Optional[np.ndarray] = None
) -> float:
    """
    Compute stylometry similarity using cohort-normalized z-scores.

    Process:
    1. Compute z-score relative to profile mean/std
    2. Normalize using cohort statistics (if available)
    3. Map through sigmoid to [0, 1]

    Args:
        style_vector: Current sample's style features
        profile_style_mean: Mean of user's style features
        profile_style_std: Std dev of user's style features
        cohort_mean: Cohort mean for normalization
        cohort_std: Cohort std dev for normalization

    Returns:
        Stylometry similarity score in [0, 1]
    """
    if len(style_vector.shape) > 1:
        style_vector = style_vector.flatten()
    if len(profile_style_mean.shape) > 1:
        profile_style_mean = profile_style_mean.flatten()

    # Compute distance
    diff = style_vector - profile_style_mean

    if profile_style_std is not None:
        if len(profile_style_std.shape) > 1:
            profile_style_std = profile_style_std.flatten()
        # Normalize by profile std
        epsilon = 1e-6
        z_scores = diff / (profile_style_std + epsilon)
    else:
        z_scores = diff

    # Compute overall distance (L2 norm of z-scores)
    distance = np.linalg.norm(z_scores)

    # Cohort normalization (if available)
    if cohort_mean is not None and cohort_std is not None:
        # Normalize the distance using cohort statistics
        cohort_normalized = (distance - cohort_mean.mean()) / (cohort_std.mean() + 1e-6)
    else:
        cohort_normalized = distance

    # Map to similarity using sigmoid
    # Lower distance = higher similarity
    # Adjust scale: distance of 0 should give ~0.95, distance of 2 should give ~0.5
    # Use a gentler scale to avoid extremes
    similarity = sigmoid(-cohort_normalized, scale=0.5, offset=0.0)

    return float(np.clip(similarity, 0.0, 1.0))


def detect_llm_likeness(
    text: str,
    stats: Optional[Dict[str, Any]] = None
) -> Tuple[float, bool]:
    """
    Detect LLM-likeness using heuristics.

    This is a stub implementation using simple heuristics:
    - Sentence length variance (LLMs tend to have consistent lengths)
    - Punctuation entropy (LLMs use punctuation very consistently)
    - Other statistical regularities

    Args:
        text: Input text
        stats: Optional pre-computed statistics

    Returns:
        Tuple of (llm_penalty, is_llm_like)
        - llm_penalty: Penalty to apply to score (0 = no penalty, 1 = maximum penalty)
        - is_llm_like: Boolean flag indicating if text appears LLM-generated
    """
    if stats is None:
        stats = {}

    # Split into sentences (simple)
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

    if len(sentences) < 2:
        # Too short to determine, assume human
        return 0.0, False

    # Compute sentence length variance
    sentence_lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(sentence_lengths)
    std_len = np.std(sentence_lengths)

    # LLMs tend to have lower variance (more consistent)
    coefficient_of_variation = std_len / (mean_len + 1e-6)

    # Compute punctuation entropy
    punctuation_chars = [c for c in text if c in '.,;:!?-']
    if len(punctuation_chars) > 5:
        punct_counts = Counter(punctuation_chars)
        total = sum(punct_counts.values())
        punct_probs = [count / total for count in punct_counts.values()]
        punct_entropy = -sum(p * math.log2(p) for p in punct_probs if p > 0)
    else:
        punct_entropy = 0.0

    # Heuristic thresholds (tunable)
    llm_like = False
    penalty = 0.0

    # Low variance in sentence length suggests LLM
    if coefficient_of_variation < 0.3:
        penalty += 0.3

    # Low punctuation entropy suggests LLM
    if punct_entropy < 1.5 and len(punctuation_chars) > 10:
        penalty += 0.2

    # Very consistent formatting
    if 'mean_len' in stats and mean_len > 15 and coefficient_of_variation < 0.25:
        penalty += 0.3

    penalty = min(penalty, 1.0)

    if penalty > 0.5:
        llm_like = True

    return penalty, llm_like


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
    Score a text sample against a user profile.

    This implements the fusion scoring function combining:
    - Semantic embedding similarity (E_s)
    - Stylometry feature similarity
    - LLM-likeness penalty

    Args:
        user_profile: User's biometric profile containing:
            - centroid: Semantic embedding centroid
            - cov_diag: Covariance diagonal (optional)
            - style_mean: Mean stylometry features
            - style_std: Std dev of stylometry features
            - cohort_style_mean: Cohort mean (optional)
            - cohort_style_std: Cohort std (optional)
        text: Input text
        embedding: Semantic embedding of the text
        style_features: Stylometry feature vector
        timings: Keystroke timing data (optional, not used in v1)
        context: Additional context (optional)
        weights: Scoring weights (optional, uses defaults)

    Returns:
        Dictionary containing:
            - final_score: Overall similarity score [0, 1]
            - semantic_score: Semantic similarity component
            - stylometry_score: Stylometry similarity component
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

    # 3. Detect LLM-likeness
    llm_penalty, llm_like = detect_llm_likeness(text)

    # 4. Fusion: weighted combination
    base_score = (
        weights['semantic'] * semantic_score +
        weights['stylometry'] * stylometry_score
    )

    # Apply LLM penalty
    final_score = base_score * (1.0 - weights['llm_penalty'] * llm_penalty)

    # Ensure in [0, 1]
    final_score = float(np.clip(final_score, 0.0, 1.0))

    return {
        'final_score': final_score,
        'semantic_score': float(semantic_score),
        'stylometry_score': float(stylometry_score),
        'llm_penalty': float(llm_penalty),
        'llm_like': llm_like,
        'components': {
            'semantic': float(semantic_score),
            'stylometry': float(stylometry_score),
            'llm_penalty': float(llm_penalty),
        },
        'weights': weights
    }
