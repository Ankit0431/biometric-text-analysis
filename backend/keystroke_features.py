"""
Keystroke timing feature extraction.

This module extracts features from keystroke timing data:
- Inter-key interval (IKI) statistics
- Typing rhythm patterns
- Histogram distribution features
"""
from typing import Dict, Any, Optional
import numpy as np


def extract_keystroke_features(timings: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    Extract keystroke timing features from timing data.
    
    Args:
        timings: Dictionary containing:
            - histogram: List of 6 bins (0-50ms, 50-100ms, etc.)
            - mean_iki: Mean inter-key interval in ms
            - std_iki: Standard deviation of IKI in ms
            - total_events: Total number of key events
    
    Returns:
        Feature vector of shape (10,) containing:
            [0-5]: Normalized histogram (6 bins)
            [6]: Normalized mean IKI (mean/1000)
            [7]: Normalized std IKI (std/1000)
            [8]: Typing speed (events per second)
            [9]: Consistency score (1 - cv)
    """
    # Default zero features if no timing data
    if timings is None or not timings:
        return np.zeros(10, dtype=np.float32)
    
    features = np.zeros(10, dtype=np.float32)
    
    # Extract histogram bins (6 bins)
    histogram = timings.get('histogram', [0, 0, 0, 0, 0, 0])
    if isinstance(histogram, list) and len(histogram) == 6:
        total = sum(histogram)
        if total > 0:
            # Normalize histogram to probabilities
            features[0:6] = np.array(histogram, dtype=np.float32) / total
    
    # Extract statistics
    mean_iki = timings.get('mean_iki', 0)
    std_iki = timings.get('std_iki', 0)
    total_events = timings.get('total_events', 0)
    
    # Normalize mean and std (convert ms to seconds)
    if mean_iki > 0:
        features[6] = min(mean_iki / 1000.0, 5.0)  # Cap at 5 seconds
    
    if std_iki > 0:
        features[7] = min(std_iki / 1000.0, 5.0)  # Cap at 5 seconds
    
    # Typing speed (events per second estimate)
    if total_events > 0 and mean_iki > 0:
        # events per second = 1000 / mean_iki
        typing_speed = min(1000.0 / mean_iki, 20.0)  # Cap at 20 keys/sec
        features[8] = typing_speed / 20.0  # Normalize to [0, 1]
    
    # Consistency score (inverse of coefficient of variation)
    if mean_iki > 0 and std_iki > 0:
        cv = std_iki / mean_iki  # Coefficient of variation
        consistency = 1.0 / (1.0 + cv)  # Higher = more consistent
        features[9] = consistency
    elif mean_iki > 0:
        features[9] = 1.0  # Perfect consistency if std is 0
    
    return features


def compute_keystroke_similarity(
    features1: np.ndarray,
    features2: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute similarity between two keystroke feature vectors.
    
    Uses cosine similarity with optional feature weighting.
    
    Args:
        features1: First feature vector
        features2: Second feature vector
        weights: Optional weight vector for features
    
    Returns:
        Similarity score in [0, 1]
    """
    if features1 is None or features2 is None:
        return 0.5  # Neutral score if missing data
    
    # Apply weights if provided
    if weights is not None:
        features1 = features1 * weights
        features2 = features2 * weights
    
    # Compute cosine similarity
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.5  # Neutral score
    
    cosine_sim = np.dot(features1, features2) / (norm1 * norm2)
    
    # Map from [-1, 1] to [0, 1]
    similarity = (cosine_sim + 1.0) / 2.0
    
    return float(np.clip(similarity, 0.0, 1.0))


def aggregate_keystroke_features(features_list: list[np.ndarray]) -> Dict[str, Any]:
    """
    Aggregate keystroke features from multiple samples.
    
    Args:
        features_list: List of keystroke feature vectors
    
    Returns:
        Dictionary with mean and std of features
    """
    if not features_list:
        return {
            'mean': np.zeros(10, dtype=np.float32),
            'std': np.zeros(10, dtype=np.float32)
        }
    
    features_array = np.array(features_list, dtype=np.float32)
    
    return {
        'mean': np.mean(features_array, axis=0),
        'std': np.std(features_array, axis=0)
    }
