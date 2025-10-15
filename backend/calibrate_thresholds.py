"""
Threshold calibration script for biometric text authentication.

Calibrates authentication thresholds based on cohort data to achieve
target False Accept Rate (FAR) levels.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional


def compute_similarity_scores(
    user_vectors: List[np.ndarray],
    cohort_vectors: List[np.ndarray]
) -> Tuple[List[float], List[float]]:
    """
    Compute genuine and impostor similarity scores.

    Args:
        user_vectors: List of vectors from the enrolling user
        cohort_vectors: List of vectors from other users (cohort)

    Returns:
        Tuple of (genuine_scores, impostor_scores)
    """
    genuine_scores = []
    impostor_scores = []

    if len(user_vectors) < 2:
        # Need at least 2 vectors for genuine scores
        return genuine_scores, impostor_scores

    # Compute user centroid
    user_centroid = np.mean(user_vectors, axis=0)
    user_centroid = user_centroid / (np.linalg.norm(user_centroid) + 1e-8)

    # Genuine scores: user vectors vs user centroid
    for vec in user_vectors:
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        score = float(np.dot(user_centroid, vec_norm))
        # Map from [-1, 1] to [0, 1]
        score = (score + 1.0) / 2.0
        genuine_scores.append(score)

    # Impostor scores: cohort vectors vs user centroid
    for vec in cohort_vectors:
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        score = float(np.dot(user_centroid, vec_norm))
        # Map from [-1, 1] to [0, 1]
        score = (score + 1.0) / 2.0
        impostor_scores.append(score)

    return genuine_scores, impostor_scores


def calibrate_thresholds(
    user_vectors: List[np.ndarray],
    cohort_vectors: List[np.ndarray],
    target_far_high: float = 0.005,  # 0.5% FAR for high threshold
    target_far_med: float = 0.02,     # 2% FAR for medium threshold
    default_high: float = 0.84,
    default_med: float = 0.72
) -> Dict[str, float]:
    """
    Calibrate thresholds based on cohort similarity scores.

    Process:
    1. Compute genuine scores (user samples vs user centroid)
    2. Compute impostor scores (cohort samples vs user centroid)
    3. Find thresholds that achieve target FAR

    Args:
        user_vectors: Enrollment vectors from the user
        cohort_vectors: Vectors from cohort users
        target_far_high: Target False Accept Rate for high threshold
        target_far_med: Target False Accept Rate for medium threshold
        default_high: Default high threshold if calibration not possible
        default_med: Default medium threshold if calibration not possible

    Returns:
        Dictionary with 'high' and 'med' threshold values
    """
    genuine_scores, impostor_scores = compute_similarity_scores(
        user_vectors, cohort_vectors
    )

    # If not enough data for calibration, use defaults
    if len(impostor_scores) < 10:
        return {
            "high": default_high,
            "med": default_med
        }

    # Sort impostor scores in descending order
    impostor_sorted = sorted(impostor_scores, reverse=True)

    # Find threshold at target FAR
    def find_threshold_at_far(target_far: float, default: float) -> float:
        """Find threshold that achieves target FAR."""
        n = len(impostor_sorted)
        # Index for target FAR
        idx = int(target_far * n)

        if idx >= n:
            # Not enough samples, use default
            return default

        # Threshold is the score at the FAR percentile
        threshold = impostor_sorted[idx]

        # Ensure threshold is in reasonable range [0.5, 0.95]
        threshold = max(0.50, min(0.95, threshold))

        return float(threshold)

    threshold_high = find_threshold_at_far(target_far_high, default_high)
    threshold_med = find_threshold_at_far(target_far_med, default_med)

    # Ensure proper ordering: high > med
    if threshold_med >= threshold_high:
        threshold_med = threshold_high - 0.05
        threshold_med = max(0.50, threshold_med)

    return {
        "high": threshold_high,
        "med": threshold_med
    }


def estimate_far_frr(
    genuine_scores: List[float],
    impostor_scores: List[float],
    threshold: float
) -> Tuple[float, float]:
    """
    Estimate FAR and FRR at a given threshold.

    Args:
        genuine_scores: Genuine user scores
        impostor_scores: Impostor scores
        threshold: Decision threshold

    Returns:
        Tuple of (FAR, FRR)
    """
    # FAR: False Accept Rate (impostors accepted)
    if len(impostor_scores) > 0:
        far = sum(1 for s in impostor_scores if s >= threshold) / len(impostor_scores)
    else:
        far = 0.0

    # FRR: False Reject Rate (genuine users rejected)
    if len(genuine_scores) > 0:
        frr = sum(1 for s in genuine_scores if s < threshold) / len(genuine_scores)
    else:
        frr = 0.0

    return far, frr


def calibrate_with_stats(
    user_vectors: List[np.ndarray],
    cohort_vectors: List[np.ndarray],
    target_far_high: float = 0.005,
    target_far_med: float = 0.02,
    default_high: float = 0.84,
    default_med: float = 0.72
) -> Dict[str, any]:
    """
    Calibrate thresholds and return detailed statistics.

    Args:
        user_vectors: Enrollment vectors from the user
        cohort_vectors: Vectors from cohort users
        target_far_high: Target FAR for high threshold
        target_far_med: Target FAR for medium threshold
        default_high: Default high threshold
        default_med: Default medium threshold

    Returns:
        Dictionary with thresholds and statistics
    """
    thresholds = calibrate_thresholds(
        user_vectors, cohort_vectors,
        target_far_high, target_far_med,
        default_high, default_med
    )

    genuine_scores, impostor_scores = compute_similarity_scores(
        user_vectors, cohort_vectors
    )

    # Compute FAR/FRR at calibrated thresholds
    far_high, frr_high = estimate_far_frr(
        genuine_scores, impostor_scores, thresholds["high"]
    )
    far_med, frr_med = estimate_far_frr(
        genuine_scores, impostor_scores, thresholds["med"]
    )

    return {
        "thresholds": thresholds,
        "stats": {
            "n_genuine_samples": len(genuine_scores),
            "n_impostor_samples": len(impostor_scores),
            "high_threshold": {
                "value": thresholds["high"],
                "far": far_high,
                "frr": frr_high
            },
            "med_threshold": {
                "value": thresholds["med"],
                "far": far_med,
                "frr": frr_med
            },
            "genuine_score_stats": {
                "mean": float(np.mean(genuine_scores)) if genuine_scores else 0.0,
                "std": float(np.std(genuine_scores)) if genuine_scores else 0.0,
                "min": float(np.min(genuine_scores)) if genuine_scores else 0.0,
                "max": float(np.max(genuine_scores)) if genuine_scores else 0.0,
            },
            "impostor_score_stats": {
                "mean": float(np.mean(impostor_scores)) if impostor_scores else 0.0,
                "std": float(np.std(impostor_scores)) if impostor_scores else 0.0,
                "min": float(np.min(impostor_scores)) if impostor_scores else 0.0,
                "max": float(np.max(impostor_scores)) if impostor_scores else 0.0,
            }
        }
    }
