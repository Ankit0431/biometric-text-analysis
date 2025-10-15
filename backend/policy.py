"""
Policy decision logic for biometric text authentication.

This module implements the decision-making logic that determines whether to:
- ALLOW: Accept the authentication
- CHALLENGE: Request additional verification
- STEP_UP: Require stronger authentication
- DENY: Reject the authentication
"""
from typing import Dict, List, Any, Optional
from enum import Enum


class Decision(str, Enum):
    """Authentication decision types."""
    ALLOW = "allow"
    CHALLENGE = "challenge"
    STEP_UP = "step_up"
    DENY = "deny"


class PolicyVersion(str, Enum):
    """Policy version for tracking."""
    V1 = "v1.0"


# Minimum word counts for different flows
MIN_WORDS_VERIFY = 50
MIN_WORDS_ENROLL = 70


def decide(
    score: float,
    threshold_high: float,
    threshold_med: float,
    text: str,
    word_count: int,
    llm_like: bool = False,
    flow: str = "verify",
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Make policy decision based on score, thresholds, and other factors.

    Decision logic:
    - score >= threshold_high AND sufficient length AND not LLM → ALLOW
    - threshold_med <= score < threshold_high → CHALLENGE
    - score < threshold_med → STEP_UP
    - word_count < minimum for flow → CHALLENGE (too short)
    - LLM-like detected → CHALLENGE or STEP_UP depending on score

    Args:
        score: Final similarity score [0, 1]
        threshold_high: High confidence threshold (e.g., 0.84)
        threshold_med: Medium confidence threshold (e.g., 0.72)
        text: Input text
        word_count: Number of words in text
        llm_like: Whether text appears LLM-generated
        flow: Flow type ('verify' or 'enroll')
        context: Additional context

    Returns:
        Dictionary containing:
            - decision: Decision type (allow/challenge/step_up/deny)
            - reasons: List of reason codes
            - score: The input score
            - thresholds: The thresholds used
            - policy_version: Policy version
    """
    reasons = []
    decision = Decision.ALLOW

    # Determine minimum word count based on flow
    if flow == "enroll":
        min_words = MIN_WORDS_ENROLL
    else:
        min_words = MIN_WORDS_VERIFY

    # Check word count (SHORT_LEN)
    if word_count < min_words:
        reasons.append("SHORT_LEN")
        decision = Decision.CHALLENGE
        return {
            "decision": decision,
            "reasons": reasons,
            "score": score,
            "thresholds": {
                "high": threshold_high,
                "med": threshold_med
            },
            "policy_version": PolicyVersion.V1
        }

    # Check for LLM-like text
    if llm_like:
        reasons.append("LLM_LIKE")
        if score >= threshold_med:
            decision = Decision.CHALLENGE
        else:
            decision = Decision.STEP_UP
        return {
            "decision": decision,
            "reasons": reasons,
            "score": score,
            "thresholds": {
                "high": threshold_high,
                "med": threshold_med
            },
            "policy_version": PolicyVersion.V1
        }

    # Main scoring logic
    if score >= threshold_high:
        # High confidence - allow
        decision = Decision.ALLOW
        reasons.append("HIGH_CONFIDENCE")

    elif score >= threshold_med:
        # Medium confidence - challenge
        decision = Decision.CHALLENGE
        reasons.append("MED_CONFIDENCE")

    else:
        # Low confidence - step up
        decision = Decision.STEP_UP
        reasons.append("LOW_CONFIDENCE")

    return {
        "decision": decision,
        "reasons": reasons,
        "score": score,
        "thresholds": {
            "high": threshold_high,
            "med": threshold_med
        },
        "policy_version": PolicyVersion.V1
    }


def should_log_decision(decision: str) -> bool:
    """
    Determine if a decision should be logged to the database.

    Args:
        decision: Decision type

    Returns:
        True if should be logged
    """
    # Log all decisions for audit trail
    return True


def get_default_thresholds() -> Dict[str, float]:
    """
    Get default thresholds.

    Returns:
        Dictionary with 'high' and 'med' thresholds
    """
    return {
        "high": 0.84,
        "med": 0.72
    }


def validate_thresholds(threshold_high: float, threshold_med: float) -> bool:
    """
    Validate that thresholds are in valid ranges and properly ordered.

    Args:
        threshold_high: High threshold
        threshold_med: Medium threshold

    Returns:
        True if valid, False otherwise
    """
    if not (0.0 <= threshold_med <= 1.0):
        return False
    if not (0.0 <= threshold_high <= 1.0):
        return False
    if threshold_med >= threshold_high:
        return False
    return True


def adjust_thresholds_for_risk(
    base_thresholds: Dict[str, float],
    risk_level: str = "medium"
) -> Dict[str, float]:
    """
    Adjust thresholds based on risk level.

    Args:
        base_thresholds: Base thresholds
        risk_level: Risk level ('low', 'medium', 'high')

    Returns:
        Adjusted thresholds
    """
    if risk_level == "high":
        # Stricter thresholds for high-risk scenarios
        return {
            "high": min(base_thresholds["high"] + 0.05, 0.95),
            "med": min(base_thresholds["med"] + 0.05, 0.85)
        }
    elif risk_level == "low":
        # More lenient for low-risk
        return {
            "high": max(base_thresholds["high"] - 0.05, 0.70),
            "med": max(base_thresholds["med"] - 0.05, 0.60)
        }
    else:
        # Medium risk - use base
        return base_thresholds
