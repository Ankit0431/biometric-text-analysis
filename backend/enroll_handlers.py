"""
Enrollment flow handlers for biometric text authentication.

Handles /enroll/start and /enroll/submit endpoints.
"""
import uuid
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from schemas import (
    EnrollStartRequest, EnrollStartResponse,
    EnrollSubmitRequest, EnrollSubmitResponse,
    ChallengeInfo
)
from challenge_bank import select_challenges, validate_challenge_response
from normalizer import normalize
from features import extract_features
from encoder import encode
from calibrate_thresholds import calibrate_thresholds
from policy import get_default_thresholds


# Configuration
DEFAULT_REQUIRED_SAMPLES = 8
RAW_TEXT_TTL_DAYS = 0  # Don't store raw text by default
SESSION_TTL_MINUTES = 60


# In-memory session storage (in production, use Redis)
_enrollment_sessions: Dict[str, Dict[str, Any]] = {}


class EnrollmentSession:
    """Manages an enrollment session for a user."""

    def __init__(
        self,
        user_id: str,
        lang: str = "en",
        domain: str = "chat",
        required_samples: int = DEFAULT_REQUIRED_SAMPLES
    ):
        self.session_token = str(uuid.uuid4())
        self.user_id = user_id
        self.lang = lang
        self.domain = domain
        self.required_samples = required_samples
        self.challenges = select_challenges(num_challenges=required_samples)
        self.submissions: List[Dict[str, Any]] = []
        self.created_at = datetime.utcnow()
        self.expires_at = self.created_at + timedelta(minutes=SESSION_TTL_MINUTES)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_token": self.session_token,
            "user_id": self.user_id,
            "lang": self.lang,
            "domain": self.domain,
            "required_samples": self.required_samples,
            "challenges": [c.dict() for c in self.challenges],
            "submissions": self.submissions,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat()
        }

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.utcnow() > self.expires_at

    def get_remaining(self) -> int:
        """Get number of remaining samples needed."""
        return self.required_samples - len(self.submissions)

    def is_complete(self) -> bool:
        """Check if all required samples have been submitted."""
        return len(self.submissions) >= self.required_samples


def enroll_start(request: EnrollStartRequest) -> EnrollStartResponse:
    """
    Start enrollment process for a user.

    Creates a new enrollment session with randomized challenges.

    Args:
        request: EnrollStartRequest containing user_id, lang, domain

    Returns:
        EnrollStartResponse with challenges and session token
    """
    # Create new enrollment session
    session = EnrollmentSession(
        user_id=request.user_id,
        lang=request.lang,
        domain=request.domain,
        required_samples=DEFAULT_REQUIRED_SAMPLES
    )

    # Store session (in production, use Redis)
    _enrollment_sessions[session.session_token] = session

    # Return response
    return EnrollStartResponse(
        challenges=session.challenges,
        session_token=session.session_token,
        required_samples=session.required_samples
    )


def enroll_submit(request: EnrollSubmitRequest) -> EnrollSubmitResponse:
    """
    Submit an enrollment sample.

    Process:
    1. Validate session and challenge
    2. Normalize text
    3. Extract features and encoding
    4. Store vectors (not raw text if TTL=0)
    5. If all samples collected, compute profile and calibrate thresholds

    Args:
        request: EnrollSubmitRequest with challenge response

    Returns:
        EnrollSubmitResponse with acceptance status and remaining count
    """
    warnings = []

    # Retrieve session
    session = _enrollment_sessions.get(request.session_token)
    if not session:
        return EnrollSubmitResponse(
            accepted=False,
            remaining=0,
            warnings=["invalid_session_token"],
            profile_ready=False
        )

    # Check if session expired
    if session.is_expired():
        return EnrollSubmitResponse(
            accepted=False,
            remaining=0,
            warnings=["session_expired"],
            profile_ready=False
        )

    # Verify user_id matches
    if session.user_id != request.user_id:
        return EnrollSubmitResponse(
            accepted=False,
            remaining=session.get_remaining(),
            warnings=["user_id_mismatch"],
            profile_ready=False
        )

    # Validate challenge response
    is_valid, issues = validate_challenge_response(
        request.challenge_id,
        request.text
    )

    if not is_valid:
        warnings.extend(issues)
        return EnrollSubmitResponse(
            accepted=False,
            remaining=session.get_remaining(),
            warnings=warnings,
            profile_ready=False
        )

    # Normalize text
    try:
        normalized = normalize(request.text)
    except Exception as e:
        warnings.append(f"normalization_error_{str(e)}")
        return EnrollSubmitResponse(
            accepted=False,
            remaining=session.get_remaining(),
            warnings=warnings,
            profile_ready=False
        )

    # Check if normalized text is rejected
    if normalized.get('rejected_reasons'):
        warnings.extend(normalized['rejected_reasons'])
        return EnrollSubmitResponse(
            accepted=False,
            remaining=session.get_remaining(),
            warnings=warnings,
            profile_ready=False
        )

    # Extract features
    try:
        # Get tokens from normalized result
        tokens = normalized.get('tokens', normalized['text'].split())
        style_features, style_stats = extract_features(
            normalized['text'],
            tokens=tokens,
            lang=session.lang
        )
    except Exception as e:
        warnings.append(f"feature_extraction_error_{str(e)}")
        return EnrollSubmitResponse(
            accepted=False,
            remaining=session.get_remaining(),
            warnings=warnings,
            profile_ready=False
        )

    # Encode text
    try:
        embedding = encode([normalized['text']], lang=session.lang)
        embedding = embedding[0]  # Get first (only) embedding
    except Exception as e:
        warnings.append(f"encoding_error_{str(e)}")
        return EnrollSubmitResponse(
            accepted=False,
            remaining=session.get_remaining(),
            warnings=warnings,
            profile_ready=False
        )

    # Store submission (without raw text if TTL=0)
    submission = {
        "challenge_id": request.challenge_id,
        "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
        "style_features": style_features.tolist() if isinstance(style_features, np.ndarray) else style_features,
        "style_stats": style_stats,
        "word_count": normalized.get('word_count', 0),
        "submitted_at": datetime.utcnow().isoformat(),
    }

    # Only store raw text if TTL > 0
    if RAW_TEXT_TTL_DAYS > 0:
        submission["raw_text"] = request.text
        submission["normalized_text"] = normalized['text']

    session.submissions.append(submission)

    # Check if enrollment complete
    profile_ready = False
    if session.is_complete():
        try:
            # Compute profile
            profile_ready = _compute_and_store_profile(session)
            if profile_ready:
                warnings.append("profile_created")
        except Exception as e:
            warnings.append(f"profile_creation_error_{str(e)}")

    return EnrollSubmitResponse(
        accepted=True,
        remaining=session.get_remaining(),
        warnings=warnings,
        profile_ready=profile_ready
    )


def _compute_and_store_profile(session: EnrollmentSession) -> bool:
    """
    Compute user profile from enrollment samples.

    Args:
        session: Completed enrollment session

    Returns:
        True if profile was successfully created
    """
    # Extract all embeddings and style features
    embeddings = []
    style_features = []

    for sub in session.submissions:
        emb = np.array(sub["embedding"], dtype=np.float32)
        style = np.array(sub["style_features"], dtype=np.float32)
        embeddings.append(emb)
        style_features.append(style)

    embeddings = np.array(embeddings)
    style_features = np.array(style_features)

    # Compute centroid (mean embedding)
    centroid = np.mean(embeddings, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)  # Normalize

    # Compute diagonal covariance
    cov_matrix = np.cov(embeddings, rowvar=False)
    cov_diag = np.diag(cov_matrix) if cov_matrix.ndim > 1 else np.array([cov_matrix])

    # Compute style statistics
    style_mean = np.mean(style_features, axis=0)
    style_std = np.std(style_features, axis=0)

    # Aggregate stylometry stats from all submissions
    stylometry_stats = {}
    for sub in session.submissions:
        for key, value in sub.get("style_stats", {}).items():
            if key not in stylometry_stats:
                stylometry_stats[key] = []
            stylometry_stats[key].append(value)

    # Average the stats
    stylometry_stats_aggregated = {
        key: np.mean(values) for key, values in stylometry_stats.items()
    }

    # Calibrate thresholds (if cohort data available)
    # For now, use default thresholds or simple calibration
    thresholds = _calibrate_thresholds_for_user(
        embeddings,
        cohort_vectors=[]  # TODO: Load from database
    )

    # Create profile dictionary
    profile = {
        "user_id": session.user_id,
        "lang": session.lang,
        "domain": session.domain,
        "centroid": centroid,
        "cov_diag": cov_diag,
        "n_samples": len(session.submissions),
        "style_mean": style_mean,
        "style_std": style_std,
        "stylometry_stats": stylometry_stats_aggregated,
        "threshold_high": thresholds["high"],
        "threshold_med": thresholds["med"],
        "last_update": datetime.utcnow()
    }

    # In production, store in database
    # For now, store in memory
    session.profile = profile

    return True


def _calibrate_thresholds_for_user(
    user_vectors: np.ndarray,
    cohort_vectors: List[np.ndarray]
) -> Dict[str, float]:
    """
    Calibrate thresholds for a user.

    Args:
        user_vectors: User's enrollment vectors
        cohort_vectors: Cohort vectors for calibration

    Returns:
        Dictionary with 'high' and 'med' thresholds
    """
    if len(cohort_vectors) > 10:
        # Use calibration with cohort data
        thresholds = calibrate_thresholds(
            user_vectors=list(user_vectors),
            cohort_vectors=cohort_vectors,
            target_far_high=0.005,  # 0.5%
            target_far_med=0.02      # 2%
        )
    else:
        # Use default thresholds
        thresholds = get_default_thresholds()

    return thresholds


def get_session(session_token: str) -> Optional[EnrollmentSession]:
    """
    Retrieve an enrollment session by token.

    Args:
        session_token: Session token

    Returns:
        EnrollmentSession or None if not found
    """
    return _enrollment_sessions.get(session_token)


def cleanup_expired_sessions():
    """Remove expired sessions from memory."""
    expired = [
        token for token, session in _enrollment_sessions.items()
        if session.is_expired()
    ]
    for token in expired:
        del _enrollment_sessions[token]
