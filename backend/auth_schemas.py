"""
Authentication schemas for login and signup.
"""
from pydantic import BaseModel, Field
from typing import Optional


class SignupRequest(BaseModel):
    """Request for user signup."""
    name: str = Field(..., min_length=1, max_length=100)
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class SignupResponse(BaseModel):
    """Response after signup."""
    user_id: str
    username: str
    name: str
    message: str


class LoginRequest(BaseModel):
    """Request for user login."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Response after login (first factor)."""
    user_id: str
    username: str
    name: str
    biometric_enrolled: bool
    requires_mfa: bool
    message: str


class BiometricMFARequest(BaseModel):
    """Request for biometric MFA verification."""
    user_id: str
    text: str
    challenge_id: Optional[str] = None  # Optional for backward compatibility
    timings: Optional[dict] = None


class BiometricMFAResponse(BaseModel):
    """Response after biometric MFA."""
    success: bool
    decision: str
    score: float
    message: str


class MFAChallengeResponse(BaseModel):
    """Response containing MFA challenge prompt."""
    challenge_id: str
    prompt: str
    min_words: int
    timebox_s: int


class IdentifyUserRequest(BaseModel):
    """Request for user identification (1:N matching)."""
    text: str = Field(..., min_length=10)
    timings: Optional[dict] = None


class IdentifyUserResponse(BaseModel):
    """Response after user identification."""
    identified_user: Optional[str] = None  # user_id of best match, None if unknown
    username: Optional[str] = None  # username of best match
    confidence_score: float  # confidence score [0, 1]
    all_scores: dict  # scores for all users (for debugging)
    message: str  # explanation
