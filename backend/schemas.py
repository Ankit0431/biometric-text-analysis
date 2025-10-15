"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any


class VerifyRequest(BaseModel):
    """Request payload for /verify endpoint."""
    user_id: str = Field(..., min_length=1, max_length=255)
    text: str = Field(..., min_length=10)
    lang: str = Field(default="en", max_length=10)
    domain_hint: str = Field(default="chat", max_length=50)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    timings: Optional[Dict[str, Any]] = None

    @validator("text")
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("text cannot be empty")
        return v


class VerifyResponse(BaseModel):
    """Response payload for /verify endpoint."""
    decision: str  # allow | challenge | step_up | deny
    score: float
    reasons: List[str]
    thresholds: Dict[str, float]


class EnrollStartRequest(BaseModel):
    """Request payload for /enroll/start."""
    user_id: str = Field(..., min_length=1, max_length=255)
    lang: str = Field(default="en", max_length=10)
    domain: str = Field(default="chat", max_length=50)


class ChallengeInfo(BaseModel):
    """Challenge metadata."""
    challenge_id: str
    prompt: str
    min_words: int
    timebox_s: int
    constraints: List[str]


class EnrollStartResponse(BaseModel):
    """Response payload for /enroll/start."""
    challenges: List[ChallengeInfo]
    session_token: str
    required_samples: int


class EnrollSubmitRequest(BaseModel):
    """Request payload for /enroll/submit."""
    challenge_id: str
    user_id: str
    text: str = Field(..., min_length=50)
    timings: Optional[Dict[str, Any]] = None
    session_token: str


class EnrollSubmitResponse(BaseModel):
    """Response payload for /enroll/submit."""
    accepted: bool
    remaining: int
    warnings: List[str]
    profile_ready: bool = False


class ChallengeStartRequest(BaseModel):
    """Request payload for /challenge/prepare."""
    user_id: str
    lang: str = Field(default="en")
    domain: str = Field(default="chat")


class ChallengeStartResponse(BaseModel):
    """Response for /challenge/prepare."""
    challenge_id: str
    prompt: str
    min_words: int
    timebox_s: int
    constraints: List[str]


class ChallengeSubmitRequest(BaseModel):
    """Request for /challenge/submit."""
    challenge_id: str
    user_id: str
    text: str
    timings: Optional[Dict[str, Any]] = None


class ChallengeSubmitResponse(BaseModel):
    """Response for /challenge/submit."""
    decision: str
    score: float
    reasons: List[str]
    thresholds: Dict[str, float]
