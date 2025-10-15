from fastapi import FastAPI, HTTPException, Depends, Request
from contextlib import asynccontextmanager
import os

from schemas import (
    VerifyRequest, VerifyResponse,
    EnrollStartRequest, EnrollStartResponse,
    EnrollSubmitRequest, EnrollSubmitResponse,
    ChallengeStartRequest, ChallengeStartResponse,
    ChallengeSubmitRequest, ChallengeSubmitResponse,
)
from authz import authz
from db import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: connect to database
    await db.connect()
    yield
    # Shutdown: disconnect from database
    await db.disconnect()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest, request: Request):
    """
    Verify a text sample for a claimed user (1:1 verification).

    Returns a decision: allow, challenge, step_up, or deny.
    """
    # Rate limiting
    await authz.check_rate_limit(request, req.user_id)

    # Placeholder: in-progress implementation returns a mocked decision
    if not req.text or len(req.text.split()) < 10:
        raise HTTPException(status_code=400, detail="text too short")

    # Log decision (placeholder score/decision)
    decision_id = await db.log_decision(
        user_id=req.user_id,
        kind="verify",
        lang=req.lang,
        domain=req.domain_hint,
        score=0.5,
        decision="challenge",
        reasons=["DEMO"],
        len_words=len(req.text.split()),
        policy_version="v1",
    )

    return VerifyResponse(
        decision="challenge",
        score=0.5,
        reasons=["DEMO"],
        thresholds={"high": 0.84, "med": 0.72},
    )


@app.post("/enroll/start", response_model=EnrollStartResponse)
async def enroll_start(req: EnrollStartRequest, request: Request):
    """
    Start enrollment flow: return a sequence of challenges.
    """
    await authz.check_rate_limit(request, req.user_id)

    # TODO: implement challenge selection and session token generation
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.post("/enroll/submit", response_model=EnrollSubmitResponse)
async def enroll_submit(req: EnrollSubmitRequest, request: Request):
    """
    Submit an enrollment sample.
    """
    await authz.check_rate_limit(request, req.user_id)

    # TODO: implement enrollment sample processing
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.post("/challenge/prepare", response_model=ChallengeStartResponse)
async def challenge_prepare(req: ChallengeStartRequest, request: Request):
    """
    Prepare a single-use challenge for step-up authentication.
    """
    await authz.check_rate_limit(request, req.user_id)

    # TODO: implement challenge generation
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.post("/challenge/submit", response_model=ChallengeSubmitResponse)
async def challenge_submit(req: ChallengeSubmitRequest, request: Request):
    """
    Submit a challenge response and get a decision.
    """
    await authz.check_rate_limit(request, req.user_id)

    # TODO: implement challenge scoring
    raise HTTPException(status_code=501, detail="Not implemented yet")
