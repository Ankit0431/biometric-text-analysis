from fastapi import FastAPI, HTTPException, Depends, Request
from contextlib import asynccontextmanager
import os
import uuid

from schemas import (
    VerifyRequest, VerifyResponse,
    EnrollStartRequest, EnrollStartResponse,
    EnrollSubmitRequest, EnrollSubmitResponse,
    ChallengeStartRequest, ChallengeStartResponse,
    ChallengeSubmitRequest, ChallengeSubmitResponse,
)
from authz import authz
from db import db
from enroll_handlers import enroll_start, enroll_submit
from verify_handlers import VerifyHandler, ChallengeHandler
from cache import RedisCache
from encoder import TextEncoder

# Create singleton Redis cache instance
redis_cache = RedisCache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: connect to database and Redis
    await db.connect()
    await redis_cache.connect()
    print("✅ Database and Redis connected successfully")

    yield

    # Shutdown: disconnect from database and Redis
    await db.disconnect()
    await redis_cache.disconnect()


app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    # Check consent for keystroke data
    if req.timings:
        await authz.check_consent(req.user_id, "keystrokes")

    try:
        # Get user profile from database
        profile = await db.get_profile(req.user_id, req.lang, req.domain_hint)

        if not profile:
            return VerifyResponse(
                decision="deny",
                score=0.0,
                reasons=["USER_NOT_ENROLLED"],
                thresholds={"high": 0.84, "med": 0.72},
            )

        # Initialize verify handler with encoder
        encoder = TextEncoder()
        handler = VerifyHandler(encoder=encoder)

        # Verify the sample
        result = await handler.verify_sample(
            user_id=req.user_id,
            text=req.text,
            profile=profile,
            lang=req.lang,
            domain=req.domain_hint,
            timings=req.timings,
            context=req.context
        )

        # Log decision to database
        await db.log_decision(
            user_id=req.user_id,
            kind="verify",
            lang=req.lang,
            domain=req.domain_hint,
            score=result['score'],
            decision=result['decision'],
            reasons=result['reasons'],
            len_words=len(req.text.split()),
            policy_version="v1",
        )

        return VerifyResponse(
            decision=result['decision'],
            score=result['score'],
            reasons=result['reasons'],
            thresholds=result['thresholds'],
        )

    except Exception as e:
        # Log error and return safe fallback
        print(f"Error in verify endpoint: {e}")
        return VerifyResponse(
            decision="deny",
            score=0.0,
            reasons=["ERROR_INTERNAL"],
            thresholds={"high": 0.84, "med": 0.72},
        )


@app.post("/enroll/start", response_model=EnrollStartResponse)
async def api_enroll_start(req: EnrollStartRequest, request: Request):
    """
    Start enrollment flow: return a sequence of challenges.
    """
    await authz.check_rate_limit(request, req.user_id)

    try:
        # Call enroll_start function
        result = enroll_start(req)
        return result

    except Exception as e:
        print(f"Error in enroll/start endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start enrollment: {str(e)}")


@app.post("/enroll/submit", response_model=EnrollSubmitResponse)
async def api_enroll_submit(req: EnrollSubmitRequest, request: Request):
    """
    Submit an enrollment sample.
    """
    await authz.check_rate_limit(request, req.user_id)

    # Check consent for keystroke data
    if req.timings:
        await authz.check_consent(req.user_id, "keystrokes")

    try:
        # Call enroll_submit function
        result = enroll_submit(req)
        return result

    except Exception as e:
        print(f"Error in enroll/submit endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit sample: {str(e)}")


@app.post("/challenge/prepare", response_model=ChallengeStartResponse)
async def challenge_prepare(req: ChallengeStartRequest, request: Request):
    """
    Prepare a single-use challenge for step-up authentication.
    """
    await authz.check_rate_limit(request, req.user_id)

    try:
        # Initialize challenge handler
        handler = ChallengeHandler(
            user_id=req.user_id,
            lang=req.lang,
            domain=req.domain
        )

        # Prepare challenge
        result = await handler.prepare_challenge()

        return ChallengeStartResponse(
            challenge_id=result['challenge_id'],
            prompt=result['prompt'],
            min_words=result['min_words'],
            timebox_s=result['timebox_s'],
            constraints=result['constraints']
        )

    except Exception as e:
        print(f"Error in challenge/prepare endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare challenge: {str(e)}")


@app.post("/challenge/submit", response_model=ChallengeSubmitResponse)
async def challenge_submit(req: ChallengeSubmitRequest, request: Request):
    """
    Submit a challenge response and get a decision.
    """
    await authz.check_rate_limit(request, req.user_id)

    # Check consent for keystroke data
    if req.timings:
        await authz.check_consent(req.user_id, "keystrokes")

    try:
        # Initialize challenge handler
        handler = ChallengeHandler(
            user_id=req.user_id,
            lang="en",
            domain="chat"
        )

        # Submit challenge response
        result = await handler.submit_challenge(
            challenge_id=req.challenge_id,
            text=req.text,
            timings=req.timings
        )

        # Log decision
        await db.log_decision(
            user_id=req.user_id,
            kind="challenge",
            lang="en",
            domain="chat",
            score=result['score'],
            decision=result['decision'],
            reasons=result['reasons'],
            len_words=len(req.text.split()),
            policy_version="v1",
        )

        return ChallengeSubmitResponse(
            decision=result['decision'],
            score=result['score'],
            reasons=result['reasons'],
            thresholds=result['thresholds']
        )

    except Exception as e:
        print(f"Error in challenge/submit endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit challenge: {str(e)}")
