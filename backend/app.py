from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from schemas import (
    VerifyRequest, VerifyResponse,
    EnrollStartRequest, EnrollStartResponse,
    EnrollSubmitRequest, EnrollSubmitResponse,
    ChallengeStartRequest, ChallengeStartResponse,
    ChallengeSubmitRequest, ChallengeSubmitResponse,
)
from auth_schemas import (
    SignupRequest, SignupResponse,
    LoginRequest, LoginResponse,
    BiometricMFARequest, BiometricMFAResponse,
    MFAChallengeResponse
)
from authz import authz
from db import db
from enroll_handlers import enroll_start, enroll_submit
from verify_handlers import VerifyHandler, ChallengeHandler
from cache import RedisCache
from encoder import TextEncoder, get_encoder
from auth_handler import signup_user, login_user

# Create singleton Redis cache instance
redis_cache = RedisCache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: connect to database and Redis
    await db.connect()
    await redis_cache.connect()
    
    # Initialize global encoder
    get_encoder()
    
    # Initialize adaptive profile management system
    from adaptive_profiles import initialize_adaptive_system, shutdown_adaptive_system
    await initialize_adaptive_system(db)
    
    print("✅ Database, Redis, Encoder, and Adaptive Profiles initialized successfully")

    yield

    # Shutdown: disconnect from database, Redis, and adaptive system
    await shutdown_adaptive_system()
    await db.disconnect()
    await redis_cache.disconnect()


app = FastAPI(lifespan=lifespan)

# Add validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error: {exc.errors()}")
    print(f"Request body: {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(await request.body())}
    )

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


@app.post("/auth/signup", response_model=SignupResponse)
async def api_signup(req: SignupRequest):
    """
    Sign up a new user with username and password.
    """
    try:
        result = await signup_user(req.name, req.username, req.password)
        return SignupResponse(
            user_id=result["user_id"],
            username=result["username"],
            name=result["name"],
            message="Signup successful. Please complete biometric enrollment."
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error in signup endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to create account")


@app.post("/auth/login", response_model=LoginResponse)
async def api_login(req: LoginRequest):
    """
    Authenticate user with username and password (first factor).
    """
    try:
        result = await login_user(req.username, req.password)
        return LoginResponse(
            user_id=result["user_id"],
            username=result["username"],
            name=result["name"],
            biometric_enrolled=result["biometric_enrolled"],
            requires_mfa=result["biometric_enrolled"],
            message="Login successful" if not result["biometric_enrolled"] else "Please complete biometric MFA"
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        print(f"Error in login endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to authenticate")


@app.get("/auth/mfa-challenge", response_model=MFAChallengeResponse)
async def api_get_mfa_challenge(user_id: str):
    """
    Get a challenge prompt for MFA verification.
    Selects a prompt from the same categories used during enrollment but different from enrolled prompts.
    """
    try:
        # Get user profile to find enrolled challenge IDs
        profile = await db.get_profile(user_id, "en", "chat")
        
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found. Please complete enrollment first.")
        
        # Extract enrolled challenge IDs from prompt_answers
        prompt_answers = profile.get("prompt_answers", {})
        enrolled_challenge_ids = list(prompt_answers.keys()) if prompt_answers else []
        
        # If no enrolled challenges found, fall back to any challenge
        if not enrolled_challenge_ids:
            from challenge_bank import select_challenges
            challenges = select_challenges(num_challenges=1)
            challenge = challenges[0]
        else:
            # Select an MFA challenge related to enrolled categories
            from challenge_bank import select_mfa_challenge
            challenge = select_mfa_challenge(enrolled_challenge_ids)
        
        return MFAChallengeResponse(
            challenge_id=challenge.challenge_id,
            prompt=challenge.prompt,
            min_words=challenge.min_words,
            timebox_s=challenge.timebox_s
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting MFA challenge: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate challenge")


@app.post("/auth/biometric-mfa", response_model=BiometricMFAResponse)
async def api_biometric_mfa(req: BiometricMFARequest, request: Request):
    """
    Verify biometric text as second factor (MFA).
    This performs 1:1 matching using only the user's enrolled profile.
    """
    print(f"DEBUG: Received MFA request - user_id: {req.user_id}, text length: {len(req.text)}, timings: {type(req.timings)}")
    try:
        # Get user profile from database (1:1 matching only)
        profile = await db.get_profile(req.user_id, "en", "chat")
        
        if not profile:
            return BiometricMFAResponse(
                success=False,
                decision="deny",
                score=0.0,
                message="Biometric profile not found. Please complete enrollment first."
            )
        
        # Debug: Check profile data
        print(f"DEBUG PROFILE: centroid shape={profile['centroid'].shape if hasattr(profile['centroid'], 'shape') else 'N/A'}, style_mean={type(profile.get('style_mean'))}, style_std={type(profile.get('style_std'))}")
        
        # Initialize verify handler with GLOBAL encoder (same one used during enrollment)
        encoder = get_encoder()
        handler = VerifyHandler(encoder=encoder, db=db)
        
        # Verify the sample (1:1 matching against user's profile only)
        result = await handler.verify_sample(
            user_id=req.user_id,
            text=req.text,
            profile=profile,
            lang="en",
            domain="chat",
            timings=req.timings,
            context={}
        )
        
        # Debug logging
        print(f"DEBUG: Verification result - score: {result['score']}, decision: {result['decision']}, reasons: {result['reasons']}")
        print(f"DEBUG: Thresholds - high: {result['thresholds']['high']}, med: {result['thresholds']['med']}")
        
        # Log decision to database
        await db.log_decision(
            user_id=req.user_id,
            kind="biometric_mfa",
            lang="en",
            domain="chat",
            score=result['score'],
            decision=result['decision'],
            reasons=result['reasons'],
            len_words=len(req.text.split()),
            policy_version="v1",
        )
        
        success = result['decision'] == 'allow'
        
        # Check if LLM was detected
        if 'LLM_GENERATED_TEXT_DETECTED' in result['reasons']:
            message = result.get('message', 'AI-generated text detected. Please write naturally in your own words.')
        elif success:
            message = "Authentication successful"
        else:
            message = f"Authentication failed: {', '.join(result['reasons'])}"
        
        return BiometricMFAResponse(
            success=success,
            decision=result['decision'],
            score=result['score'],
            message=message
        )
    
    except Exception as e:
        print(f"Error in biometric MFA endpoint: {e}")
        return BiometricMFAResponse(
            success=False,
            decision="deny",
            score=0.0,
            message="An error occurred during authentication"
        )



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
        handler = VerifyHandler(encoder=encoder, db=db)

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
        result = await enroll_submit(req)
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
        encoder = TextEncoder()
        handler = ChallengeHandler(encoder=encoder, db=db)

        # Prepare challenge
        result = handler.prepare_challenge(
            user_id=req.user_id,
            lang=req.lang,
            domain=req.domain
        )

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
        encoder = TextEncoder()
        handler = ChallengeHandler(encoder=encoder, db=db)

        # Get user profile
        profile = await db.get_profile(req.user_id, lang="en", domain="chat")
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Submit challenge response
        result = await handler.submit_challenge(
            challenge_id=req.challenge_id,
            user_id=req.user_id,
            text=req.text,
            profile=profile,
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
