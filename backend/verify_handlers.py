"""
Handlers for /verify and /challenge endpoints.

This module implements:
- Verify flow: score a sample against user profile
- Challenge flow: prepare and submit step-up challenges
"""
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np

from normalizer import normalize
from features import extract_features
from encoder import TextEncoder
from scoring import score_sample
from policy import decide, get_default_thresholds
from challenge_bank import select_challenges, get_challenge_by_id


class VerifyHandler:
    """Handler for verify endpoint logic."""

    def __init__(self, encoder: TextEncoder):
        self.encoder = encoder

    async def verify_sample(
        self,
        user_id: str,
        text: str,
        profile: Dict[str, Any],
        lang: str = "en",
        domain: str = "chat",
        timings: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Score a text sample against a user profile and return decision.

        Args:
            user_id: User identifier
            text: Raw text to verify
            profile: User profile with centroid, thresholds, etc.
            lang: Language code
            domain: Domain hint
            timings: Optional keystroke timing data
            context: Optional context metadata

        Returns:
            Dict with decision, score, reasons, thresholds
        """
        # Normalize text
        normalized = normalize(text)

        # Check rejection reasons
        if normalized.rejected_reasons:
            # For short text, force challenge
            if "SHORT_LEN" in normalized.rejected_reasons:
                return {
                    "decision": "challenge",
                    "score": 0.0,
                    "reasons": ["SHORT_LEN"],
                    "thresholds": {
                        "high": profile.get("threshold_high", 0.84),
                        "med": profile.get("threshold_med", 0.72),
                    },
                }
            # Other rejections -> deny
            return {
                "decision": "deny",
                "score": 0.0,
                "reasons": normalized.rejected_reasons,
                "thresholds": {
                    "high": profile.get("threshold_high", 0.84),
                    "med": profile.get("threshold_med", 0.72),
                },
            }

        # Compute stylometry features
        stylo_vec, stylo_stats = extract_features(
            normalized.text,
            normalized.tokens,
            lang=lang
        )
        
        # Detect LLM-generated text FIRST and reject immediately if detected
        from scoring import detect_llm_likeness
        llm_penalty, is_llm_like = detect_llm_likeness(normalized.text, stylo_stats)
        
        # Always log LLM detection results for debugging
        print(f"DEBUG LLM CHECK: penalty={llm_penalty:.4f}, is_llm_like={is_llm_like}")
        print(f"  text_length={len(normalized.text)}, num_sentences={normalized.text.count('.')}")
        
        if is_llm_like:
            print(f"⚠️ REJECTING: LLM-generated text detected!")
            return {
                "decision": "deny",
                "score": 0.0,
                "reasons": ["LLM_GENERATED_TEXT_DETECTED"],
                "thresholds": {
                    "high": profile.get("threshold_high", 0.84),
                    "med": profile.get("threshold_med", 0.72),
                },
                "message": "AI-generated text detected. Please write naturally in your own words."
            }

        # Encode text
        embedding = self.encoder.encode([normalized.text])[0]

        # Score the sample
        score_result = score_sample(
            user_profile=profile,
            text=normalized.text,
            embedding=embedding,
            style_features=stylo_vec,
            timings=timings,
            context=context or {},
        )
        
        # Debug logging
        print(f"DEBUG SCORING: semantic={score_result['semantic_score']:.4f}, stylometry={score_result['stylometry_score']:.4f}, keystroke={score_result.get('keystroke_score', 0.5):.4f}, llm_penalty={score_result['llm_penalty']:.4f}, final={score_result['final_score']:.4f}")

        # Apply policy decision
        decision_result = decide(
            score=score_result["final_score"],
            threshold_high=profile.get("threshold_high", 0.84),
            threshold_med=profile.get("threshold_med", 0.72),
            text=normalized.text,
            word_count=normalized.word_count,
            llm_like=score_result.get("llm_like", False),
        )

        return {
            "decision": decision_result["decision"],
            "score": score_result["final_score"],
            "reasons": decision_result["reasons"],
            "thresholds": {
                "high": profile.get("threshold_high", 0.84),
                "med": profile.get("threshold_med", 0.72),
            },
        }


class ChallengeHandler:
    """Handler for challenge preparation and submission."""

    def __init__(self, encoder: TextEncoder):
        self.encoder = encoder
        self.verify_handler = VerifyHandler(encoder)

    def prepare_challenge(
        self,
        user_id: str,
        lang: str = "en",
        domain: str = "chat",
    ) -> Dict[str, Any]:
        """
        Prepare a single-use challenge for step-up authentication.

        Args:
            user_id: User identifier
            lang: Language code
            domain: Domain hint

        Returns:
            Dict with challenge_id, prompt, min_words, timebox_s, constraints
        """
        # Select a single challenge
        challenges = select_challenges(num_challenges=1, seed=None)

        if not challenges:
            raise ValueError("No challenges available")

        challenge = challenges[0]

        # Generate unique challenge ID
        challenge_id = str(uuid.uuid4())

        # Return challenge metadata
        return {
            "challenge_id": challenge_id,
            "prompt": challenge.prompt,
            "min_words": challenge.min_words,
            "timebox_s": challenge.timebox_s,
            "constraints": challenge.constraints,
            "internal_challenge_ref": challenge.challenge_id,  # Store for validation
        }

    async def submit_challenge(
        self,
        challenge_id: str,
        user_id: str,
        text: str,
        profile: Dict[str, Any],
        challenge_data: Optional[Dict[str, Any]] = None,
        lang: str = "en",
        domain: str = "chat",
        timings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit and score a challenge response.

        Args:
            challenge_id: Unique challenge identifier
            user_id: User identifier
            text: Challenge response text
            profile: User profile
            challenge_data: Challenge metadata from cache
            lang: Language code
            domain: Domain hint
            timings: Optional keystroke timing data

        Returns:
            Dict with decision, score, reasons, thresholds
        """
        # Validate challenge exists
        if challenge_data is None:
            return {
                "decision": "deny",
                "score": 0.0,
                "reasons": ["INVALID_CHALLENGE"],
                "thresholds": get_default_thresholds(),
            }

        # Validate user matches
        if challenge_data.get("user_id") != user_id:
            return {
                "decision": "deny",
                "score": 0.0,
                "reasons": ["USER_MISMATCH"],
                "thresholds": get_default_thresholds(),
            }

        # Validate challenge from bank
        internal_ref = challenge_data.get("internal_challenge_ref")
        if internal_ref:
            challenge_def = get_challenge_by_id(internal_ref)
            if challenge_def:
                # Check word count requirement
                word_count = len(text.split())
                if word_count < challenge_def["min_words"]:
                    return {
                        "decision": "deny",
                        "score": 0.0,
                        "reasons": ["CHALLENGE_TOO_SHORT"],
                        "thresholds": get_default_thresholds(),
                    }

        # Score using verify logic
        result = await self.verify_handler.verify_sample(
            user_id=user_id,
            text=text,
            profile=profile,
            lang=lang,
            domain=domain,
            timings=timings,
        )

        return result
