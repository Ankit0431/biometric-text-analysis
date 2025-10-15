"""
Authorization, authentication, and rate limiting utilities.
"""
from fastapi import HTTPException, Header, Request
from typing import Optional
import time
from collections import defaultdict


class AuthZ:
    """Simple authorization and rate limiting for prototype."""

    def __init__(self):
        # In-memory rate limit tracking (use Redis in production)
        self.rate_limits = defaultdict(list)
        self.rate_window_seconds = 60
        self.max_requests_per_window = 100

    async def verify_token(self, authorization: Optional[str] = Header(None)) -> dict:
        """
        Verify authorization header and extract user/tenant info.

        For prototype: simple header-based auth.
        In production: validate JWT or OAuth token.

        Returns:
            Dict with user_id, tenant_id
        """
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header required")

        # Prototype: expect "Bearer <user_id>:<tenant_id>"
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization format")

        token = authorization.replace("Bearer ", "")
        parts = token.split(":")

        if len(parts) != 2:
            raise HTTPException(status_code=401, detail="Invalid token format")

        user_id, tenant_id = parts
        return {"user_id": user_id, "tenant_id": tenant_id}

    async def check_rate_limit(self, request: Request, identifier: str):
        """
        Simple in-memory rate limiter.

        Args:
            request: FastAPI request
            identifier: user_id, IP, or other identifier

        Raises:
            HTTPException if rate limit exceeded
        """
        now = time.time()
        window_start = now - self.rate_window_seconds

        # Clean old entries
        self.rate_limits[identifier] = [
            ts for ts in self.rate_limits[identifier] if ts > window_start
        ]

        # Check limit
        if len(self.rate_limits[identifier]) >= self.max_requests_per_window:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.max_requests_per_window} requests per {self.rate_window_seconds}s"
            )

        # Record this request
        self.rate_limits[identifier].append(now)

    def require_consent(self, user_id: str, consent_version: str):
        """
        Check if user has accepted the required consent version.

        For prototype: placeholder that returns True.
        In production: query database for user consent record.
        """
        # TODO: implement actual consent checking
        return True


# Global authz instance
authz = AuthZ()
