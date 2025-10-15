"""
Redis cache for profile data and session management.

This module provides caching for:
- User profiles (centroid, thresholds, stats) to reduce DB hits
- Challenge sessions for verify flow
"""
import redis.asyncio as redis
import os
import json
import numpy as np
from typing import Optional, Dict, Any


class RedisCache:
    """Async Redis cache manager."""

    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.profile_ttl = int(os.getenv("PROFILE_CACHE_TTL", "3600"))  # 1 hour default

    async def connect(self):
        """Initialize Redis connection."""
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD", None),
            db=int(os.getenv("REDIS_DB", "0")),
            decode_responses=False,  # We'll handle encoding manually
        )

    async def disconnect(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()

    def _profile_key(self, user_id: str, lang: str, domain: str) -> str:
        """Generate cache key for profile."""
        return f"profile:{user_id}:{lang}:{domain}"

    def _challenge_key(self, challenge_id: str) -> str:
        """Generate cache key for challenge session."""
        return f"challenge:{challenge_id}"

    async def get_profile(self, user_id: str, lang: str, domain: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached profile data.

        Returns:
            Dict with profile data or None if not in cache
        """
        if not self.redis:
            return None

        key = self._profile_key(user_id, lang, domain)
        data = await self.redis.get(key)

        if not data:
            return None

        # Deserialize JSON
        profile = json.loads(data.decode('utf-8'))

        # Convert centroid back to numpy array
        if 'centroid' in profile:
            profile['centroid'] = np.array(profile['centroid'], dtype=np.float32)

        return profile

    async def set_profile(
        self,
        user_id: str,
        lang: str,
        domain: str,
        profile: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """
        Cache profile data.

        Args:
            user_id: User identifier
            lang: Language code
            domain: Domain hint
            profile: Profile data dict
            ttl: Time-to-live in seconds (uses default if None)
        """
        if not self.redis:
            return

        key = self._profile_key(user_id, lang, domain)
        ttl = ttl or self.profile_ttl

        # Serialize profile (convert numpy to list)
        cache_data = profile.copy()
        if 'centroid' in cache_data and isinstance(cache_data['centroid'], np.ndarray):
            cache_data['centroid'] = cache_data['centroid'].tolist()

        await self.redis.setex(
            key,
            ttl,
            json.dumps(cache_data).encode('utf-8')
        )

    async def invalidate_profile(self, user_id: str, lang: str, domain: str):
        """
        Remove profile from cache.
        """
        if not self.redis:
            return

        key = self._profile_key(user_id, lang, domain)
        await self.redis.delete(key)

    async def set_challenge(
        self,
        challenge_id: str,
        data: Dict[str, Any],
        ttl: int = 300  # 5 minutes default
    ):
        """
        Store challenge session data.

        Args:
            challenge_id: Unique challenge identifier
            data: Challenge data (user_id, prompt, created_at, etc.)
            ttl: Time-to-live in seconds
        """
        if not self.redis:
            return

        key = self._challenge_key(challenge_id)
        await self.redis.setex(
            key,
            ttl,
            json.dumps(data).encode('utf-8')
        )

    async def get_challenge(self, challenge_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve challenge session data.

        Returns:
            Dict with challenge data or None if not found/expired
        """
        if not self.redis:
            return None

        key = self._challenge_key(challenge_id)
        data = await self.redis.get(key)

        if not data:
            return None

        return json.loads(data.decode('utf-8'))

    async def delete_challenge(self, challenge_id: str):
        """
        Delete challenge session (single-use).
        """
        if not self.redis:
            return

        key = self._challenge_key(challenge_id)
        await self.redis.delete(key)

    async def ping(self) -> bool:
        """
        Test Redis connection.

        Returns:
            True if connected, False otherwise
        """
        if not self.redis:
            return False

        try:
            await self.redis.ping()
            return True
        except Exception:
            return False


# Global cache instance
cache = RedisCache()
