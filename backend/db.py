"""
Database connection and helper functions for Postgres with pgvector.
"""
import asyncpg # type: ignore
import os
from typing import Optional, List, Dict, Any
import json
import numpy as np


class Database:
    """Async Postgres connection pool manager."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Initialize connection pool."""
        self.pool = await asyncpg.create_pool(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            database=os.getenv("POSTGRES_DB", "biometric"),
            min_size=2,
            max_size=10,
        )

    async def disconnect(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()

    async def get_profile(self, user_id: str, lang: str, domain: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user profile by (user_id, lang, domain).

        Returns:
            Dict with profile data or None if not found
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT user_id, lang, domain, centroid, cov_diag, n_samples,
                       stylometry_stats, threshold_high, threshold_med, prompt_answers, last_update
                FROM profiles
                WHERE user_id = $1 AND lang = $2 AND domain = $3
                """,
                user_id, lang, domain
            )
            if row:
                # pgvector returns the vector as a string like "[1.0, 2.0, ...]"
                # Parse it back to numpy array
                centroid_str = row["centroid"]
                if isinstance(centroid_str, str):
                    # Remove brackets and split by comma
                    centroid_list = json.loads(centroid_str)
                    centroid = np.array(centroid_list, dtype=np.float32)
                else:
                    centroid = np.array(centroid_str, dtype=np.float32)

                # Parse stylometry stats and extract style_mean/style_std/keystroke_mean/keystroke_std
                stylometry_stats = json.loads(row["stylometry_stats"]) if isinstance(row["stylometry_stats"], str) else row["stylometry_stats"]
                style_mean = stylometry_stats.get('style_mean')
                style_std = stylometry_stats.get('style_std')
                keystroke_mean = stylometry_stats.get('keystroke_mean')
                keystroke_std = stylometry_stats.get('keystroke_std')
                
                # Parse prompt_answers
                prompt_answers = None
                if row["prompt_answers"]:
                    prompt_answers = json.loads(row["prompt_answers"]) if isinstance(row["prompt_answers"], str) else row["prompt_answers"]
                
                return {
                    "user_id": row["user_id"],
                    "lang": row["lang"],
                    "domain": row["domain"],
                    "centroid": centroid,
                    "cov_diag": row["cov_diag"],
                    "n_samples": row["n_samples"],
                    "stylometry_stats": stylometry_stats,
                    "style_mean": np.array(style_mean, dtype=np.float32) if style_mean else None,
                    "style_std": np.array(style_std, dtype=np.float32) if style_std else None,
                    "keystroke_mean": np.array(keystroke_mean, dtype=np.float32) if keystroke_mean else None,
                    "keystroke_std": np.array(keystroke_std, dtype=np.float32) if keystroke_std else None,
                    "prompt_answers": prompt_answers,
                    "threshold_high": row["threshold_high"],
                    "threshold_med": row["threshold_med"],
                    "last_update": row["last_update"],
                }
            return None

    async def upsert_profile(
        self,
        user_id: str,
        lang: str,
        domain: str,
        centroid: np.ndarray,
        cov_diag: List[float],
        n_samples: int,
        stylometry_stats: Dict[str, Any],
        threshold_high: float,
        threshold_med: float,
        prompt_answers: Optional[Dict[str, Any]] = None,
    ):
        """
        Insert or update a user profile.
        """
        async with self.pool.acquire() as conn:
            # Convert centroid to list of floats for pgvector
            centroid_list = centroid.tolist() if isinstance(centroid, np.ndarray) else centroid
            # Format as pgvector string: "[1.0, 2.0, 3.0]"
            centroid_str = str(centroid_list)

            await conn.execute(
                """
                INSERT INTO profiles
                (user_id, lang, domain, centroid, cov_diag, n_samples,
                 stylometry_stats, threshold_high, threshold_med, prompt_answers, last_update)
                VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8, $9, $10, now())
                ON CONFLICT (user_id, lang, domain)
                DO UPDATE SET
                    centroid = $4::vector,
                    cov_diag = $5,
                    n_samples = $6,
                    stylometry_stats = $7,
                    threshold_high = $8,
                    threshold_med = $9,
                    prompt_answers = $10,
                    last_update = now()
                """,
                user_id, lang, domain, centroid_str, cov_diag, n_samples,
                json.dumps(stylometry_stats), threshold_high, threshold_med,
                json.dumps(prompt_answers) if prompt_answers else None
            )

    async def log_decision(
        self,
        user_id: str,
        kind: str,
        lang: str,
        domain: str,
        score: float,
        decision: str,
        reasons: List[str],
        len_words: int,
        policy_version: str = "v1",
    ) -> int:
        """
        Log a decision to the decisions table.

        Returns:
            The ID of the inserted decision record
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO decisions
                (user_id, kind, lang, domain, score, decision, reasons, len_words, policy_version, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, now())
                RETURNING id
                """,
                user_id, kind, lang, domain, score, decision, json.dumps(reasons), len_words, policy_version
            )
            return row["id"]

    async def sample_cohort_vectors(
        self, lang: str, domain: str, exclude_user_id: str, limit: int = 200
    ) -> List[np.ndarray]:
        """
        Sample cohort vectors for impostor distribution (used in calibration).

        Returns:
            List of numpy arrays representing cohort vectors
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT vec FROM cohort_vectors
                WHERE lang = $1 AND domain = $2 AND user_id != $3
                ORDER BY random()
                LIMIT $4
                """,
                lang, domain, exclude_user_id, limit
            )
            return [np.array(row["vec"], dtype=np.float32) for row in rows]

    async def create_user(
        self, user_id: str, tenant_id: str, locale: str = None, consent_version: str = None
    ):
        """Create a new user record."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (user_id, tenant_id, locale, consent_version, created_at)
                VALUES ($1, $2, $3, $4, now())
                ON CONFLICT (user_id) DO NOTHING
                """,
                user_id, tenant_id, locale, consent_version
            )

    async def create_user_with_auth(
        self, user_id: str, username: str, password_hash: str, name: str, 
        tenant_id: str, locale: str = None
    ):
        """Create a new user with authentication credentials."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (user_id, username, password_hash, name, tenant_id, locale, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, now())
                """,
                user_id, username, password_hash, name, tenant_id, locale
            )

    async def get_user_by_username(self, username: str):
        """Get user by username."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT user_id, username, password_hash, name, biometric_enrolled, 
                       tenant_id, locale, created_at
                FROM users
                WHERE username = $1 AND deleted_at IS NULL
                """,
                username
            )
            if row:
                return dict(row)
            return None

    async def update_biometric_enrolled(self, user_id: str, enrolled: bool = True):
        """Update biometric enrollment status."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE users SET biometric_enrolled = $2 WHERE user_id = $1
                """,
                user_id, enrolled
            )

    async def delete_user(self, user_id: str):
        """Soft delete a user (set deleted_at timestamp)."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE users SET deleted_at = now() WHERE user_id = $1
                """,
                user_id
            )

    async def get_all_enrolled_profiles(self, lang: str = "en", domain: str = "chat") -> List[Dict[str, Any]]:
        """
        Retrieve all enrolled user profiles for a given language and domain.
        Used for 1:N identification.

        Args:
            lang: Language code (default: "en")
            domain: Domain (default: "chat")

        Returns:
            List of profile dictionaries with user information
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT p.user_id, p.lang, p.domain, p.centroid, p.cov_diag, p.n_samples,
                       p.stylometry_stats, p.threshold_high, p.threshold_med, 
                       u.username, u.name
                FROM profiles p
                LEFT JOIN users u ON p.user_id = u.user_id
                WHERE p.lang = $1 AND p.domain = $2 AND p.n_samples >= 3
                ORDER BY p.last_update DESC
                """,
                lang, domain
            )
            
            profiles = []
            for row in rows:
                # Parse centroid
                centroid_str = row["centroid"]
                if isinstance(centroid_str, str):
                    centroid_list = json.loads(centroid_str)
                    centroid = np.array(centroid_list, dtype=np.float32)
                else:
                    centroid = np.array(centroid_str, dtype=np.float32)

                # Parse stylometry stats
                stylometry_stats = json.loads(row["stylometry_stats"]) if isinstance(row["stylometry_stats"], str) else row["stylometry_stats"]
                style_mean = stylometry_stats.get('style_mean')
                style_std = stylometry_stats.get('style_std')
                keystroke_mean = stylometry_stats.get('keystroke_mean')
                keystroke_std = stylometry_stats.get('keystroke_std')
                
                # Convert lists to numpy arrays safely
                if isinstance(style_mean, list):
                    style_mean = np.array(style_mean, dtype=np.float32)
                if isinstance(style_std, list):
                    style_std = np.array(style_std, dtype=np.float32)
                if isinstance(keystroke_mean, list):
                    keystroke_mean = np.array(keystroke_mean, dtype=np.float32)
                if isinstance(keystroke_std, list):
                    keystroke_std = np.array(keystroke_std, dtype=np.float32)
                
                profiles.append({
                    "user_id": row["user_id"],
                    "username": row["username"],
                    "name": row["name"],
                    "lang": row["lang"],
                    "domain": row["domain"],
                    "centroid": centroid,
                    "cov_diag": row["cov_diag"],
                    "n_samples": row["n_samples"],
                    "stylometry_stats": stylometry_stats,
                    "style_mean": style_mean,
                    "style_std": style_std,
                    "keystroke_mean": keystroke_mean,
                    "keystroke_std": keystroke_std,
                    "threshold_high": row["threshold_high"],
                    "threshold_med": row["threshold_med"],
                })
            
            return profiles


# Global database instance
db = Database()
