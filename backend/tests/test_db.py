"""
Integration tests for database layer.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pytest_asyncio
import asyncio
import numpy as np
from db import Database


@pytest_asyncio.fixture
async def test_db():
    """Create a test database connection."""
    db = Database()
    await db.connect()
    yield db
    await db.disconnect()


@pytest.mark.asyncio
async def test_create_user(test_db):
    """Test creating a user."""
    await test_db.create_user(
        user_id="test_user_1",
        tenant_id="tenant_1",
        locale="en",
        consent_version="v1"
    )
    # Verify no exception raised


@pytest.mark.asyncio
async def test_upsert_and_get_profile(test_db):
    """Test upserting and retrieving a profile."""
    user_id = "test_user_profile"
    lang = "en"
    domain = "chat"

    # Create user first
    await test_db.create_user(user_id, "tenant_1")

    # Upsert profile
    centroid = np.random.randn(512).astype(np.float32)
    cov_diag = np.random.rand(512).tolist()
    stylometry_stats = {"mean_word_len": 4.5, "punct_rate": 0.02}

    await test_db.upsert_profile(
        user_id=user_id,
        lang=lang,
        domain=domain,
        centroid=centroid,
        cov_diag=cov_diag,
        n_samples=8,
        stylometry_stats=stylometry_stats,
        threshold_high=0.84,
        threshold_med=0.72,
    )

    # Retrieve profile
    profile = await test_db.get_profile(user_id, lang, domain)

    assert profile is not None
    assert profile["user_id"] == user_id
    assert profile["lang"] == lang
    assert profile["domain"] == domain
    assert profile["n_samples"] == 8
    assert abs(profile["threshold_high"] - 0.84) < 0.001
    assert abs(profile["threshold_med"] - 0.72) < 0.001
    assert isinstance(profile["centroid"], np.ndarray)
    assert profile["centroid"].shape == (512,)
    assert profile["stylometry_stats"] == stylometry_stats


@pytest.mark.asyncio
async def test_log_decision(test_db):
    """Test logging a decision."""
    decision_id = await test_db.log_decision(
        user_id="test_user_decision",
        kind="verify",
        lang="en",
        domain="chat",
        score=0.75,
        decision="allow",
        reasons=["HIGH_SCORE"],
        len_words=120,
        policy_version="v1",
    )

    assert isinstance(decision_id, int)
    assert decision_id > 0


@pytest.mark.asyncio
async def test_get_nonexistent_profile(test_db):
    """Test retrieving a profile that doesn't exist."""
    profile = await test_db.get_profile("nonexistent_user", "en", "chat")
    assert profile is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
