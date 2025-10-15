"""
Tests for Step 8: Verify & Challenge endpoints.

Tests:
- VerifyHandler: scoring with profile, handling short text, rejection reasons
- ChallengeHandler: challenge preparation, submission, validation
- Cache: profile caching, challenge session management
"""
import pytest
import numpy as np
from datetime import datetime, timedelta

from verify_handlers import VerifyHandler, ChallengeHandler
from encoder import TextEncoder
from policy import get_default_thresholds


# Fixtures
@pytest.fixture
def encoder():
    """Create text encoder instance."""
    return TextEncoder()


@pytest.fixture
def verify_handler(encoder):
    """Create verify handler instance."""
    return VerifyHandler(encoder)


@pytest.fixture
def challenge_handler(encoder):
    """Create challenge handler instance."""
    return ChallengeHandler(encoder)


@pytest.fixture
def mock_profile():
    """Create a mock user profile."""
    # Create realistic centroid
    centroid = np.random.randn(512).astype(np.float32)
    centroid = centroid / np.linalg.norm(centroid)  # L2 normalize

    # Create style features
    style_mean = np.random.randn(512).astype(np.float32) * 0.1
    style_std = np.ones(512, dtype=np.float32) * 0.1

    return {
        "user_id": "test_user",
        "lang": "en",
        "domain": "chat",
        "centroid": centroid,
        "cov_diag": [0.01] * 512,
        "n_samples": 8,
        "style_mean": style_mean,
        "style_std": style_std,
        "stylometry_stats": {
            "avg_sentence_len": 15.0,
            "avg_word_len": 4.5,
            "punct_rate": 0.05,
        },
        "threshold_high": 0.84,
        "threshold_med": 0.72,
    }


# Test VerifyHandler
class TestVerifyHandler:
    """Tests for VerifyHandler class."""

    @pytest.mark.asyncio
    async def test_verify_sample_valid(self, verify_handler, mock_profile):
        """Test verifying a valid sample."""
        text = "This is a test message with sufficient length to pass normalization. " * 5

        result = await verify_handler.verify_sample(
            user_id="test_user",
            text=text,
            profile=mock_profile,
        )

        assert "decision" in result
        assert result["decision"] in ["allow", "challenge", "step_up", "deny"]
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0
        assert "reasons" in result
        assert isinstance(result["reasons"], list)
        assert "thresholds" in result
        assert result["thresholds"]["high"] == 0.84
        assert result["thresholds"]["med"] == 0.72

    @pytest.mark.asyncio
    async def test_verify_sample_short_text(self, verify_handler, mock_profile):
        """Test that short text triggers challenge."""
        text = "Too short"

        result = await verify_handler.verify_sample(
            user_id="test_user",
            text=text,
            profile=mock_profile,
        )

        assert result["decision"] == "challenge"
        assert "SHORT_LEN" in result["reasons"]
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_verify_sample_with_timings(self, verify_handler, mock_profile):
        """Test verify with keystroke timing data."""
        text = "This is a message with timing data included in the verification process. " * 3
        timings = {
            "histogram": [5, 10, 15, 8, 3, 1],
            "mean_iki": 150.0,
            "std_iki": 50.0,
        }

        result = await verify_handler.verify_sample(
            user_id="test_user",
            text=text,
            profile=mock_profile,
            timings=timings,
        )

        assert "decision" in result
        assert "score" in result

    @pytest.mark.asyncio
    async def test_verify_sample_with_context(self, verify_handler, mock_profile):
        """Test verify with additional context."""
        text = "Testing context metadata in verification flow. " * 3
        context = {
            "ip_address": "192.168.1.1",
            "device_type": "desktop",
        }

        result = await verify_handler.verify_sample(
            user_id="test_user",
            text=text,
            profile=mock_profile,
            context=context,
        )

        assert "decision" in result


# Test ChallengeHandler
class TestChallengeHandler:
    """Tests for ChallengeHandler class."""

    def test_prepare_challenge(self, challenge_handler):
        """Test challenge preparation."""
        result = challenge_handler.prepare_challenge(
            user_id="test_user",
            lang="en",
            domain="chat",
        )

        assert "challenge_id" in result
        assert "prompt" in result
        assert "min_words" in result
        assert "timebox_s" in result
        assert "constraints" in result
        assert isinstance(result["constraints"], list)
        assert result["min_words"] >= 50

    def test_prepare_challenge_deterministic(self, challenge_handler):
        """Test that different challenges are generated."""
        result1 = challenge_handler.prepare_challenge(user_id="user1")
        result2 = challenge_handler.prepare_challenge(user_id="user2")

        # Challenge IDs should be different
        assert result1["challenge_id"] != result2["challenge_id"]

    @pytest.mark.asyncio
    async def test_submit_challenge_valid(self, challenge_handler, mock_profile):
        """Test submitting a valid challenge response."""
        # Prepare challenge
        challenge = challenge_handler.prepare_challenge(user_id="test_user")

        # Create challenge data
        challenge_data = {
            "user_id": "test_user",
            "internal_challenge_ref": challenge["internal_challenge_ref"],
            "created_at": datetime.utcnow().isoformat(),
        }

        # Submit response
        text = "This is my detailed response to the challenge prompt. " * 10
        result = await challenge_handler.submit_challenge(
            challenge_id=challenge["challenge_id"],
            user_id="test_user",
            text=text,
            profile=mock_profile,
            challenge_data=challenge_data,
        )

        assert "decision" in result
        assert result["decision"] in ["allow", "challenge", "step_up", "deny"]
        assert "score" in result

    @pytest.mark.asyncio
    async def test_submit_challenge_invalid_session(self, challenge_handler, mock_profile):
        """Test submitting with invalid challenge session."""
        result = await challenge_handler.submit_challenge(
            challenge_id="invalid_id",
            user_id="test_user",
            text="Some response text.",
            profile=mock_profile,
            challenge_data=None,  # No cached session
        )

        assert result["decision"] == "deny"
        assert "INVALID_CHALLENGE" in result["reasons"]

    @pytest.mark.asyncio
    async def test_submit_challenge_user_mismatch(self, challenge_handler, mock_profile):
        """Test submitting with mismatched user."""
        challenge_data = {
            "user_id": "different_user",
            "internal_challenge_ref": "ch_work_1",
            "created_at": datetime.utcnow().isoformat(),
        }

        result = await challenge_handler.submit_challenge(
            challenge_id="some_id",
            user_id="test_user",
            text="Response text.",
            profile=mock_profile,
            challenge_data=challenge_data,
        )

        assert result["decision"] == "deny"
        assert "USER_MISMATCH" in result["reasons"]

    @pytest.mark.asyncio
    async def test_submit_challenge_too_short(self, challenge_handler, mock_profile):
        """Test submitting response that's too short."""
        challenge_data = {
            "user_id": "test_user",
            "internal_challenge_ref": "ch_work_1",
            "created_at": datetime.utcnow().isoformat(),
        }

        # Submit short text
        text = "Too short response"
        result = await challenge_handler.submit_challenge(
            challenge_id="some_id",
            user_id="test_user",
            text=text,
            profile=mock_profile,
            challenge_data=challenge_data,
        )

        # Short text should trigger challenge, not deny
        assert result["decision"] == "challenge"
        assert "SHORT_LEN" in result["reasons"]


# Test Cache (mock-based since Redis may not be available)
class TestCacheLogic:
    """Tests for cache key generation and data serialization logic."""

    def test_profile_key_generation(self):
        """Test profile cache key format."""
        from cache import RedisCache
        cache = RedisCache()

        key = cache._profile_key("user123", "en", "chat")
        assert key == "profile:user123:en:chat"

    def test_challenge_key_generation(self):
        """Test challenge cache key format."""
        from cache import RedisCache
        cache = RedisCache()

        key = cache._challenge_key("challenge_abc")
        assert key == "challenge:challenge_abc"


# Integration test
class TestVerifyIntegration:
    """Integration tests for verify flow."""

    @pytest.mark.asyncio
    async def test_complete_verify_flow(self, verify_handler, mock_profile):
        """Test complete verify flow from text to decision."""
        # Sample text that should pass normalization
        text = """
        Hello, I wanted to share my thoughts on the recent project.
        The implementation went well, though we faced some challenges
        with the integration phase. Overall, I think we can be proud
        of what we accomplished together as a team.
        """

        result = await verify_handler.verify_sample(
            user_id="test_user",
            text=text,
            profile=mock_profile,
            lang="en",
            domain="chat",
        )

        # Verify result structure
        assert "decision" in result
        assert "score" in result
        assert "reasons" in result
        assert "thresholds" in result

        # Verify decision is valid
        assert result["decision"] in ["allow", "challenge", "step_up", "deny"]

        # Verify thresholds match profile
        assert result["thresholds"]["high"] == mock_profile["threshold_high"]
        assert result["thresholds"]["med"] == mock_profile["threshold_med"]

    @pytest.mark.asyncio
    async def test_challenge_flow_end_to_end(self, challenge_handler, mock_profile):
        """Test complete challenge flow from preparation to submission."""
        # Step 1: Prepare challenge
        challenge = challenge_handler.prepare_challenge(
            user_id="test_user",
            lang="en",
            domain="chat",
        )

        assert "challenge_id" in challenge
        assert "prompt" in challenge

        # Step 2: Simulate cache storage
        challenge_data = {
            "user_id": "test_user",
            "internal_challenge_ref": challenge["internal_challenge_ref"],
            "created_at": datetime.utcnow().isoformat(),
        }

        # Step 3: Submit response
        response_text = """
        In response to this challenge, I would like to share my perspective
        on the topic. Based on my experience, I believe that effective communication
        is key to success in any collaborative environment. We need to ensure
        that all team members are aligned on goals and expectations.
        """

        result = await challenge_handler.submit_challenge(
            challenge_id=challenge["challenge_id"],
            user_id="test_user",
            text=response_text,
            profile=mock_profile,
            challenge_data=challenge_data,
        )

        # Verify result
        assert "decision" in result
        assert result["decision"] in ["allow", "challenge", "step_up", "deny"]
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0
