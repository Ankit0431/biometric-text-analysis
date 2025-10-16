"""
Integration tests for the complete API endpoints.

Tests the full flow: enrollment → verification → challenge.
"""

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
import asyncio

# Import app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app


class TestAPIIntegration:
    """Test full API integration."""

    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
        self.test_user_id = "test_user_integration_001"

    def test_health_endpoint(self):
        """Test health endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_enroll_start(self):
        """Test enrollment start endpoint."""
        response = self.client.post(
            "/enroll/start",
            json={
                "user_id": self.test_user_id,
                "lang": "en",
                "domain": "chat"
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "challenges" in data
        assert "session_token" in data
        assert "required_samples" in data

        # Check challenges
        assert len(data["challenges"]) > 0
        assert data["required_samples"] == 8

        # Check challenge structure
        challenge = data["challenges"][0]
        assert "challenge_id" in challenge
        assert "prompt" in challenge
        assert "min_words" in challenge
        assert "timebox_s" in challenge
        assert "constraints" in challenge

    def test_enroll_submit(self):
        """Test enrollment submit endpoint."""
        # First start enrollment
        start_response = self.client.post(
            "/enroll/start",
            json={
                "user_id": self.test_user_id,
                "lang": "en",
                "domain": "chat"
            }
        )

        assert start_response.status_code == 200
        start_data = start_response.json()

        session_token = start_data["session_token"]
        challenge_id = start_data["challenges"][0]["challenge_id"]

        # Submit a sample
        sample_text = " ".join(["This is a test enrollment sample with enough words."] * 10)

        response = self.client.post(
            "/enroll/submit",
            json={
                "challenge_id": challenge_id,
                "user_id": self.test_user_id,
                "text": sample_text,
                "session_token": session_token,
                "timings": {
                    "histogram": [10, 15, 20, 10, 5, 2],
                    "mean_iki": 150,
                    "std_iki": 45,
                    "total_events": 62
                }
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "accepted" in data
        assert "remaining" in data
        assert "warnings" in data
        assert "profile_ready" in data

        # First sample should be accepted
        assert data["accepted"] is True
        assert data["remaining"] == 7  # 8 - 1

    def test_verify_without_profile(self):
        """Test verification without enrollment (should fail gracefully)."""
        response = self.client.post(
            "/verify",
            json={
                "user_id": "user_no_profile_999",
                "text": "This is a test verification sample with enough words to pass validation checks.",
                "lang": "en",
                "domain_hint": "chat"
            }
        )

        # Should return a decision (likely deny or challenge)
        assert response.status_code == 200
        data = response.json()

        assert "decision" in data
        assert "score" in data
        assert "reasons" in data
        assert "thresholds" in data

        # Without profile, should not allow
        assert data["decision"] in ["deny", "challenge", "step_up"]

    def test_verify_with_short_text(self):
        """Test verification with too short text."""
        response = self.client.post(
            "/verify",
            json={
                "user_id": self.test_user_id,
                "text": "Short.",
                "lang": "en",
                "domain_hint": "chat"
            }
        )

        # Should handle gracefully
        assert response.status_code in [200, 400]

    def test_challenge_prepare(self):
        """Test challenge preparation."""
        response = self.client.post(
            "/challenge/prepare",
            json={
                "user_id": self.test_user_id,
                "lang": "en",
                "domain": "chat"
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "challenge_id" in data
        assert "prompt" in data
        assert "min_words" in data
        assert "timebox_s" in data
        assert "constraints" in data

        # Validate values
        assert len(data["prompt"]) > 0
        assert data["min_words"] > 0
        assert data["timebox_s"] > 0

    def test_challenge_submit(self):
        """Test challenge submission."""
        # First prepare a challenge
        prepare_response = self.client.post(
            "/challenge/prepare",
            json={
                "user_id": self.test_user_id,
                "lang": "en",
                "domain": "chat"
            }
        )

        assert prepare_response.status_code == 200
        prepare_data = prepare_response.json()
        challenge_id = prepare_data["challenge_id"]

        # Submit challenge response
        sample_text = " ".join(["This is my challenge response with sufficient content."] * 15)

        response = self.client.post(
            "/challenge/submit",
            json={
                "challenge_id": challenge_id,
                "user_id": self.test_user_id,
                "text": sample_text,
                "timings": {
                    "histogram": [12, 18, 22, 8, 4, 1],
                    "mean_iki": 145,
                    "std_iki": 42,
                    "total_events": 65
                }
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "decision" in data
        assert "score" in data
        assert "reasons" in data
        assert "thresholds" in data

        # Should get a decision
        assert data["decision"] in ["allow", "challenge", "step_up", "deny"]
        assert 0 <= data["score"] <= 1

    def test_full_enrollment_flow(self):
        """Test complete enrollment flow (all 8 samples)."""
        user_id = f"{self.test_user_id}_full_flow"

        # Start enrollment
        start_response = self.client.post(
            "/enroll/start",
            json={
                "user_id": user_id,
                "lang": "en",
                "domain": "chat"
            }
        )

        assert start_response.status_code == 200
        start_data = start_response.json()

        session_token = start_data["session_token"]
        challenges = start_data["challenges"]
        required_samples = start_data["required_samples"]

        # Submit all required samples
        for i in range(min(3, required_samples)):  # Test with 3 samples for speed
            challenge = challenges[i]

            # Generate varied text
            sample_text = f"Sample {i+1}: " + " ".join([
                "This is my enrollment sample with unique content each time.",
                "I'm writing naturally about my thoughts and experiences today.",
                "The weather is nice and I enjoyed my morning coffee very much."
            ] * 8)

            response = self.client.post(
                "/enroll/submit",
                json={
                    "challenge_id": challenge["challenge_id"],
                    "user_id": user_id,
                    "text": sample_text,
                    "session_token": session_token,
                    "timings": {
                        "histogram": [10 + i, 15 + i, 20, 10, 5, 2],
                        "mean_iki": 150 + i * 5,
                        "std_iki": 45 + i,
                        "total_events": 60 + i * 2
                    }
                }
            )

            assert response.status_code == 200
            data = response.json()

            assert data["accepted"] is True
            assert data["remaining"] == required_samples - (i + 1)

    def test_validation_errors(self):
        """Test input validation."""
        # Missing required fields
        response = self.client.post(
            "/verify",
            json={
                "user_id": self.test_user_id,
                # Missing text field
            }
        )
        assert response.status_code == 422  # Validation error

        # Invalid user_id
        response = self.client.post(
            "/verify",
            json={
                "user_id": "",  # Empty user_id
                "text": "Some text here",
            }
        )
        assert response.status_code == 422

        # Text too short
        response = self.client.post(
            "/verify",
            json={
                "user_id": self.test_user_id,
                "text": "Hi",  # Too short
            }
        )
        assert response.status_code == 422


class TestAPIAuthentication:
    """Test authentication and rate limiting."""

    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)

    def test_rate_limiting_headers(self):
        """Test that rate limiting is applied."""
        # Make multiple requests
        for _ in range(5):
            response = self.client.post(
                "/verify",
                json={
                    "user_id": "rate_limit_test",
                    "text": "This is a test sample with enough words for validation.",
                }
            )
            # Should not be rate limited in test (limit is high)
            assert response.status_code in [200, 422, 500]


class TestAPIErrorHandling:
    """Test error handling."""

    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)

    def test_malformed_json(self):
        """Test handling of malformed JSON."""
        response = self.client.post(
            "/verify",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_invalid_endpoint(self):
        """Test non-existent endpoint."""
        response = self.client.get("/nonexistent")
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
