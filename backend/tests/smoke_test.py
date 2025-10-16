"""
Quick smoke test for API endpoints.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("✅ Health endpoint works")


def test_enroll_start():
    """Test enroll/start endpoint."""
    response = client.post(
        "/enroll/start",
        json={
            "user_id": "test_user_001",
            "lang": "en",
            "domain": "chat"
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Enroll/start works - got {len(data['challenges'])} challenges")
    else:
        print(f"❌ Enroll/start failed: {response.text}")


def test_verify():
    """Test verify endpoint."""
    response = client.post(
        "/verify",
        json={
            "user_id": "test_user_002",
            "text": "This is a test verification sample with enough words to pass validation checks and testing.",
            "lang": "en",
            "domain_hint": "chat"
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Verify works - decision: {data['decision']}, score: {data['score']}")
    else:
        print(f"❌ Verify failed: {response.text}")


if __name__ == "__main__":
    print("=" * 60)
    print("API Smoke Tests")
    print("=" * 60)

    try:
        test_health()
        test_enroll_start()
        test_verify()

        print("\n" + "=" * 60)
        print("✅ All smoke tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
