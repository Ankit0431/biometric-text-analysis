"""
Test script for user identification functionality.
"""
import asyncio
from identify_user import user_identifier
from db import db

async def test_identify_user():
    """Test the user identification functionality."""
    print("Testing user identification...")
    
    try:
        # Connect to database
        await db.connect()
        print("✅ Connected to database")
        
        # Test text
        test_text = "This is a test message to see if the user identification system works properly. I am writing this to check how well the biometric system can identify users based on their writing patterns."
        
        # Run identification
        result = await user_identifier.identify_user(test_text)
        
        print(f"✅ Identification completed")
        print(f"Identified user: {result.get('identified_user')}")
        print(f"Username: {result.get('username')}")
        print(f"Confidence: {result.get('confidence_score'):.4f}")
        print(f"Message: {result.get('message')}")
        print(f"All scores: {result.get('all_scores')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Disconnect from database
        await db.disconnect()
        print("✅ Disconnected from database")

if __name__ == "__main__":
    asyncio.run(test_identify_user())