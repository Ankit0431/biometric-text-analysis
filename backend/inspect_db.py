"""
Database inspection script to debug the tuple issue.
"""
import asyncio
from db import db
import json

async def inspect_database():
    """Inspect database data to understand the format issue."""
    print("Inspecting database data...")
    
    try:
        # Connect to database
        await db.connect()
        print("✅ Connected to database")
        
        # Get all profiles
        profiles = await db.get_all_enrolled_profiles("en", "chat")
        print(f"Found {len(profiles)} enrolled profiles")
        
        for i, profile in enumerate(profiles):
            print(f"\n--- Profile {i+1} ---")
            print(f"User ID: {profile['user_id']}")
            print(f"Username: {profile.get('username', 'N/A')}")
            print(f"N samples: {profile['n_samples']}")
            
            # Check centroid
            centroid = profile.get('centroid')
            print(f"Centroid type: {type(centroid)}")
            if hasattr(centroid, 'shape'):
                print(f"Centroid shape: {centroid.shape}")
            else:
                print(f"Centroid value: {centroid}")
            
            # Check style features
            style_mean = profile.get('style_mean')
            style_std = profile.get('style_std')
            print(f"Style mean type: {type(style_mean)}")
            print(f"Style std type: {type(style_std)}")
            
            if hasattr(style_mean, 'shape'):
                print(f"Style mean shape: {style_mean.shape}")
            else:
                print(f"Style mean value (first 5): {style_mean[:5] if style_mean is not None else None}")
            
            # Check stylometry_stats
            stylo_stats = profile.get('stylometry_stats', {})
            print(f"Stylometry stats keys: {list(stylo_stats.keys()) if isinstance(stylo_stats, dict) else 'Not a dict'}")
            
            if isinstance(stylo_stats, dict):
                for key, value in stylo_stats.items():
                    print(f"  {key}: {type(value)} - {value[:5] if isinstance(value, list) and len(value) > 5 else value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Disconnect from database
        await db.disconnect()
        print("✅ Disconnected from database")

if __name__ == "__main__":
    asyncio.run(inspect_database())