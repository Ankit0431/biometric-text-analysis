"""
Authentication handler for login/signup with password hashing.
"""
import hashlib
import uuid
from typing import Optional, Dict, Any
from db import db


def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(password) == password_hash


async def signup_user(name: str, username: str, password: str) -> Dict[str, Any]:
    """
    Create a new user account.
    
    Args:
        name: User's full name
        username: Unique username
        password: Plain text password (will be hashed)
    
    Returns:
        Dictionary with user_id and success status
    
    Raises:
        ValueError: If username already exists
    """
    # Check if username already exists
    existing_user = await db.get_user_by_username(username)
    if existing_user:
        raise ValueError("Username already exists")
    
    # Generate user_id
    user_id = str(uuid.uuid4())
    
    # Hash password
    password_hash = hash_password(password)
    
    # Create user in database
    await db.create_user_with_auth(
        user_id=user_id,
        username=username,
        password_hash=password_hash,
        name=name,
        tenant_id="default",
        locale="en"
    )
    
    return {
        "user_id": user_id,
        "username": username,
        "name": name,
        "success": True
    }


async def login_user(username: str, password: str) -> Dict[str, Any]:
    """
    Authenticate a user with username and password.
    
    Args:
        username: Username
        password: Plain text password
    
    Returns:
        Dictionary with user info if successful
    
    Raises:
        ValueError: If credentials are invalid
    """
    # Get user from database
    user = await db.get_user_by_username(username)
    if not user:
        raise ValueError("Invalid username or password")
    
    # Verify password
    if not verify_password(password, user["password_hash"]):
        raise ValueError("Invalid username or password")
    
    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "name": user["name"],
        "biometric_enrolled": user["biometric_enrolled"],
        "success": True
    }
