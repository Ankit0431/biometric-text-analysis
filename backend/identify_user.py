"""
User identification module for 1:N matching.

This module implements functionality to identify users based on their
biometric text patterns by comparing input text against all enrolled
user profiles in the database.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from db import db
from encoder import get_encoder
from features import extract_features
from scoring import score_sample
from keystroke_features import extract_keystroke_features

logger = logging.getLogger(__name__)


class UserIdentifier:
    """Handles 1:N user identification based on biometric text patterns."""
    
    def __init__(self, confidence_threshold: float = 0.65):
        """
        Initialize user identifier.
        
        Args:
            confidence_threshold: Minimum confidence score to identify a user.
                                 Below this threshold, return 'unknown'.
        """
        self.confidence_threshold = confidence_threshold
        self.encoder = get_encoder()
    
    async def identify_user(
        self, 
        text: str, 
        timings: Optional[Dict[str, Any]] = None,
        lang: str = "en",
        domain: str = "chat"
    ) -> Dict[str, Any]:
        """
        Identify the most likely user based on input text and typing patterns.
        
        Args:
            text: Input text to analyze
            timings: Optional keystroke timing data
            lang: Language code (default: "en")
            domain: Domain (default: "chat")
        
        Returns:
            Dictionary containing:
            - identified_user: user_id of best match or None if unknown
            - username: username of best match or None
            - confidence_score: confidence score [0, 1]
            - all_scores: scores for all users (for debugging)
            - message: explanation of the result
        """
        try:
            # Get all enrolled profiles
            profiles = await db.get_all_enrolled_profiles(lang, domain)
            
            if not profiles:
                return {
                    "identified_user": None,
                    "username": None,
                    "confidence_score": 0.0,
                    "all_scores": {},
                    "message": "No enrolled users found in database"
                }
            
            # Extract features from input text
            embedding = self.encoder.encode(text)
            style_features_result = extract_features(text, lang)
            
            # Handle tuple return from extract_features
            if isinstance(style_features_result, tuple):
                style_features = style_features_result[0]  # Get the style vector
                style_stats = style_features_result[1]     # Get the style statistics
            else:
                style_features = style_features_result
                style_stats = None
            
            # Also handle embedding shape (flatten if needed)
            if len(embedding.shape) > 1:
                embedding = embedding.flatten()
            
            # Process keystroke timings if available
            processed_timings = None
            if timings:
                print(f"DEBUG: Processing timings: {timings}")
                # If it contains raw events, convert to histogram format
                if 'events' in timings:
                    processed_timings = self._convert_events_to_histogram(timings['events'])
                    print(f"DEBUG: Converted events to histogram: {processed_timings}")
                else:
                    # Timings are already in histogram format
                    processed_timings = timings
                    print(f"DEBUG: Using histogram data directly: {processed_timings}")
            else:
                print("DEBUG: No timing data provided")
            
            # Score against all profiles
            user_scores = {}
            for profile in profiles:
                try:
                    # Score this text against the current profile
                    result = score_sample(
                        user_profile=profile,
                        text=text,
                        embedding=embedding,
                        style_features=style_features,
                        timings=processed_timings
                    )
                    
                    user_scores[profile['user_id']] = {
                        'username': profile['username'],
                        'name': profile['name'],
                        'score': result['final_score'],
                        'components': result['components'],
                        'n_samples': profile['n_samples']
                    }
                    
                    logger.debug(f"User {profile['username']}: score={result['final_score']:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to score against user {profile['user_id']}: {e}")
                    user_scores[profile['user_id']] = {
                        'username': profile.get('username', 'unknown'),
                        'name': profile.get('name', 'unknown'),
                        'score': 0.0,
                        'error': str(e)
                    }
            
            # Find the best match
            if not user_scores:
                return {
                    "identified_user": None,
                    "username": None,
                    "confidence_score": 0.0,
                    "all_scores": {},
                    "message": "Failed to score against any enrolled users"
                }
            
            # Get the user with highest score
            best_user_id = max(user_scores.keys(), key=lambda uid: user_scores[uid]['score'])
            best_score = user_scores[best_user_id]['score']
            best_username = user_scores[best_user_id]['username']
            
            # Calculate confidence based on score and separation from second-best
            confidence_score = self._calculate_confidence(user_scores, best_user_id)
            
            # Determine if we're confident enough to identify the user
            if confidence_score >= self.confidence_threshold:
                message = f"Identified as {best_username} with {confidence_score:.2%} confidence"
                logger.info(f"User identified: {best_username} (score={best_score:.4f}, confidence={confidence_score:.4f})")
            else:
                best_user_id = None
                best_username = None
                message = f"Unknown user - highest match was {user_scores[max(user_scores.keys(), key=lambda uid: user_scores[uid]['score'])]['username']} with {confidence_score:.2%} confidence (below {self.confidence_threshold:.0%} threshold)"
                logger.info(f"User not identified: confidence {confidence_score:.4f} below threshold {self.confidence_threshold:.4f}")
            
            return {
                "identified_user": best_user_id,
                "username": best_username,
                "confidence_score": float(confidence_score),
                "all_scores": {uid: data['score'] for uid, data in user_scores.items()},
                "message": message
            }
            
        except Exception as e:
            logger.error(f"User identification failed: {e}")
            return {
                "identified_user": None,
                "username": None,
                "confidence_score": 0.0,
                "all_scores": {},
                "message": f"Identification failed: {str(e)}"
            }
    
    def _calculate_confidence(self, user_scores: Dict[str, Dict], best_user_id: str) -> float:
        """
        Calculate confidence score based on the best score and separation from others.
        
        Args:
            user_scores: Dictionary of user scores
            best_user_id: ID of the user with the highest score
        
        Returns:
            Confidence score [0, 1]
        """
        scores = [data['score'] for data in user_scores.values()]
        scores.sort(reverse=True)
        
        best_score = scores[0]
        
        # Base confidence is the score itself
        confidence = best_score
        
        # Boost confidence if there's good separation from second-best
        if len(scores) > 1:
            second_best = scores[1]
            separation = best_score - second_best
            
            # If separation is > 0.1, boost confidence
            if separation > 0.1:
                confidence = min(1.0, confidence + separation * 0.3)
            # If separation is < 0.05, reduce confidence (too close)
            elif separation < 0.05:
                confidence = confidence * 0.8
        
        # Penalize if best score is too low
        if best_score < 0.5:
            confidence = confidence * 0.7
        
        # Boost if score is very high
        elif best_score > 0.8:
            confidence = min(1.0, confidence * 1.1)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _convert_events_to_histogram(self, events: List[Dict]) -> Dict[str, Any]:
        """
        Convert raw keystroke events to histogram format.
        
        Args:
            events: List of event dictionaries with 'timestamp', 'type', 'key'
        
        Returns:
            Dictionary with histogram, mean_iki, std_iki, total_events
        """
        if not events:
            return {
                'histogram': [0, 0, 0, 0, 0, 0],
                'mean_iki': 0,
                'std_iki': 0,
                'total_events': 0
            }
        
        # Extract inter-key intervals (time between consecutive keydown events)
        ikis = []
        last_timestamp = None
        
        for event in events:
            if event.get('type') == 'keydown':
                timestamp = event.get('timestamp', 0)
                if last_timestamp is not None:
                    iki = timestamp - last_timestamp
                    if 0 < iki < 5000:  # Ignore pauses > 5 seconds
                        ikis.append(iki)
                last_timestamp = timestamp
        
        if not ikis:
            return {
                'histogram': [0, 0, 0, 0, 0, 0],
                'mean_iki': 0,
                'std_iki': 0,
                'total_events': 0
            }
        
        # Create histogram bins: [0-50ms, 50-100ms, 100-150ms, 150-200ms, 200-300ms, 300+ms]
        bins = [50, 100, 150, 200, 300, float('inf')]
        histogram = [0] * 6
        
        for iki in ikis:
            for i, bin_edge in enumerate(bins):
                if iki < bin_edge:
                    histogram[i] += 1
                    break
        
        # Calculate statistics
        mean_iki = np.mean(ikis) if ikis else 0
        std_iki = np.std(ikis) if ikis else 0
        
        return {
            'histogram': histogram,
            'mean_iki': float(mean_iki),
            'std_iki': float(std_iki),
            'total_events': len(ikis)
        }


# Global identifier instance
user_identifier = UserIdentifier()