"""
Adaptive Profile Update System

This module implements continual learning for user profiles using:
- EMA (Exponential Moving Average) updates for centroids
- Adaptive threshold adjustment based on recent performance
- Asynchronous profile persistence
- Rolling window statistics for threshold calibration
"""
import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
import numpy as np
import statistics

logger = logging.getLogger(__name__)


@dataclass
class VerificationEvent:
    """Record of a verification attempt."""
    timestamp: datetime
    score: float
    decision: str
    text_length: int
    success: bool


class AdaptiveProfileManager:
    """Manages adaptive profile updates with EMA and threshold adjustment."""
    
    def __init__(
        self,
        semantic_alpha: float = 0.9,
        stylometry_alpha: float = 0.8,
        threshold_alpha: float = 0.95,
        window_size: int = 50,
        min_updates_for_adaptation: int = 10
    ):
        """
        Initialize adaptive profile manager.
        
        Args:
            semantic_alpha: EMA factor for semantic embeddings (higher = more stability)
            stylometry_alpha: EMA factor for stylometry features (higher = more stability)
            threshold_alpha: EMA factor for threshold adaptation (higher = more stability)
            window_size: Number of recent events to consider for adaptation
            min_updates_for_adaptation: Minimum successful verifications before adaptation
        """
        self.semantic_alpha = semantic_alpha
        self.stylometry_alpha = stylometry_alpha
        self.threshold_alpha = threshold_alpha
        self.window_size = window_size
        self.min_updates_for_adaptation = min_updates_for_adaptation
        
        # Track recent verification events per user
        self.user_events: Dict[str, deque] = {}
        
        # Queue for async profile updates
        self.update_queue = asyncio.Queue()
        self.update_task = None
        
    def update_profile_vector(
        self,
        old_vec: np.ndarray,
        new_vec: np.ndarray,
        alpha: float = 0.9
    ) -> np.ndarray:
        """
        Update profile vector using EMA (Exponential Moving Average).
        
        Args:
            old_vec: Previous profile vector
            new_vec: New observation vector
            alpha: EMA factor (0.0 = ignore old, 1.0 = ignore new)
            
        Returns:
            Updated profile vector
        """
        if old_vec is None or len(old_vec) == 0:
            return new_vec.copy()
            
        if new_vec is None or len(new_vec) == 0:
            return old_vec.copy()
            
        # Ensure vectors have same shape
        if old_vec.shape != new_vec.shape:
            logger.warning(f"Vector shape mismatch: {old_vec.shape} vs {new_vec.shape}")
            return new_vec.copy()
        
        # EMA update: new = alpha * old + (1 - alpha) * new
        updated_vec = alpha * old_vec + (1 - alpha) * new_vec
        
        # Maintain L2 normalization for embeddings
        norm = np.linalg.norm(updated_vec)
        if norm > 0:
            updated_vec = updated_vec / norm
            
        return updated_vec
    
    def update_stylometry_stats(
        self,
        old_stats: Dict[str, Any],
        new_stats: Dict[str, Any],
        alpha: float = 0.8
    ) -> Dict[str, Any]:
        """
        Update stylometry statistics using EMA.
        
        Args:
            old_stats: Previous stylometry statistics
            new_stats: New stylometry observation
            alpha: EMA factor for stylometry features
            
        Returns:
            Updated stylometry statistics
        """
        if not old_stats:
            return new_stats.copy()
            
        if not new_stats:
            return old_stats.copy()
            
        updated_stats = {}
        
        # Update each numeric statistic
        for key, new_value in new_stats.items():
            if isinstance(new_value, (int, float)):
                old_value = old_stats.get(key, new_value)
                updated_stats[key] = alpha * old_value + (1 - alpha) * new_value
            else:
                # Non-numeric values (strings, etc.) - use new value
                updated_stats[key] = new_value
                
        # Include any old stats not present in new stats
        for key, old_value in old_stats.items():
            if key not in updated_stats:
                updated_stats[key] = old_value
                
        return updated_stats
    
    def record_verification_event(
        self,
        user_id: str,
        score: float,
        decision: str,
        text_length: int
    ):
        """
        Record a verification event for adaptive learning.
        
        Args:
            user_id: User identifier
            score: Verification score
            decision: Verification decision (allow/deny/challenge)
            text_length: Length of input text
        """
        if user_id not in self.user_events:
            self.user_events[user_id] = deque(maxlen=self.window_size)
            
        event = VerificationEvent(
            timestamp=datetime.now(),
            score=score,
            decision=decision,
            text_length=text_length,
            success=(decision == "allow")
        )
        
        self.user_events[user_id].append(event)
        
        logger.debug(f"Recorded verification event for {user_id}: score={score:.3f}, decision={decision}")
    
    def calculate_adaptive_thresholds(
        self,
        user_id: str,
        current_high: float = 0.84,
        current_med: float = 0.72
    ) -> Tuple[float, float]:
        """
        Calculate adaptive thresholds based on recent verification history.
        
        Args:
            user_id: User identifier
            current_high: Current high threshold
            current_med: Current medium threshold
            
        Returns:
            Tuple of (new_high_threshold, new_med_threshold)
        """
        if user_id not in self.user_events:
            return current_high, current_med
            
        events = list(self.user_events[user_id])
        if len(events) < self.min_updates_for_adaptation:
            return current_high, current_med
            
        # Analyze recent performance
        recent_events = [e for e in events if e.timestamp > datetime.now() - timedelta(days=7)]
        if not recent_events:
            return current_high, current_med
            
        # Calculate success rate and score statistics
        success_rate = sum(1 for e in recent_events if e.success) / len(recent_events)
        successful_scores = [e.score for e in recent_events if e.success]
        failed_scores = [e.score for e in recent_events if not e.success]
        
        logger.info(f"Adaptive threshold analysis for {user_id}: "
                   f"success_rate={success_rate:.2f}, "
                   f"successful_scores={len(successful_scores)}, "
                   f"failed_scores={len(failed_scores)}")
        
        new_high = current_high
        new_med = current_med
        
        if successful_scores:
            # Calculate percentiles of successful scores
            successful_mean = statistics.mean(successful_scores)
            successful_median = statistics.median(successful_scores)
            successful_min = min(successful_scores)
            
            # If success rate is high and scores are consistently good, we can be more lenient
            if success_rate > 0.8 and successful_mean > current_high:
                # Gradually lower thresholds (be more lenient)
                adjustment = min(0.02, (successful_median - current_high) * 0.1)
                new_high = current_high - adjustment * (1 - self.threshold_alpha)
                new_med = current_med - adjustment * (1 - self.threshold_alpha)
                
            # If success rate is low, we should be more strict
            elif success_rate < 0.6:
                # Gradually raise thresholds (be more strict)
                if failed_scores:
                    failed_max = max(failed_scores)
                    if failed_max > current_med:
                        adjustment = min(0.02, (failed_max - current_med) * 0.1)
                        new_high = current_high + adjustment * (1 - self.threshold_alpha)
                        new_med = current_med + adjustment * (1 - self.threshold_alpha)
        
        # Apply EMA smoothing to threshold changes
        new_high = self.threshold_alpha * current_high + (1 - self.threshold_alpha) * new_high
        new_med = self.threshold_alpha * current_med + (1 - self.threshold_alpha) * new_med
        
        # Ensure reasonable bounds
        new_high = max(0.7, min(0.95, new_high))
        new_med = max(0.6, min(0.85, new_med))
        
        # Ensure med < high
        if new_med >= new_high:
            new_med = new_high - 0.05
            
        logger.info(f"Adaptive thresholds for {user_id}: "
                   f"high {current_high:.3f} -> {new_high:.3f}, "
                   f"med {current_med:.3f} -> {new_med:.3f}")
        
        return new_high, new_med
    
    def should_update_profile(
        self,
        user_id: str,
        score: float,
        decision: str,
        min_score_threshold: float = 0.8
    ) -> bool:
        """
        Determine if profile should be updated based on verification result.
        
        Args:
            user_id: User identifier
            score: Verification score
            decision: Verification decision
            min_score_threshold: Minimum score required for profile update
            
        Returns:
            True if profile should be updated
        """
        # Only update on successful high-confidence verifications
        if decision != "allow":
            return False
            
        if score < min_score_threshold:
            return False
            
        # Check if user has too many recent updates (avoid over-fitting)
        if user_id in self.user_events:
            recent_events = [
                e for e in self.user_events[user_id]
                if e.timestamp > datetime.now() - timedelta(hours=1)
                and e.success
            ]
            if len(recent_events) > 5:  # Max 5 profile updates per hour
                logger.debug(f"Skipping profile update for {user_id} - too many recent updates")
                return False
        
        return True
    
    async def queue_profile_update(
        self,
        user_id: str,
        lang: str,
        domain: str,
        current_profile: Dict[str, Any],
        new_embedding: np.ndarray,
        new_style_features: np.ndarray,
        new_style_stats: Dict[str, Any],
        verification_score: float,
        decision: str
    ):
        """
        Queue a profile update for asynchronous processing.
        
        Args:
            user_id: User identifier
            lang: Language code
            domain: Domain
            current_profile: Current user profile
            new_embedding: New semantic embedding
            new_style_features: New stylometry features
            new_style_stats: New stylometry statistics
            verification_score: Verification score that triggered update
            decision: Verification decision
        """
        update_data = {
            'user_id': user_id,
            'lang': lang,
            'domain': domain,
            'current_profile': current_profile,
            'new_embedding': new_embedding,
            'new_style_features': new_style_features,
            'new_style_stats': new_style_stats,
            'verification_score': verification_score,
            'decision': decision,
            'timestamp': datetime.now()
        }
        
        await self.update_queue.put(update_data)
        logger.debug(f"Queued profile update for {user_id}")
    
    def compute_updated_profile(
        self,
        current_profile: Dict[str, Any],
        new_embedding: np.ndarray,
        new_style_features: np.ndarray,
        new_style_stats: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Compute updated profile using EMA.
        
        Args:
            current_profile: Current user profile
            new_embedding: New semantic embedding
            new_style_features: New stylometry features
            new_style_stats: New stylometry statistics
            user_id: User identifier for threshold adaptation
            
        Returns:
            Updated profile dictionary
        """
        updated_profile = current_profile.copy()
        
        # Update semantic centroid
        if 'centroid' in current_profile and current_profile['centroid'] is not None:
            old_centroid = np.array(current_profile['centroid'])
            new_centroid = self.update_profile_vector(
                old_centroid, 
                new_embedding, 
                alpha=self.semantic_alpha
            )
            updated_profile['centroid'] = new_centroid.tolist()
        else:
            updated_profile['centroid'] = new_embedding.tolist()
        
        # Update stylometry statistics
        if 'stylometry_stats' in current_profile and current_profile['stylometry_stats']:
            old_stylo_stats = current_profile['stylometry_stats']
            updated_stylo_stats = self.update_stylometry_stats(
                old_stylo_stats,
                new_style_stats,
                alpha=self.stylometry_alpha
            )
            updated_profile['stylometry_stats'] = updated_stylo_stats
        else:
            updated_profile['stylometry_stats'] = new_style_stats
        
        # Update sample count
        updated_profile['n_samples'] = current_profile.get('n_samples', 0) + 1
        
        # Calculate adaptive thresholds
        current_high = current_profile.get('threshold_high', 0.84)
        current_med = current_profile.get('threshold_med', 0.72)
        
        new_high, new_med = self.calculate_adaptive_thresholds(
            user_id, current_high, current_med
        )
        
        updated_profile['threshold_high'] = new_high
        updated_profile['threshold_med'] = new_med
        
        logger.info(f"Profile update computed for {user_id}: "
                   f"samples {current_profile.get('n_samples', 0)} -> {updated_profile['n_samples']}, "
                   f"thresholds ({current_high:.3f}, {current_med:.3f}) -> ({new_high:.3f}, {new_med:.3f})")
        
        return updated_profile
    
    async def start_update_worker(self, db):
        """Start the background worker for processing profile updates."""
        self.update_task = asyncio.create_task(self._profile_update_worker(db))
        logger.info("Started adaptive profile update worker")
    
    async def stop_update_worker(self):
        """Stop the background worker."""
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped adaptive profile update worker")
    
    async def _profile_update_worker(self, db):
        """Background worker that processes queued profile updates."""
        while True:
            try:
                # Wait for update request
                update_data = await self.update_queue.get()
                
                # Process the update
                await self._process_profile_update(db, update_data)
                
                # Mark task as done
                self.update_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in profile update worker: {e}")
                # Continue processing other updates
                continue
    
    async def _process_profile_update(self, db, update_data: Dict[str, Any]):
        """Process a single profile update."""
        try:
            user_id = update_data['user_id']
            lang = update_data['lang']
            domain = update_data['domain']
            
            # Compute updated profile
            updated_profile = self.compute_updated_profile(
                current_profile=update_data['current_profile'],
                new_embedding=update_data['new_embedding'],
                new_style_features=update_data['new_style_features'],
                new_style_stats=update_data['new_style_stats'],
                user_id=user_id
            )
            
            # Save to database
            await db.save_profile(
                user_id=user_id,
                lang=lang,
                domain=domain,
                centroid=np.array(updated_profile['centroid']),
                cov_diag=updated_profile.get('cov_diag'),
                n_samples=updated_profile['n_samples'],
                stylometry_stats=updated_profile['stylometry_stats'],
                threshold_high=updated_profile['threshold_high'],
                threshold_med=updated_profile['threshold_med'],
                prompt_answers=updated_profile.get('prompt_answers')
            )
            
            logger.info(f"Successfully updated profile for {user_id} in {lang}/{domain}")
            
        except Exception as e:
            logger.error(f"Failed to process profile update: {e}")
            raise


# Global instance
adaptive_manager = AdaptiveProfileManager()


def update_profile_vector(old_vec: np.ndarray, new_vec: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """
    Convenience function for EMA profile vector updates.
    
    Args:
        old_vec: Previous profile vector
        new_vec: New observation vector  
        alpha: EMA factor (higher = more stable, lower = more adaptive)
        
    Returns:
        Updated profile vector
    """
    return adaptive_manager.update_profile_vector(old_vec, new_vec, alpha)


async def initialize_adaptive_system(db):
    """Initialize the adaptive profile management system."""
    await adaptive_manager.start_update_worker(db)
    logger.info("Adaptive profile management system initialized")


async def shutdown_adaptive_system():
    """Shutdown the adaptive profile management system."""
    await adaptive_manager.stop_update_worker()
    logger.info("Adaptive profile management system shutdown")