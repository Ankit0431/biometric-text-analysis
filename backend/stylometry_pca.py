#!/usr/bin/env python3
"""
Enhanced Stylometry Pipeline with PCA-based Dimensionality Reduction

This module provides an enhanced stylometry pipeline that includes:
1. StandardScaler for feature normalization
2. PCA with whitening for dimensionality reduction and decorrelation
3. Improved similarity computation using transformed features

The pipeline follows this flow:
1. Extract raw stylometry features
2. Apply StandardScaler normalization
3. Apply PCA transformation with whitening
4. Compute similarities in the PCA space
"""

import os
import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from features import extract_features

logger = logging.getLogger(__name__)


class EnhancedStylometryPipeline:
    """Enhanced stylometry pipeline with PCA-based dimensionality reduction."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the enhanced stylometry pipeline.
        
        Args:
            models_dir: Directory to save/load PCA models
        """
        self.models_dir = models_dir
        self.pca_model_path = os.path.join(models_dir, "pca_stylometry.pkl")
        self.scaler = None
        self.pca = None
        self.is_fitted = False
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Try to load existing models
        self._load_models()
    
    def _load_models(self) -> bool:
        """
        Load existing PCA models if available.
        
        Returns:
            True if models were loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.pca_model_path):
                self.scaler, self.pca = joblib.load(self.pca_model_path)
                self.is_fitted = True
                logger.info(f"Loaded PCA stylometry models from {self.pca_model_path}")
                logger.info(f"PCA components: {self.pca.n_components_}, explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load PCA models: {e}")
            self.scaler = None
            self.pca = None
            self.is_fitted = False
        
        return False
    
    def fit(self, style_vectors: np.ndarray, n_components: float = 0.95, random_state: int = 42) -> None:
        """
        Fit the PCA pipeline on a collection of style vectors.
        
        Args:
            style_vectors: Array of shape (n_samples, n_features) containing style vectors
            n_components: Number of components to keep (if float, keep components explaining this variance)
            random_state: Random state for reproducibility
        """
        logger.info(f"Fitting PCA pipeline on {style_vectors.shape[0]} samples with {style_vectors.shape[1]} features")
        
        # Initialize and fit StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(style_vectors)
        
        # Initialize and fit PCA with whitening
        self.pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
        X_pca = self.pca.fit_transform(X_scaled)
        
        self.is_fitted = True
        
        # Save the fitted models
        self._save_models()
        
        logger.info(f"PCA pipeline fitted successfully:")
        logger.info(f"  Components retained: {self.pca.n_components_}")
        logger.info(f"  Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        logger.info(f"  Original dimensions: {style_vectors.shape[1]} -> Reduced dimensions: {X_pca.shape[1]}")
    
    def _save_models(self) -> None:
        """Save the fitted PCA models to disk."""
        try:
            joblib.dump((self.scaler, self.pca), self.pca_model_path)
            logger.info(f"Saved PCA stylometry models to {self.pca_model_path}")
        except Exception as e:
            logger.error(f"Failed to save PCA models: {e}")
    
    def transform(self, style_vector: np.ndarray) -> np.ndarray:
        """
        Transform a style vector using the fitted PCA pipeline.
        
        Args:
            style_vector: Input style vector of shape (n_features,)
            
        Returns:
            Transformed vector in PCA space
            
        Raises:
            ValueError: If pipeline is not fitted
        """
        if not self.is_fitted:
            raise ValueError("PCA pipeline is not fitted. Call fit() first or ensure models are loaded.")
        
        # Ensure 2D shape for sklearn
        if len(style_vector.shape) == 1:
            style_vector = style_vector.reshape(1, -1)
        
        # Apply scaling and PCA transformation
        X_scaled = self.scaler.transform(style_vector)
        X_pca = self.pca.transform(X_scaled)
        
        return X_pca.flatten() if X_pca.shape[0] == 1 else X_pca
    
    def extract_and_transform_features(self, text: str, tokens: list, lang: str = "en") -> Tuple[np.ndarray, Dict]:
        """
        Extract stylometry features and transform them using PCA pipeline.
        
        Args:
            text: Input text
            tokens: List of word tokens
            lang: Language code
            
        Returns:
            Tuple of (transformed_features, stats_dict)
        """
        # Extract raw stylometry features
        raw_features, stats = extract_features(text, tokens, lang)
        
        # Transform using PCA pipeline if fitted
        if self.is_fitted:
            transformed_features = self.transform(raw_features)
        else:
            logger.warning("PCA pipeline not fitted, returning raw features")
            transformed_features = raw_features
        
        return transformed_features, stats
    
    def compute_enhanced_similarity(
        self,
        style_vector: np.ndarray,
        profile_style_mean: np.ndarray,
        profile_style_std: Optional[np.ndarray] = None,
        use_cosine: bool = True,
        use_euclidean: bool = True
    ) -> float:
        """
        Compute enhanced stylometry similarity in PCA space.
        
        Args:
            style_vector: Current sample's transformed style features
            profile_style_mean: Mean of user's transformed style features
            profile_style_std: Std dev of user's transformed style features (optional)
            use_cosine: Whether to include cosine similarity
            use_euclidean: Whether to include Euclidean distance
            
        Returns:
            Enhanced similarity score in [0, 1]
        """
        # Ensure vectors are 1D
        if len(style_vector.shape) > 1:
            style_vector = style_vector.flatten()
        if len(profile_style_mean.shape) > 1:
            profile_style_mean = profile_style_mean.flatten()
        
        similarities = []
        
        if use_cosine:
            # Cosine similarity (works well in PCA space)
            cos_sim = cosine_similarity(
                style_vector.reshape(1, -1),
                profile_style_mean.reshape(1, -1)
            )[0, 0]
            cos_sim_normalized = (cos_sim + 1.0) / 2.0  # Map from [-1,1] to [0,1]
            similarities.append(cos_sim_normalized)
        
        if use_euclidean:
            # Normalized Euclidean distance in PCA space
            # PCA whitening makes this more reliable
            euclidean_dist = np.linalg.norm(style_vector - profile_style_mean)
            
            # Normalize by expected distance (heuristic for PCA space)
            # In whitened PCA space, distances tend to be in a more predictable range
            expected_distance = np.sqrt(len(style_vector))  # Rough estimate
            normalized_dist = euclidean_dist / expected_distance
            
            # Convert distance to similarity using sigmoid
            dist_similarity = 1.0 / (1.0 + np.exp(normalized_dist - 1.0))
            similarities.append(dist_similarity)
        
        # Combine similarities (equal weighting)
        if similarities:
            combined_similarity = np.mean(similarities)
        else:
            logger.warning("No similarity metrics enabled, returning 0.5")
            combined_similarity = 0.5
        
        return float(np.clip(combined_similarity, 0.0, 1.0))


# Global pipeline instance
_pipeline = None


def get_enhanced_pipeline(models_dir: str = "models") -> EnhancedStylometryPipeline:
    """Get or create global enhanced stylometry pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = EnhancedStylometryPipeline(models_dir)
    return _pipeline


def train_pca_pipeline(
    training_texts: list,
    training_tokens: list,
    n_components: float = 0.95,
    models_dir: str = "models",
    lang: str = "en"
) -> EnhancedStylometryPipeline:
    """
    Train PCA pipeline on a collection of texts.
    
    Args:
        training_texts: List of training texts
        training_tokens: List of corresponding token lists
        n_components: Variance ratio to retain or number of components
        models_dir: Directory to save models
        lang: Language code
        
    Returns:
        Trained pipeline
    """
    logger.info(f"Training PCA pipeline on {len(training_texts)} texts")
    
    # Extract features for all training texts
    style_vectors = []
    for text, tokens in zip(training_texts, training_tokens):
        features, _ = extract_features(text, tokens, lang)
        style_vectors.append(features)
    
    style_vectors = np.array(style_vectors)
    logger.info(f"Extracted features: shape={style_vectors.shape}")
    
    # Create and fit pipeline
    pipeline = EnhancedStylometryPipeline(models_dir)
    pipeline.fit(style_vectors, n_components=n_components)
    
    return pipeline


def compute_pca_similarity(
    text: str,
    tokens: list,
    profile_style_mean: np.ndarray,
    profile_style_std: Optional[np.ndarray] = None,
    lang: str = "en",
    models_dir: str = "models"
) -> float:
    """
    Compute stylometry similarity using PCA-transformed features.
    
    Args:
        text: Input text
        tokens: List of word tokens
        profile_style_mean: Mean of user's PCA-transformed style features
        profile_style_std: Std dev of user's PCA-transformed style features
        lang: Language code
        models_dir: Directory containing PCA models
        
    Returns:
        Similarity score in [0, 1]
    """
    pipeline = get_enhanced_pipeline(models_dir)
    
    if not pipeline.is_fitted:
        logger.warning("PCA pipeline not fitted, falling back to raw features")
        from scoring import compute_stylometry_similarity
        raw_features, _ = extract_features(text, tokens, lang)
        return compute_stylometry_similarity(raw_features, profile_style_mean, profile_style_std)
    
    # Extract and transform features
    transformed_features, _ = pipeline.extract_and_transform_features(text, tokens, lang)
    
    # Compute enhanced similarity
    return pipeline.compute_enhanced_similarity(
        transformed_features,
        profile_style_mean,
        profile_style_std
    )