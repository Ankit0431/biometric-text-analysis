#!/usr/bin/env python3
"""
PCA Stylometry Training Script

This script demonstrates how to train the PCA-enhanced stylometry pipeline
using enrollment data or sample texts. It can be used to:

1. Initialize the PCA pipeline from enrollment data in the database
2. Train on sample texts for demonstration
3. Update an existing PCA model with new data

Usage:
    python train_pca_stylometry.py --mode database    # Train from enrollment data
    python train_pca_stylometry.py --mode sample      # Train from sample texts
    python train_pca_stylometry.py --mode update      # Update existing model
"""

import os
import sys
import argparse
import logging
import numpy as np
from typing import List, Tuple

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stylometry_pca import train_pca_pipeline, get_enhanced_pipeline
from db import get_all_enrollment_data
from normalizer import normalize_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_texts() -> Tuple[List[str], List[List[str]]]:
    """
    Generate sample texts for PCA training demonstration.
    
    Returns:
        Tuple of (texts, token_lists)
    """
    sample_texts = [
        "I think this is a great idea. The implementation looks solid and the documentation is clear.",
        "The weather today is absolutely beautiful. I love how the sun is shining through the clouds.",
        "Programming can be challenging, but it's incredibly rewarding when you solve a difficult problem.",
        "My favorite book is definitely 'To Kill a Mockingbird'. The themes are timeless and relevant.",
        "I've been working on this project for months. The attention to detail really shows in the final product.",
        "Coffee in the morning is essential for me. I can't start my day without a good cup of joe.",
        "The new restaurant downtown has amazing food. Their pasta dishes are absolutely phenomenal.",
        "Learning new languages is fascinating. Each language offers a unique perspective on the world.",
        "Music has always been a passion of mine. There's something magical about how it connects people.",
        "Travel opens your mind to new cultures and experiences. Every trip teaches you something valuable.",
        "Technology is advancing at an incredible pace. It's exciting to see what innovations lie ahead.",
        "Reading books is one of my favorite pastimes. There's nothing like getting lost in a good story.",
        "Exercise is important for both physical and mental health. I try to stay active every day.",
        "Cooking is both an art and a science. The combination of creativity and precision is amazing.",
        "Gardening teaches patience and provides a connection to nature that's hard to find elsewhere.",
        "The ocean has always fascinated me with its vastness and mystery. Each wave tells a story.",
        "Photography captures moments that would otherwise be lost to time. It's a beautiful art form.",
        "Writing allows us to express thoughts and emotions in ways that spoken words sometimes cannot.",
        "Science fiction movies make us think about the future and our place in the universe.",
        "Good conversation with friends is one of life's greatest pleasures and most valuable experiences."
    ]
    
    # Normalize texts and extract tokens
    texts = []
    token_lists = []
    
    for text in sample_texts:
        try:
            normalized = normalize_text(text, lang="en")
            if normalized.success:
                texts.append(normalized.text)
                token_lists.append(normalized.tokens)
        except Exception as e:
            logger.warning(f"Failed to normalize text: {e}")
            continue
    
    logger.info(f"Generated {len(texts)} sample texts for training")
    return texts, token_lists


def load_enrollment_data() -> Tuple[List[str], List[List[str]]]:
    """
    Load enrollment data from the database for PCA training.
    
    Returns:
        Tuple of (texts, token_lists)
    """
    try:
        enrollment_data = get_all_enrollment_data()
        texts = []
        token_lists = []
        
        for data in enrollment_data:
            # Extract text samples from enrollment data
            if 'text_samples' in data:
                for sample in data['text_samples']:
                    try:
                        normalized = normalize_text(sample, lang="en")
                        if normalized.success:
                            texts.append(normalized.text)
                            token_lists.append(normalized.tokens)
                    except Exception as e:
                        logger.warning(f"Failed to normalize enrollment text: {e}")
                        continue
        
        logger.info(f"Loaded {len(texts)} texts from enrollment data")
        return texts, token_lists
        
    except Exception as e:
        logger.error(f"Failed to load enrollment data: {e}")
        return [], []


def train_from_database(n_components: float = 0.95) -> None:
    """
    Train PCA pipeline from enrollment data in database.
    
    Args:
        n_components: Variance ratio to retain
    """
    logger.info("Training PCA pipeline from database enrollment data")
    
    texts, token_lists = load_enrollment_data()
    
    if not texts:
        logger.error("No enrollment data found. Please ensure users have enrolled first.")
        return
    
    if len(texts) < 10:
        logger.warning(f"Only {len(texts)} texts available. Consider adding more enrollment data for better PCA.")
    
    # Train the pipeline
    pipeline = train_pca_pipeline(
        training_texts=texts,
        training_tokens=token_lists,
        n_components=n_components,
        models_dir="models",
        lang="en"
    )
    
    logger.info("PCA pipeline training completed successfully!")


def train_from_samples(n_components: float = 0.95) -> None:
    """
    Train PCA pipeline from sample texts for demonstration.
    
    Args:
        n_components: Variance ratio to retain
    """
    logger.info("Training PCA pipeline from sample texts")
    
    texts, token_lists = generate_sample_texts()
    
    # Train the pipeline
    pipeline = train_pca_pipeline(
        training_texts=texts,
        training_tokens=token_lists,
        n_components=n_components,
        models_dir="models",
        lang="en"
    )
    
    logger.info("PCA pipeline training completed successfully!")


def update_existing_model() -> None:
    """
    Update existing PCA model with new data.
    """
    logger.info("Updating existing PCA model")
    
    pipeline = get_enhanced_pipeline()
    
    if not pipeline.is_fitted:
        logger.error("No existing PCA model found. Train a new model first.")
        return
    
    logger.info("Current PCA model statistics:")
    logger.info(f"  Components: {pipeline.pca.n_components_}")
    logger.info(f"  Explained variance: {pipeline.pca.explained_variance_ratio_.sum():.3f}")
    
    # For updating, we would need to retrain with combined old and new data
    # This is a simplified example
    logger.info("To update the model, retrain with combined old and new data")
    logger.info("Consider using incremental PCA for large datasets")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train PCA-enhanced stylometry pipeline")
    parser.add_argument(
        "--mode",
        choices=["database", "sample", "update"],
        default="sample",
        help="Training mode: database (from enrollment data), sample (demo), or update (existing model)"
    )
    parser.add_argument(
        "--components",
        type=float,
        default=0.95,
        help="PCA components to retain (as variance ratio)"
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to save PCA models"
    )
    
    args = parser.parse_args()
    
    # Ensure models directory exists
    os.makedirs(args.models_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PCA Stylometry Training Script")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Components: {args.components}")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info("-" * 60)
    
    try:
        if args.mode == "database":
            train_from_database(args.components)
        elif args.mode == "sample":
            train_from_samples(args.components)
        elif args.mode == "update":
            update_existing_model()
        
        logger.info("-" * 60)
        logger.info("Training completed successfully!")
        
        # Test the trained pipeline
        pipeline = get_enhanced_pipeline(args.models_dir)
        if pipeline.is_fitted:
            logger.info("Pipeline validation:")
            logger.info(f"  PCA components: {pipeline.pca.n_components_}")
            logger.info(f"  Explained variance ratio: {pipeline.pca.explained_variance_ratio_.sum():.3f}")
            logger.info(f"  Feature dimensions: {pipeline.pca.n_features_in_} -> {pipeline.pca.n_components_}")
        else:
            logger.warning("Pipeline is not fitted after training")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())