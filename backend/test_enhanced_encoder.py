#!/usr/bin/env python3
"""
Test the enhanced encoder with sentence transformer support.
"""

import os
import sys
import logging

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from encoder import TextEncoder, get_encoder, encode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_encoder():
    """Test the enhanced encoder functionality."""
    logger.info("Testing Enhanced Text Encoder")
    logger.info("=" * 50)
    
    # Test texts
    test_texts = [
        "I believe that machine learning will revolutionize how we approach software development.",
        "The weather today is beautiful with clear skies and gentle breezes.",
        "Programming requires both logical thinking and creative problem-solving skills."
    ]
    
    # Test with sentence transformer preference
    logger.info("Testing with sentence transformer preference...")
    try:
        encoder_st = TextEncoder(prefer_sentence_transformer=True)
        embeddings_st = encoder_st.encode(test_texts)
        logger.info(f"✓ Sentence transformer mode: {embeddings_st.shape}")
        logger.info(f"  Mode: {encoder_st.mode}")
        
        # Check normalization
        norms = [(emb**2).sum()**0.5 for emb in embeddings_st]
        logger.info(f"  L2 norms: {[f'{norm:.3f}' for norm in norms]}")
        
    except Exception as e:
        logger.error(f"✗ Sentence transformer test failed: {e}")
    
    # Test with transformer preference
    logger.info("\nTesting with transformer preference...")
    try:
        encoder_tf = TextEncoder(prefer_sentence_transformer=False)
        embeddings_tf = encoder_tf.encode(test_texts)
        logger.info(f"✓ Transformer mode: {embeddings_tf.shape}")
        logger.info(f"  Mode: {encoder_tf.mode}")
        
        # Check normalization
        norms = [(emb**2).sum()**0.5 for emb in embeddings_tf]
        logger.info(f"  L2 norms: {[f'{norm:.3f}' for norm in norms]}")
        
    except Exception as e:
        logger.error(f"✗ Transformer test failed: {e}")
    
    # Test global encoder function
    logger.info("\nTesting global encoder function...")
    try:
        embeddings_global = encode(test_texts)
        logger.info(f"✓ Global encoder: {embeddings_global.shape}")
        
        # Check normalization
        norms = [(emb**2).sum()**0.5 for emb in embeddings_global]
        logger.info(f"  L2 norms: {[f'{norm:.3f}' for norm in norms]}")
        
    except Exception as e:
        logger.error(f"✗ Global encoder test failed: {e}")
    
    logger.info("=" * 50)
    logger.info("Enhanced encoder test completed!")


if __name__ == "__main__":
    test_encoder()