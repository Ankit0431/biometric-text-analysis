#!/usr/bin/env python3
"""
Integration test for PCA-enhanced stylometry in the API context.
"""

import os
import sys
import logging
import numpy as np

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scoring import compute_stylometry_similarity
from normalizer import normalize
from features import extract_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  
)
logger = logging.getLogger(__name__)


def test_api_integration():
    """Test PCA integration in API context."""
    logger.info("Testing PCA integration in API scoring context")
    
    # Sample texts that would be used in enrollment and verification
    enrollment_text = """
    I have always been passionate about technology and its potential to solve complex problems. 
    My writing style tends to be methodical and structured, as I believe in presenting information 
    in a logical sequence. When I'm explaining technical concepts, I try to break them down into 
    manageable pieces that build upon each other. This approach has served me well throughout my 
    career in software development, where clear communication is just as important as technical skills.
    """
    
    verification_text = """
    Technology has always fascinated me, particularly its ability to address challenging issues 
    across various domains. My approach to writing is typically systematic and well-organized, 
    reflecting my belief that information should be presented in a coherent manner. When discussing 
    technical topics, I prefer to decompose complex ideas into smaller, more digestible components. 
    This methodology has been invaluable in my professional journey in the tech industry.
    """
    
    # Normalize the texts
    norm_enroll = normalize(enrollment_text, flow_type="enroll")
    norm_verify = normalize(verification_text, flow_type="verify")
    
    if norm_enroll.rejected_reasons or norm_verify.rejected_reasons:
        logger.error("Text normalization failed")
        return False
    
    # Extract features
    enroll_features, _ = extract_features(norm_enroll.text, norm_enroll.tokens, lang="en")
    verify_features, _ = extract_features(norm_verify.text, norm_verify.tokens, lang="en")
    
    if enroll_features is None or verify_features is None:
        logger.error("Feature extraction failed")
        return False
    
    logger.info(f"Extracted features: {enroll_features.shape}")
    
    # Test with PCA enabled
    try:
        pca_similarity = compute_stylometry_similarity(
            style_vector=verify_features,
            profile_style_mean=enroll_features,
            profile_style_std=None,
            use_pca=True
        )
        logger.info(f"✓ PCA-enhanced similarity: {pca_similarity:.4f}")
    except Exception as e:
        logger.error(f"✗ PCA similarity computation failed: {e}")
        return False
    
    # Test with legacy mode (PCA disabled)
    try:
        legacy_similarity = compute_stylometry_similarity(
            style_vector=verify_features, 
            profile_style_mean=enroll_features,
            profile_style_std=None,
            use_pca=False
        )
        logger.info(f"✓ Legacy similarity: {legacy_similarity:.4f}")
    except Exception as e:
        logger.error(f"✗ Legacy similarity computation failed: {e}")
        return False
    
    # Compare the results
    similarity_diff = abs(pca_similarity - legacy_similarity)
    logger.info(f"Similarity difference (PCA vs Legacy): {similarity_diff:.4f}")
    
    if similarity_diff > 0.5:
        logger.warning("Large difference between PCA and legacy similarities")
    else:
        logger.info("PCA and legacy similarities are reasonably close")
    
    logger.info("✓ API integration test completed successfully")
    return True


if __name__ == "__main__":
    success = test_api_integration()
    sys.exit(0 if success else 1)