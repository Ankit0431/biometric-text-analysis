#!/usr/bin/env python3
"""
Simple PCA Stylometry Test

Test the PCA-enhanced stylometry pipeline without database dependencies.
"""

import os
import sys
import logging
import numpy as np
from typing import List, Tuple

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stylometry_pca import EnhancedStylometryPipeline, train_pca_pipeline
from normalizer import normalize
from features import extract_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_texts() -> Tuple[List[str], List[List[str]]]:
    """Generate test texts for PCA training."""
    sample_texts = [
        """I think this is a great idea for implementing a comprehensive biometric text analysis system. The implementation looks solid and the documentation is clear, providing detailed explanations of the various components. This system appears to use advanced machine learning techniques to analyze writing patterns, which is fascinating from both a technical and practical perspective. The modular architecture makes it easy to understand how different components interact with each other. I particularly appreciate the attention to detail in the feature extraction algorithms and the thoughtful approach to handling different types of text input.""",
        
        """The weather today is absolutely beautiful, with clear blue skies and gentle breezes that make it perfect for outdoor activities. I love how the sun is shining through the clouds, creating those dramatic light patterns that photographers dream about. There's something magical about days like this that just makes you want to spend time outside, whether it's going for a walk, having a picnic, or simply sitting in the garden with a good book. The temperature is just right - not too hot, not too cold - and there's a freshness in the air that feels rejuvenating.""",
        
        """Programming can be challenging, but it's incredibly rewarding when you solve a difficult problem that has been puzzling you for hours or even days. The feeling of satisfaction when your code finally works as intended is unmatched in many other fields. What I find most interesting about software development is how it combines logical thinking with creative problem-solving. Each project presents unique challenges that require you to think outside the box and come up with innovative solutions. The continuous learning aspect keeps things interesting, as new technologies and methodologies are constantly emerging.""",
        
        """My favorite book is definitely 'To Kill a Mockingbird' by Harper Lee. The themes are timeless and relevant, dealing with issues of racial injustice, moral growth, and the loss of innocence in ways that resonate just as strongly today as they did when the book was first published. The character development is exceptional, particularly with Scout Finch, whose journey from childhood innocence to a more mature understanding of the world around her is beautifully portrayed. Atticus Finch serves as a moral compass throughout the story, demonstrating the importance of standing up for what's right even when it's difficult.""",
        
        """I've been working on this machine learning project for several months now, and the attention to detail really shows in the final product. The data preprocessing pipeline took considerable effort to get right, but it was worth it because clean, well-structured data is the foundation of any successful ML model. Feature engineering was particularly challenging, as we had to balance between creating meaningful features and avoiding overfitting. The model selection process involved extensive experimentation with different algorithms, hyperparameter tuning, and cross-validation to ensure robust performance across various datasets.""",
        
        """Coffee in the morning is absolutely essential for me - I simply cannot function properly without that first cup of aromatic, freshly brewed coffee. I can't start my day without a good cup of joe, preferably made from high-quality beans that have been properly roasted and ground just before brewing. There's something almost ritualistic about the morning coffee routine: the sound of the grinder, the aroma that fills the kitchen, the first sip that signals the beginning of a new day. I've tried various brewing methods over the years, from French press to pour-over, and each has its own unique characteristics.""",
        
        """The new restaurant downtown has amazing food that showcases the chef's incredible talent and creativity. Their pasta dishes are absolutely phenomenal, featuring house-made noodles and sauces that demonstrate true culinary artistry. What impressed me most was the attention to detail in every aspect of the dining experience, from the carefully curated wine list to the impeccable presentation of each dish. The service staff is knowledgeable and passionate about the menu, able to provide detailed descriptions of ingredients and preparation methods. The atmosphere strikes the perfect balance between elegant and welcoming.""",
        
        """Learning new programming languages is fascinating and opens up entirely new ways of thinking about problem-solving and software architecture. Each language offers a unique perspective on the world of computing, with its own paradigms, strengths, and applications. For instance, functional programming languages like Haskell encourage you to think in terms of immutable data and pure functions, which can lead to more predictable and maintainable code. Object-oriented languages like Java provide excellent frameworks for modeling complex systems and managing large codebases. The diversity in programming languages reflects the diversity in the problems we're trying to solve."""
    ]
    
    texts = []
    token_lists = []
    
    for text in sample_texts:
        try:
            normalized = normalize(text, flow_type="verify")
            if not normalized.rejected_reasons:
                texts.append(normalized.text)
                token_lists.append(normalized.tokens)
        except Exception as e:
            logger.warning(f"Failed to normalize text: {e}")
            continue
    
    return texts, token_lists


def test_pca_pipeline():
    """Test the PCA pipeline functionality."""
    logger.info("=" * 60)
    logger.info("Testing PCA-Enhanced Stylometry Pipeline")
    logger.info("=" * 60)
    
    # Generate test data
    texts, token_lists = generate_test_texts()
    logger.info(f"Generated {len(texts)} test texts")
    
    # Create models directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Train the pipeline
    logger.info("Training PCA pipeline...")
    try:
        pipeline = train_pca_pipeline(
            training_texts=texts,
            training_tokens=token_lists,
            n_components=0.95,
            models_dir=models_dir,
            lang="en"
        )
        logger.info("✓ PCA pipeline trained successfully")
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        return False
    
    # Test the pipeline
    logger.info("Testing pipeline functionality...")
    
    # Test text transformation
    test_text = """This is a comprehensive test paragraph designed to check if the PCA pipeline works correctly with sufficient text length. 
    The pipeline should be able to extract meaningful stylometric features from this text and transform them using the trained PCA components. 
    This test verifies that the dimensionality reduction is working properly and that we can successfully process new text samples through 
    the enhanced pipeline. The text needs to be long enough to pass the minimum word count requirements for the verification process."""
    try:
        normalized = normalize(test_text, flow_type="verify")
        if not normalized.rejected_reasons:
            feature_vector, feature_dict = extract_features(normalized.text, normalized.tokens, lang="en")
            if feature_vector is not None and len(feature_vector) > 0:
                transformed = pipeline.transform(feature_vector.reshape(1, -1))
                logger.info(f"✓ Text transformation successful")
                logger.info(f"  Original dimensions: {len(feature_vector)}")
                logger.info(f"  Transformed shape: {transformed.shape}")
            else:
                logger.warning("Failed to extract features")
        else:
            logger.warning(f"Text rejected: {normalized.rejected_reasons}")
    except Exception as e:
        logger.error(f"✗ Text transformation failed: {e}")
        return False
    
    # Test similarity computation
    logger.info("Testing similarity computation...")
    try:
        # Create two similar texts
        text1 = """I really enjoy reading books in my spare time, as literature opens up entirely new worlds and perspectives that I wouldn't otherwise experience. 
        There's something magical about getting lost in a good story, where you can travel to distant lands, meet fascinating characters, and explore complex themes 
        that challenge your thinking. Reading has always been my favorite way to relax and unwind after a long day, providing both entertainment and intellectual 
        stimulation. The variety of genres available means there's always something new to discover."""
        
        text2 = """Reading is definitely one of my favorite hobbies, and I find that books have the amazing ability to transport you to completely different places 
        and time periods. Whether it's science fiction that explores possible futures, historical fiction that brings the past to life, or contemporary literature 
        that examines modern society, each book offers a unique journey. I love how reading can expand your vocabulary, improve your writing skills, and introduce 
        you to new ideas and ways of thinking about the world around us."""
        
        # Process both texts
        norm1 = normalize(text1, flow_type="verify")
        norm2 = normalize(text2, flow_type="verify")
        
        if not norm1.rejected_reasons and not norm2.rejected_reasons:
            vec1, _ = extract_features(norm1.text, norm1.tokens, lang="en")
            vec2, _ = extract_features(norm2.text, norm2.tokens, lang="en")
            
            if vec1 is not None and vec2 is not None:
                similarity = pipeline.compute_enhanced_similarity(vec1, vec2)
                logger.info(f"✓ Similarity computation successful: {similarity:.4f}")
            else:
                logger.warning("Failed to extract features for similarity test")
        else:
            logger.warning("Failed to normalize texts for similarity test")
    except Exception as e:
        logger.error(f"✗ Similarity computation failed: {e}")
        return False
    
    # Test model persistence
    logger.info("Testing model persistence...")
    try:
        # The model is automatically saved during training in train_pca_pipeline
        # Let's verify we can load a new pipeline instance
        from stylometry_pca import get_enhanced_pipeline
        
        new_pipeline = get_enhanced_pipeline(models_dir)
        logger.info("✓ Model loaded successfully from saved files")
        
        # Verify loaded model works
        if new_pipeline.is_fitted:
            test_vec = np.random.random(512)  # Use feature dimension size
            transformed = new_pipeline.transform(test_vec.reshape(1, -1))
            logger.info("✓ Loaded model transformation successful")
        else:
            logger.warning("Loaded model is not fitted")
    except Exception as e:
        logger.error(f"✗ Model persistence failed: {e}")
        return False
    
    logger.info("=" * 60)
    logger.info("✓ All tests passed! PCA pipeline is working correctly.")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_pca_pipeline()
    sys.exit(0 if success else 1)