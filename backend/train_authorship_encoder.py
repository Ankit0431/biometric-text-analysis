#!/usr/bin/env python3
"""
Authorship Embedding Model Training

This script fine-tunes a sentence transformer model for user-specific authorship style
using triplet loss. The model learns to distinguish between different users' writing
styles by creating embeddings where texts from the same user are closer together
and texts from different users are farther apart.

Dataset structure: user_id, anchor_text, positive_text, negative_text
- anchor_text: Reference text from a user
- positive_text: Another text from the same user  
- negative_text: Text from a different user

Usage:
    python train_authorship_encoder.py --data triplets.csv --epochs 5 --batch-size 16
    python train_authorship_encoder.py --generate-sample-data --train
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import json
import random
from datetime import datetime

# Sentence transformers imports
try:
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from sentence_transformers.evaluation import TripletEvaluator
    from torch.utils.data import DataLoader
    import torch
except ImportError as e:
    print(f"Error: Missing required dependencies. Please install: pip install sentence-transformers torch")
    print(f"Import error: {e}")
    sys.exit(1)

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from normalizer import normalize
from features import extract_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AuthorshipEncoderTrainer:
    """Trainer for authorship embedding model using sentence transformers."""
    
    def __init__(
        self,
        base_model: str = "paraphrase-MiniLM-L6-v2",
        models_dir: str = "models",
        device: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            base_model: Base sentence transformer model to fine-tune
            models_dir: Directory to save trained models
            device: Device to use for training (cuda/cpu)
        """
        self.base_model = base_model
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = None
        self.training_stats = {}
        
    def load_base_model(self) -> None:
        """Load the base sentence transformer model."""
        logger.info(f"Loading base model: {self.base_model}")
        
        try:
            self.model = SentenceTransformer(self.base_model, device=self.device)
            logger.info(f"✓ Base model loaded successfully")
            logger.info(f"Model max sequence length: {self.model.max_seq_length}")
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
    
    def load_triplet_data(self, data_path: str) -> pd.DataFrame:
        """
        Load triplet training data from CSV file.
        
        Args:
            data_path: Path to CSV file with columns: user_id, anchor_text, positive_text, negative_text
            
        Returns:
            DataFrame with triplet data
        """
        logger.info(f"Loading triplet data from: {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            required_columns = ['user_id', 'anchor_text', 'positive_text', 'negative_text']
            
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Remove rows with empty text
            df = df.dropna(subset=['anchor_text', 'positive_text', 'negative_text'])
            df = df[df['anchor_text'].str.strip() != '']
            df = df[df['positive_text'].str.strip() != '']
            df = df[df['negative_text'].str.strip() != '']
            
            logger.info(f"✓ Loaded {len(df)} triplets for {df['user_id'].nunique()} users")
            
            # Show data statistics
            user_counts = df['user_id'].value_counts()
            logger.info(f"Triplets per user - Mean: {user_counts.mean():.1f}, Std: {user_counts.std():.1f}")
            logger.info(f"Top users by triplets: {dict(user_counts.head())}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load triplet data: {e}")
            raise
    
    def preprocess_triplet_data(self, df: pd.DataFrame, min_words: int = 20) -> pd.DataFrame:
        """
        Preprocess triplet data by normalizing texts and filtering short texts.
        
        Args:
            df: Raw triplet data
            min_words: Minimum words required per text
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing triplet data...")
        
        processed_rows = []
        
        for idx, row in df.iterrows():
            try:
                # Normalize all three texts
                anchor_norm = normalize(row['anchor_text'], flow_type='verify')
                positive_norm = normalize(row['positive_text'], flow_type='verify')
                negative_norm = normalize(row['negative_text'], flow_type='verify')
                
                # Check if all texts are acceptable
                if (not anchor_norm.rejected_reasons and 
                    not positive_norm.rejected_reasons and 
                    not negative_norm.rejected_reasons and
                    anchor_norm.word_count >= min_words and
                    positive_norm.word_count >= min_words and
                    negative_norm.word_count >= min_words):
                    
                    processed_rows.append({
                        'user_id': row['user_id'],
                        'anchor_text': anchor_norm.text,
                        'positive_text': positive_norm.text,
                        'negative_text': negative_norm.text,
                        'anchor_words': anchor_norm.word_count,
                        'positive_words': positive_norm.word_count,
                        'negative_words': negative_norm.word_count
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to process row {idx}: {e}")
                continue
        
        processed_df = pd.DataFrame(processed_rows)
        logger.info(f"✓ Preprocessed data: {len(processed_df)} triplets ({len(processed_df)/len(df)*100:.1f}% retained)")
        
        return processed_df
    
    def create_training_examples(self, df: pd.DataFrame) -> List[InputExample]:
        """
        Create InputExample objects for sentence transformers training.
        
        Args:
            df: Preprocessed triplet data
            
        Returns:
            List of InputExample objects
        """
        logger.info("Creating training examples...")
        
        examples = []
        
        for _, row in df.iterrows():
            example = InputExample(
                texts=[row['anchor_text'], row['positive_text'], row['negative_text']],
                label=0.0  # Not used in triplet loss but required
            )
            examples.append(example)
        
        logger.info(f"✓ Created {len(examples)} training examples")
        return examples
    
    def create_evaluation_examples(self, df: pd.DataFrame, eval_fraction: float = 0.2) -> List[InputExample]:
        """
        Create evaluation examples for monitoring training progress.
        
        Args:
            df: Preprocessed triplet data
            eval_fraction: Fraction of data to use for evaluation
            
        Returns:
            List of evaluation InputExample objects
        """
        logger.info("Creating evaluation examples...")
        
        # Sample evaluation data
        eval_size = max(10, int(len(df) * eval_fraction))
        eval_df = df.sample(n=min(eval_size, len(df)), random_state=42)
        
        eval_examples = []
        
        for _, row in eval_df.iterrows():
            example = InputExample(
                texts=[row['anchor_text'], row['positive_text'], row['negative_text']],
                label=0.0
            )
            eval_examples.append(example)
        
        logger.info(f"✓ Created {len(eval_examples)} evaluation examples")
        return eval_examples
    
    def train_model(
        self,
        train_examples: List[InputExample],
        eval_examples: Optional[List[InputExample]] = None,
        epochs: int = 2,
        batch_size: int = 16,
        warmup_steps: int = 100,
        learning_rate: float = 2e-5,
        output_dir: Optional[str] = None
    ) -> None:
        """
        Train the authorship embedding model.
        
        Args:
            train_examples: Training examples
            eval_examples: Optional evaluation examples
            epochs: Number of training epochs
            batch_size: Training batch size
            warmup_steps: Number of warmup steps
            learning_rate: Learning rate
            output_dir: Output directory for model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_base_model() first.")
        
        if output_dir is None:
            output_dir = self.models_dir / "authorship_encoder"
        else:
            output_dir = Path(output_dir)
        
        logger.info("=" * 60)
        logger.info("Starting Authorship Embedding Model Training")
        logger.info("=" * 60)
        logger.info(f"Training examples: {len(train_examples)}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Warmup steps: {warmup_steps}")
        logger.info(f"Output directory: {output_dir}")
        
        # Create data loader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Create triplet loss
        train_loss = losses.TripletLoss(model=self.model)
        
        # Setup evaluation if provided
        evaluator = None
        if eval_examples:
            evaluator = TripletEvaluator.from_input_examples(
                examples=eval_examples,
                name="triplet_evaluation"
            )
        
        # Training arguments
        train_objectives = [(train_dataloader, train_loss)]
        
        # Record training start time
        start_time = datetime.now()
        
        try:
            # Train the model
            self.model.fit(
                train_objectives=train_objectives,
                epochs=epochs,
                warmup_steps=warmup_steps,
                evaluator=evaluator,
                evaluation_steps=500,
                output_path=str(output_dir),
                save_best_model=True,
                optimizer_params={'lr': learning_rate}
            )
            
            # Record training statistics
            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()
            
            self.training_stats = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': training_duration,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'warmup_steps': warmup_steps,
                'training_examples': len(train_examples),
                'evaluation_examples': len(eval_examples) if eval_examples else 0,
                'base_model': self.base_model,
                'device': self.device
            }
            
            # Save training statistics
            stats_file = output_dir / "training_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
            
            logger.info("=" * 60)
            logger.info("✓ Training completed successfully!")
            logger.info(f"Training duration: {training_duration/60:.1f} minutes")
            logger.info(f"Model saved to: {output_dir}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self, output_dir: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(output_path))
        logger.info(f"✓ Model saved to: {output_path}")


def generate_sample_triplet_data(output_file: str = "sample_triplets.csv", num_users: int = 10, triplets_per_user: int = 20) -> None:
    """
    Generate sample triplet data for testing.
    
    Args:
        output_file: Output CSV file path
        num_users: Number of sample users
        triplets_per_user: Number of triplets per user
    """
    logger.info(f"Generating sample triplet data: {num_users} users, {triplets_per_user} triplets each")
    
    # Sample texts for different writing styles
    tech_texts = [
        "I believe that artificial intelligence will fundamentally revolutionize the way we approach software development and engineering practices. The integration of machine learning algorithms into traditional programming workflows presents both unprecedented opportunities and complex challenges that we must carefully consider from multiple perspectives. When designing AI-powered development tools, we need to balance automation with human oversight, ensuring that the technology enhances rather than replaces human creativity and problem-solving capabilities. The key is to create systems that augment developer productivity while maintaining code quality, security, and maintainability standards that are essential for long-term project success.",
        
        "When implementing distributed systems at scale, it's absolutely crucial to consider fault tolerance, data consistency, and network partitioning issues from the very beginning of the design process. The CAP theorem reminds us that we can't have all three guarantees simultaneously in a distributed environment, so we must make informed trade-offs based on our specific business requirements and technical constraints. I've found that understanding these fundamental principles early in the architecture phase saves countless hours of debugging and redesign later in the development lifecycle. The choice between consistency and availability often depends on the specific use case and business impact of temporary data inconsistencies.",
        
        "Code review processes are absolutely essential for maintaining software quality and knowledge sharing across development teams. I prefer to focus on architectural decisions, design patterns, and potential edge cases rather than minor style issues that can be automatically handled by linting tools and code formatters. During reviews, I always look for opportunities to improve code readability, performance, and maintainability while ensuring that the implementation aligns with established coding standards and best practices. The most valuable reviews often involve discussions about alternative approaches and their trade-offs rather than simply checking for syntax errors or style violations.",
        
        "Database optimization requires a deep understanding of query execution plans, indexing strategies, and storage engine characteristics. Performance tuning is often an iterative process that involves careful analysis of bottlenecks, systematic measurement of improvements, and continuous monitoring of system behavior under various load conditions. I've learned that premature optimization can be counterproductive, but understanding the underlying database mechanics is crucial for making informed decisions about schema design, query structure, and indexing strategies. The key is to establish baseline performance metrics and then make incremental improvements while monitoring their impact on overall system performance.",
        
        "Microservices architecture offers significant scalability benefits but introduces substantial complexity in service coordination, monitoring, and deployment orchestration. The decision to adopt microservices should be based on careful consideration of team size, system requirements, operational capabilities, and the organization's ability to manage distributed systems effectively. I've observed that successful microservices implementations require strong DevOps practices, comprehensive monitoring solutions, and well-defined service boundaries that minimize inter-service dependencies. The transition from monolithic to microservices architecture should be gradual and driven by specific business needs rather than following technology trends blindly."
    ]
    
    academic_texts = [
        "The research methodology employed in this comprehensive study follows a carefully designed mixed-methods approach, systematically combining quantitative statistical analysis with detailed qualitative observational techniques and ethnographic fieldwork. This methodological triangulation approach allows for a significantly more comprehensive and nuanced understanding of the complex social phenomenon under investigation, enabling researchers to capture both the measurable aspects and the subjective experiences of participants. The integration of multiple data collection methods helps to mitigate the inherent limitations of any single approach while providing robust validation of findings through cross-verification of results obtained through different methodological lenses.",
        
        "Previous literature has established a statistically significant correlation between the primary variables of interest; however, the underlying causal mechanisms and mediating pathways remain poorly understood and inadequately theorized in the existing scholarly discourse. Our current study aims to address this critical gap in the literature by systematically examining the underlying psychological and social processes through carefully controlled experimental manipulation and longitudinal observation. The research design incorporates multiple control conditions and employs sophisticated statistical modeling techniques to isolate the effects of specific variables while accounting for potential confounding factors that have been identified in previous research.",
        
        "The theoretical framework developed and presented in this paper builds substantially upon well-established foundational principles from cognitive psychology and social learning theory while simultaneously introducing several novel conceptual frameworks that significantly extend the current understanding of the domain. These theoretical contributions have far-reaching implications for both academic theory development and practical applications in educational and therapeutic settings. The proposed model synthesizes insights from diverse disciplines including neuroscience, developmental psychology, and social anthropology to provide a more holistic understanding of the phenomenon under investigation.",
        
        "Data collection procedures were meticulously designed and rigorously tested to minimize systematic bias and ensure truly representative sampling across diverse demographic groups and geographic regions. The comprehensive protocol was thoroughly reviewed and formally approved by the institutional review board to ensure strict ethical compliance throughout all phases of the research process. Particular attention was paid to informed consent procedures, participant confidentiality protections, and the development of culturally sensitive data collection instruments that could be appropriately adapted for use with participants from various linguistic and cultural backgrounds.",
        
        "The findings emerging from this investigation suggest that conventional wisdom and widely accepted assumptions regarding this topic may need to be fundamentally reconsidered and potentially revised in light of the new evidence presented here. The empirical results consistently point toward alternative theoretical explanations that warrant extensive further investigation and systematic validation through carefully designed replication studies across different populations and contexts. These unexpected findings have significant implications for current practice and policy recommendations in the field."
    ]
    
    creative_texts = [
        "The morning light filtered through the ancient oak trees, casting dancing shadows across the cobblestone path. There was something magical about this place that made time seem to slow down and worries fade away.",
        "Her laughter echoed through the empty hallway, a sound so pure and joyful that it seemed to breathe life into the old building. It was in moments like these that I truly understood the meaning of happiness.",
        "The storm approached with ominous clouds that promised both destruction and renewal. Nature has a way of reminding us of our place in the grand scheme of things, humbling us with its awesome power.",
        "As I turned the pages of the weathered journal, each entry revealed another piece of the mystery. The handwriting was elegant and deliberate, suggesting someone who chose their words with great care and intention.",
        "The small café at the corner had become my sanctuary, a place where strangers became friends over shared stories and steaming cups of coffee. It was here that I learned the art of listening."
    ]
    
    business_texts = [
        "Our quarterly results demonstrate strong growth across all key performance indicators. The strategic initiatives implemented last year are showing positive results, and we're well-positioned for continued expansion in the coming fiscal period.",
        "Market analysis reveals significant opportunities in the emerging sectors. However, we must carefully evaluate the risks and ensure our resource allocation aligns with long-term strategic objectives and stakeholder expectations.",
        "The competitive landscape has evolved rapidly, requiring us to adapt our go-to-market strategy. Innovation and customer-centricity will be critical factors in maintaining our market position and driving sustainable growth.",
        "Operational efficiency improvements have resulted in substantial cost savings while maintaining service quality. The implementation of new processes and technologies has streamlined workflows and enhanced productivity across departments.",
        "Partnership opportunities with industry leaders could accelerate our market penetration and provide access to new customer segments. Due diligence and careful negotiation will be essential to structure mutually beneficial agreements."
    ]
    
    # User writing style profiles
    user_styles = [
        ("tech_lead", tech_texts),
        ("researcher_1", academic_texts),
        ("researcher_2", academic_texts),
        ("writer_1", creative_texts),
        ("writer_2", creative_texts),
        ("executive_1", business_texts),
        ("executive_2", business_texts),
        ("developer_1", tech_texts),
        ("analyst_1", business_texts),
        ("professor_1", academic_texts)
    ]
    
    # Generate triplets
    triplets = []
    
    for user_id, user_texts in user_styles[:num_users]:
        for _ in range(triplets_per_user):
            # Select anchor and positive from same user
            anchor_text = random.choice(user_texts)
            positive_text = random.choice([t for t in user_texts if t != anchor_text])
            
            # Select negative from different user
            other_users = [texts for uid, texts in user_styles if uid != user_id]
            negative_user_texts = random.choice(other_users)
            negative_text = random.choice(negative_user_texts)
            
            triplets.append({
                'user_id': user_id,
                'anchor_text': anchor_text,
                'positive_text': positive_text,
                'negative_text': negative_text
            })
    
    # Save to CSV
    df = pd.DataFrame(triplets)
    df.to_csv(output_file, index=False)
    
    logger.info(f"✓ Generated {len(triplets)} triplets saved to: {output_file}")
    logger.info(f"Users: {df['user_id'].unique().tolist()}")


def replace_encoder_model(trained_model_path: str, target_encoder_path: Optional[str] = None) -> None:
    """
    Replace the current embedding model with the fine-tuned one.
    
    Args:
        trained_model_path: Path to the trained authorship model
        target_encoder_path: Path where the encoder should be replaced (optional)
    """
    logger.info("Replacing current encoder with fine-tuned authorship model")
    
    trained_path = Path(trained_model_path)
    if not trained_path.exists():
        raise FileNotFoundError(f"Trained model not found: {trained_path}")
    
    # Default target path
    if target_encoder_path is None:
        target_path = Path(__file__).parent / "models" / "sentence_encoder"
    else:
        target_path = Path(target_encoder_path)
    
    try:
        # Load the trained model to verify it works
        model = SentenceTransformer(str(trained_path))
        
        # Test encoding
        test_text = "This is a test sentence to verify the model works correctly."
        embedding = model.encode(test_text)
        
        logger.info(f"✓ Model verification successful - embedding shape: {embedding.shape}")
        
        # Save to target location
        model.save(str(target_path))
        
        logger.info(f"✓ Encoder model replaced successfully")
        logger.info(f"New model location: {target_path}")
        
    except Exception as e:
        logger.error(f"Failed to replace encoder model: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train authorship embedding model")
    parser.add_argument("--data", type=str, help="Path to triplet CSV data file")
    parser.add_argument("--generate-sample-data", action="store_true", 
                        help="Generate sample triplet data")
    parser.add_argument("--train", action="store_true", 
                        help="Train the model (requires --data or --generate-sample-data)")
    parser.add_argument("--epochs", type=int, default=2, 
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, 
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, 
                        help="Number of warmup steps")
    parser.add_argument("--base-model", type=str, default="paraphrase-MiniLM-L6-v2",
                        help="Base sentence transformer model")
    parser.add_argument("--output-dir", type=str, default="models/authorship_encoder",
                        help="Output directory for trained model")
    parser.add_argument("--replace-encoder", action="store_true",
                        help="Replace current encoder with trained model")
    parser.add_argument("--min-words", type=int, default=20,
                        help="Minimum words per text sample")
    
    args = parser.parse_args()
    
    # Generate sample data if requested
    if args.generate_sample_data:
        sample_file = "sample_triplets.csv"
        generate_sample_triplet_data(sample_file)
        if not args.data:
            args.data = sample_file
    
    # Train model if requested
    if args.train:
        if not args.data:
            logger.error("Training requires --data or --generate-sample-data")
            return 1
        
        # Initialize trainer
        trainer = AuthorshipEncoderTrainer(
            base_model=args.base_model,
            models_dir="models"
        )
        
        # Load base model
        trainer.load_base_model()
        
        # Load and preprocess data
        df = trainer.load_triplet_data(args.data)
        processed_df = trainer.preprocess_triplet_data(df, min_words=args.min_words)
        
        if len(processed_df) == 0:
            logger.error("No valid triplets after preprocessing")
            return 1
        
        # Create training examples
        train_examples = trainer.create_training_examples(processed_df)
        eval_examples = trainer.create_evaluation_examples(processed_df)
        
        # Train model
        trainer.train_model(
            train_examples=train_examples,
            eval_examples=eval_examples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            output_dir=args.output_dir
        )
        
        # Replace encoder if requested
        if args.replace_encoder:
            replace_encoder_model(args.output_dir)
        
        logger.info("✓ Training process completed successfully!")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())