#!/usr/bin/env python3
"""
LLM Detection Model Training Script

This script trains a GradientBoostingClassifier to detect AI-generated text using
comprehensive linguistic features. It loads the HC3 dataset directly from Hugging Face
for high-quality training data.

Supported datasets:
- HC3 (Human ChatGPT Comparison Corpus) from Hugging Face
- Local HC3 dataset files
- Custom CSV files with 'text' and 'label' columns

Usage:
    python train_llm_detector.py --dataset-type hc3 --test-model
    python train_llm_detector.py --dataset-type csv --dataset path/to/dataset.csv
    python train_llm_detector.py --dataset-type hc3-local --dataset path/to/hc3/
"""

import os
import sys
import argparse
import numpy as np
import json
from typing import List, Tuple, Dict, Any, Optional, Union
import logging
from datetime import datetime
import re

# Try to import optional pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Try to import datasets library for HC3
try:
    from datasets import load_dataset, DatasetDict, Dataset, IterableDataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Try to import required libraries
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score, 
        recall_score, f1_score, classification_report, confusion_matrix
    )
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Required libraries not available: {e}")
    print("Please install with: pip install scikit-learn datasets")
    sys.exit(1)

# Import our LLM detection module
try:
    from llm_detection import extract_llm_features, initialize_llm_detection, SKLEARN_AVAILABLE
    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn is not available. Please install it with:")
        print("pip install scikit-learn")
        sys.exit(1)
except ImportError as e:
    print(f"ERROR: Could not import LLM detection module: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_enhanced_sample_data(n_samples: int = 1000) -> Tuple[List[str], List[int]]:
    """
    Generate enhanced sample data for training with realistic human and AI text patterns.
    
    Args:
        n_samples: Total number of samples to generate (split equally between human/AI)
        
    Returns:
        Tuple of (texts, labels) where labels are 0=human, 1=AI
    """
    logger.info(f"Generating {n_samples} enhanced sample texts...")
    
    texts = []
    labels = []
    
    # Human-like text patterns
    human_templates = [
        "I think this is really interesting because {topic}. My experience with {subject} has shown me that {insight}. What do you think about {question}?",
        "Honestly, I'm not sure about {topic}. I've been thinking about {subject} lately and it's pretty confusing. Maybe {suggestion}?",
        "OMG! I just discovered {topic} and it's amazing! I can't believe I didn't know about {subject} before. Anyone else tried {action}?",
        "So I was reading about {topic} and thought it was kinda boring at first. But then I realized {insight} about {subject}. Pretty cool stuff!",
        "My friend told me about {topic} yesterday. I'm still learning about {subject} but it seems like {observation}. Has anyone else noticed this?",
        "I've always been curious about {topic}. After studying {subject} for a while, I think {conclusion}. But I could be wrong...",
        "This might sound weird, but {topic} reminds me of {comparison}. The whole {subject} thing just feels like {emotion} to me.",
        "I stumbled across {topic} while procrastinating (lol). Spent 3 hours reading about {subject} instead of doing work. Worth it though!",
    ]
    
    # AI-like text patterns  
    ai_templates = [
        "Based on current research and analysis, {topic} demonstrates significant potential in {subject}. The key considerations include {insight} and {conclusion}.",
        "From a comprehensive perspective, {topic} represents an important development in {subject}. This analysis suggests that {insight} while maintaining {consideration}.",
        "The fundamental principles underlying {topic} indicate substantial implications for {subject}. Research demonstrates that {insight} through {method}.",
        "In examining {topic}, it becomes evident that {subject} requires careful consideration of {factor}. The analysis reveals {insight} and {implication}.",
        "Contemporary studies regarding {topic} highlight the significance of {subject} in {context}. The evidence suggests {insight} with {qualification}.",
        "A thorough evaluation of {topic} reveals important aspects of {subject}. The data indicates {insight} while considering {limitation}.",
        "The systematic approach to understanding {topic} involves analyzing {subject} through {method}. This process demonstrates {insight} and {outcome}.",
        "Research indicates that {topic} plays a crucial role in {subject}. The findings suggest {insight} with implications for {application}.",
    ]
    
    # Content pools for template filling
    topics = ["artificial intelligence", "climate change", "renewable energy", "space exploration", "quantum computing", 
              "biotechnology", "social media", "remote work", "electric vehicles", "cryptocurrency"]
    
    subjects = ["technology development", "scientific research", "human behavior", "economic systems", "environmental impact",
                "social interaction", "innovation processes", "data analysis", "problem solving", "creative expression"]
    
    insights = ["patterns emerge unexpectedly", "complexity increases over time", "simplicity often works best",
                "collaboration drives progress", "individual perspectives matter", "systematic approaches help",
                "creative solutions surprise us", "data tells interesting stories"]
    
    # Generate human samples
    n_human = n_samples // 2
    for i in range(n_human):
        template = human_templates[i % len(human_templates)]
        topic = topics[i % len(topics)]
        subject = subjects[(i + 3) % len(subjects)]
        insight = insights[i % len(insights)]
        
        # Add some variety
        question = f"how {subject} affects our daily lives"
        suggestion = f"we should explore {topic} more deeply"
        action = f"working with {subject}"
        observation = f"{topic} is more complex than expected"
        conclusion = f"{subject} has broader implications"
        comparison = f"learning about {subject}"
        emotion = "overwhelming but exciting"
        
        # Only use the variables that are actually in the template
        available_vars = {
            'topic': topic, 'subject': subject, 'insight': insight,
            'question': question, 'suggestion': suggestion, 'action': action,
            'observation': observation, 'conclusion': conclusion,
            'comparison': comparison, 'emotion': emotion
        }
        
        # Get only the variables needed by this template
        import re
        needed_vars = re.findall(r'\{(\w+)\}', template)
        template_vars = {var: available_vars.get(var, var) for var in needed_vars}
        
        text = template.format(**template_vars)
        
        texts.append(text)
        labels.append(0)  # Human
    
    # Generate AI samples
    n_ai = n_samples - n_human
    for i in range(n_ai):
        template = ai_templates[i % len(ai_templates)]
        topic = topics[(i + 2) % len(topics)]
        subject = subjects[(i + 5) % len(subjects)]
        insight = insights[(i + 1) % len(insights)]
        
        # Add variety for AI templates
        consideration = "ethical implications"
        method = "systematic evaluation"
        factor = "multiple variables"
        implication = "significant outcomes"
        context = "modern applications"
        qualification = "appropriate limitations"
        limitation = "current constraints"
        outcome = "measurable results"
        application = "practical implementation"
        
        # Only use the variables that are actually in the template
        available_vars = {
            'topic': topic, 'subject': subject, 'insight': insight,
            'consideration': consideration, 'method': method, 'factor': factor,
            'implication': implication, 'context': context, 'qualification': qualification,
            'limitation': limitation, 'outcome': outcome, 'application': application
        }
        
        # Get only the variables needed by this template
        needed_vars = re.findall(r'\{(\w+)\}', template)
        template_vars = {var: available_vars.get(var, var) for var in needed_vars}
        
        text = template.format(**template_vars)
        
        texts.append(text)
        labels.append(1)  # AI
    
    logger.info(f"Generated {len(texts)} enhanced sample texts ({n_human} human, {n_ai} AI)")
    return finalize_dataset(texts, labels)


def load_hc3_huggingface_dataset(subset: str = "all", max_samples: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """
    Load HC3 (Human ChatGPT Comparison Corpus) dataset from local CSV files.
    Uses real HC3 data downloaded from Google Drive links.
    
    Args:
        subset: Dataset subset to load ("all", "train", "test")
        max_samples: Maximum number of samples per class (None for all)
        
    Returns:
        Tuple of (texts, labels) where labels are 0=human, 1=AI
    """
    logger.info(f"Loading real HC3 dataset from local files (subset: {subset})...")
    
    try:
        # Load real HC3 data from local CSV files
        return load_hc3_from_modelscope(subset, max_samples)
        
    except Exception as e:
        logger.error(f"Failed to load HC3 dataset from local files: {e}")
        raise Exception(f"Could not load HC3 dataset from local CSV files. Error: {e}")


def load_hc3_from_modelscope(subset: str = "all", max_samples: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """
    Load HC3 dataset from local CSV files downloaded from Google Drive.
    
    Args:
        subset: Dataset subset ("all", "train", "test") 
        max_samples: Maximum samples per class
        
    Returns:
        Tuple of (texts, labels)
    """
    import os
    
    logger.info("Loading HC3 data from local CSV files...")
    
    # Check if pandas is available for CSV reading
    if not PANDAS_AVAILABLE:
        raise ImportError("Pandas is required for loading CSV files. Install with: pip install pandas")
    
    import pandas as pd
    
    # Map subset to filename
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    filename_map = {
        "train": "en_train.csv",
        "test": "en_test.csv", 
        "all": ["en_train.csv", "en_test.csv"]
    }
    
    files_to_load = filename_map.get(subset, ["en_train.csv", "en_test.csv"])
    if isinstance(files_to_load, str):
        files_to_load = [files_to_load]
    
    texts = []
    labels = []
    human_count = 0
    ai_count = 0
    
    for filename in files_to_load:
        file_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"HC3 data file not found: {file_path}")
        
        logger.info(f"Loading {filename}...")
        df = pd.read_csv(file_path)
        
        logger.info(f"Loaded {len(df)} rows from {filename}")
        
        # Process each row
        for _, row in df.iterrows():
            answer = str(row['answer']).strip()
            label = int(row['label'])  # 0=human, 1=AI
            
            # Skip short or invalid answers
            if len(answer) < 20:
                continue
            
            # Check if we need more samples of this type
            if label == 0:  # Human
                if max_samples is None or human_count < max_samples:
                    texts.append(answer)
                    labels.append(0)
                    human_count += 1
            else:  # AI
                if max_samples is None or ai_count < max_samples:
                    texts.append(answer)
                    labels.append(1)
                    ai_count += 1
            
            # Stop if we have enough samples
            if max_samples is not None and human_count >= max_samples and ai_count >= max_samples:
                break
        
        # Stop processing files if we have enough samples
        if max_samples is not None and human_count >= max_samples and ai_count >= max_samples:
            break
    
    if not texts:
        raise Exception("No valid text data found in HC3 CSV files")
    
    logger.info(f"Loaded real HC3 data: {human_count} human, {ai_count} AI samples")
    return finalize_dataset(texts, labels)


def load_hc3_from_gdrive(subset: str = "all", max_samples: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """
    Load HC3 dataset from Google Drive (fallback method).
    Note: This requires the files to be publicly accessible.
    """
    logger.warning("Google Drive access not implemented - requires authentication")
    raise Exception("Google Drive access not available. Please use ModelScope source.")


def process_hc3_dataset(ds, max_samples: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """Process HC3 dataset structure."""
    texts = []
    labels = []
    
    # Handle different dataset structures
    if hasattr(ds, 'keys') and callable(getattr(ds, 'keys')):
        # DatasetDict with multiple splits
        for split_name in ds.keys():
            logger.info(f"Processing split: {split_name}")
            split_data = ds[split_name]
            human_count, ai_count = process_hc3_split(split_data, texts, labels, max_samples)
            logger.info(f"  Split {split_name}: {human_count} human, {ai_count} AI samples")
    else:
        # Single dataset
        logger.info("Processing single dataset")
        human_count, ai_count = process_hc3_split(ds, texts, labels, max_samples)
        logger.info(f"  Dataset: {human_count} human, {ai_count} AI samples")
    
    if not texts:
        raise ValueError("No valid data found in HC3 dataset")
    
    return finalize_dataset(texts, labels)


def load_imdb_with_synthetic_ai(max_samples: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """Load IMDB dataset and create synthetic AI text as fallback."""
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library required")
        
    ds = load_dataset("imdb")
    
    texts = []
    labels = []
    
    # Use IMDB train split for human text
    train_data = ds['train']
    human_count = 0
    
    for item in train_data:
        if max_samples is None or human_count < max_samples:
            # Handle both dict-like and attribute access
            text = item.get('text', '') if hasattr(item, 'get') else getattr(item, 'text', '')
            if isinstance(text, str) and len(text.strip()) > 50:  # Ensure reasonable length
                texts.append(text.strip())
                labels.append(0)  # Human
                human_count += 1
        else:
            break
    
    # Create synthetic AI-like text by modifying human text
    ai_count = 0
    for i, text in enumerate(texts[:]):
        if max_samples is None or ai_count < max_samples:
            # Simple synthetic AI transformation
            ai_text = create_synthetic_ai_text(text)
            texts.append(ai_text)
            labels.append(1)  # AI
            ai_count += 1
        else:
            break
    
    logger.info(f"Created dataset with {human_count} human and {ai_count} synthetic AI samples")
    return finalize_dataset(texts, labels)


def create_synthetic_ai_text(human_text: str) -> str:
    """Create synthetic AI-like text from human text."""
    # Simple transformations to make text more AI-like
    lines = human_text.split('. ')
    
    # Add AI-like characteristics
    ai_phrases = [
        "As an AI language model, I would say that",
        "In my analysis,",
        "Based on the information provided,",
        "It's worth noting that",
        "From my perspective,",
    ]
    
    # Add an AI-like opening
    import random
    random.seed(hash(human_text) % 2**32)  # Consistent randomness
    ai_opening = random.choice(ai_phrases)
    
    # Make sentences more formal/structured
    formal_text = '. '.join(lines[:3])  # Truncate for AI-like brevity
    
    return f"{ai_opening} {formal_text}."


def finalize_dataset(texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
    """Shuffle and finalize the dataset."""
    # Shuffle the data
    combined = list(zip(texts, labels))
    np.random.shuffle(combined)
    
    # Unpack and convert back to lists
    shuffled_texts = []
    shuffled_labels = []
    for text, label in combined:
        shuffled_texts.append(text)
        shuffled_labels.append(label)
    
    human_total = sum(1 for l in shuffled_labels if l == 0)
    ai_total = sum(1 for l in shuffled_labels if l == 1)
    
    logger.info(f"Final dataset: {len(shuffled_texts)} total samples")
    logger.info(f"  Human samples: {human_total}")
    logger.info(f"  AI samples: {ai_total}")
    
    return shuffled_texts, shuffled_labels


def process_hc3_split(split_data, texts: List[str], labels: List[int], max_samples: Optional[int] = None) -> Tuple[int, int]:
    """
    Process a single split of the HC3 dataset.
    
    Args:
        split_data: Dataset split to process
        texts: List to append texts to
        labels: List to append labels to
        max_samples: Maximum samples per class
        
    Returns:
        Tuple of (human_count, ai_count)
    """
    human_count = 0
    ai_count = 0
    
    for item in split_data:
        # Extract human answers
        if 'human_answers' in item and item['human_answers']:
            for answer in item['human_answers']:
                if isinstance(answer, str) and len(answer.strip()) > 20:
                    if max_samples is None or human_count < max_samples:
                        texts.append(answer.strip())
                        labels.append(0)  # Human
                        human_count += 1
        
        # Extract ChatGPT answers
        if 'chatgpt_answers' in item and item['chatgpt_answers']:
            for answer in item['chatgpt_answers']:
                if isinstance(answer, str) and len(answer.strip()) > 20:
                    if max_samples is None or ai_count < max_samples:
                        texts.append(answer.strip())
                        labels.append(1)  # AI
                        ai_count += 1
        
        # Stop if we have enough samples
        if max_samples is not None and human_count >= max_samples and ai_count >= max_samples:
            break
    
    return human_count, ai_count


def load_hc3_dataset(data_path: str) -> Tuple[List[str], List[int]]:
    """
    Load HC3 (Human ChatGPT Comparison Corpus) dataset.
    
    Expected format: JSON files with human and ChatGPT responses
    
    Args:
        data_path: Path to HC3 dataset directory
        
    Returns:
        Tuple of (texts, labels)
    """
    logger.info(f"Loading HC3 dataset from {data_path}")
    
    texts = []
    labels = []
    
    # Look for JSON files in the directory
    json_files = [f for f in os.listdir(data_path) if f.endswith('.json')]
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_path}")
    
    for json_file in json_files:
        json_path = os.path.join(data_path, json_file)
        logger.info(f"Processing {json_file}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # HC3 format typically has 'human_answers' and 'chatgpt_answers'
            if isinstance(data, list):
                for item in data:
                    # Human answers
                    if 'human_answers' in item:
                        for answer in item['human_answers']:
                            if isinstance(answer, dict) and 'text' in answer:
                                texts.append(answer['text'])
                            elif isinstance(answer, str):
                                texts.append(answer)
                            labels.append(0)  # Human
                    
                    # ChatGPT answers
                    if 'chatgpt_answers' in item:
                        for answer in item['chatgpt_answers']:
                            if isinstance(answer, dict) and 'text' in answer:
                                texts.append(answer['text'])
                            elif isinstance(answer, str):
                                texts.append(answer)
                            labels.append(1)  # AI
            
        except Exception as e:
            logger.warning(f"Error processing {json_file}: {e}")
            continue
    
    if not texts:
        raise ValueError("No valid data found in HC3 dataset")
    
    logger.info(f"Loaded {len(texts)} samples from HC3 dataset")
    return texts, labels


def load_csv_dataset(csv_path: str, text_column: str = 'text', label_column: str = 'label') -> Tuple[List[str], List[int]]:
    """
    Load dataset from CSV file.
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of column containing text
        label_column: Name of column containing labels (0=human, 1=AI)
        
    Returns:
        Tuple of (texts, labels)
    """
    logger.info(f"Loading CSV dataset from {csv_path}")
    
    if not PANDAS_AVAILABLE:
        raise ImportError("Pandas is required for CSV loading. Install with: pip install pandas")
    
    try:
        df = pd.read_csv(csv_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].astype(int).tolist()
        
        # Validate labels
        unique_labels = set(labels)
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Labels must be 0 (human) or 1 (AI). Found: {unique_labels}")
        
        logger.info(f"Loaded {len(texts)} samples from CSV")
        logger.info(f"Label distribution: {sum(1 for l in labels if l == 0)} human, {sum(1 for l in labels if l == 1)} AI")
        
        return texts, labels
        
    except Exception as e:
        logger.error(f"Error loading CSV dataset: {e}")
        raise


def extract_features_parallel(texts: List[str], batch_size: int = 100) -> np.ndarray:
    """
    Extract features from texts with progress reporting.
    
    Args:
        texts: List of text samples
        batch_size: Process texts in batches for memory efficiency
        
    Returns:
        Feature matrix
    """
    logger.info(f"Extracting features from {len(texts)} texts...")
    
    features = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_features = []
        
        for text in batch_texts:
            try:
                feat = extract_llm_features(text)
                batch_features.append(feat)
            except Exception as e:
                logger.warning(f"Failed to extract features for text (len={len(text)}): {e}")
                # Use zero features for failed extractions
                batch_features.append(np.zeros(20, dtype=np.float32))
        
        features.extend(batch_features)
        
        batch_num = (i // batch_size) + 1
        logger.info(f"Processed batch {batch_num}/{n_batches}")
    
    feature_matrix = np.array(features)
    logger.info(f"Feature extraction complete. Shape: {feature_matrix.shape}")
    
    return feature_matrix


def train_model(X: np.ndarray, y: np.ndarray, model_params: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, float]]:
    """
    Train the LLM detection model with comprehensive evaluation.
    
    Args:
        X: Feature matrix
        y: Labels
        model_params: Model hyperparameters
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    if model_params is None:
        model_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 4,
            'random_state': 42,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
        }
    
    logger.info("Training LLM detection model...")
    logger.info(f"Model parameters: {model_params}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train model
    model = GradientBoostingClassifier(**model_params)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'n_features': X.shape[1]
    }
    
    # Cross-validation
    logger.info("Performing cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    metrics['cv_auc_mean'] = cv_scores.mean()
    metrics['cv_auc_std'] = cv_scores.std()
    
    # Feature importance
    feature_importance = model.feature_importances_
    top_features = np.argsort(feature_importance)[-10:][::-1]
    
    logger.info("Training Results:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.3f}")
    logger.info(f"  Precision: {metrics['precision']:.3f}")
    logger.info(f"  Recall:    {metrics['recall']:.3f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.3f}")
    logger.info(f"  AUC:       {metrics['auc']:.3f}")
    logger.info(f"  CV AUC:    {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
    logger.info(f"  Top features: {top_features[:5]}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:")
    logger.info(f"  True Negatives:  {cm[0, 0]}")
    logger.info(f"  False Positives: {cm[0, 1]}")
    logger.info(f"  False Negatives: {cm[1, 0]}")
    logger.info(f"  True Positives:  {cm[1, 1]}")
    
    return model, metrics


def save_model_and_metadata(model: Any, metrics: Dict[str, float], model_path: str):
    """
    Save the trained model and its metadata.
    
    Args:
        model: Trained model
        metrics: Training metrics
        model_path: Path to save the model
    """
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save metadata
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    metadata = {
        'metrics': metrics,
        'feature_count': 20,
        'model_type': 'GradientBoostingClassifier',
        'training_date': datetime.now().isoformat(),
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to: {metadata_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train LLM Detection Model')
    parser.add_argument('--dataset', type=str, help='Path to dataset CSV file (for CSV type only)')
    parser.add_argument('--dataset-type', choices=['csv', 'hc3-local', 'hc3'], default='hc3',
                       help='Type of dataset to load')
    parser.add_argument('--text-column', default='text', help='Name of text column in CSV')
    parser.add_argument('--label-column', default='label', help='Name of label column in CSV')
    parser.add_argument('--output', default='models/llm_detector.pkl', help='Output model path')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum samples per class (None for all)')
    parser.add_argument('--hc3-subset', default='all', help='HC3 dataset subset to load')
    parser.add_argument('--test-model', action='store_true', help='Test the trained model after training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LLM Detection Model Training Script")
    print("=" * 60)
    
    # Initialize LLM detection system
    logger.info("Initializing LLM detection system...")
    initialize_llm_detection()
    
    # Load dataset
    try:
        if args.dataset_type == 'csv' and args.dataset:
            texts, labels = load_csv_dataset(args.dataset, args.text_column, args.label_column)
        elif args.dataset_type == 'hc3-local' and args.dataset:
            texts, labels = load_hc3_dataset(args.dataset)
        elif args.dataset_type == 'hc3':
            texts, labels = load_hc3_huggingface_dataset(args.hc3_subset, args.max_samples)
        else:
            raise ValueError("Must specify --dataset path for csv/hc3-local types, or use --dataset-type hc3 for Hugging Face dataset")
        
        # Filter out very short texts
        min_length = 20
        filtered_data = [(t, l) for t, l in zip(texts, labels) if len(t) >= min_length]
        texts, labels = zip(*filtered_data)
        texts, labels = list(texts), list(labels)
        
        logger.info(f"After filtering: {len(texts)} samples")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return False
    
    # Extract features
    try:
        X = extract_features_parallel(texts)
        y = np.array(labels)
        
        # Remove samples with all-zero features
        valid_samples = ~np.all(X == 0, axis=1)
        X = X[valid_samples]
        y = y[valid_samples]
        
        logger.info(f"Valid samples after feature extraction: {len(X)}")
        
        if len(X) < 50:
            raise ValueError("Not enough valid samples for training")
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return False
    
    # Train model
    try:
        model, metrics = train_model(X, y)
        
        if metrics['auc'] < 0.6:
            logger.warning("Model AUC is low (<0.6). Consider collecting more diverse training data.")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False
    
    # Save model
    try:
        save_model_and_metadata(model, metrics, args.output)
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return False
    
    # Test the model if requested
    if args.test_model:
        logger.info("Testing the trained model...")
        try:
            from llm_detection import load_llm_model, detect_llm_likeness
            
            # Load the saved model
            load_llm_model(args.output)
            
            # Test samples
            test_samples = [
                ("Human casual", "hey that's pretty cool! didn't know that was possible lol. gonna try it later"),
                ("Human story", "so yesterday i went to the store and this crazy thing happened. this guy was trying to return a watermelon he'd already eaten!"),
                ("AI formal", "Furthermore, it is essential to consider the comprehensive implications of this systematic approach. Additionally, the implementation demonstrates significant advantages over existing methodologies."),
                ("AI trying casual", "I think this approach is really interesting. Additionally, it offers many benefits. However, there are some challenges to consider as well."),
            ]
            
            logger.info("Test Results:")
            for label, text in test_samples:
                penalty, is_llm = detect_llm_likeness(text, use_ml=True)
                status = "🤖 AI-like" if is_llm else "👤 Human-like"
                logger.info(f"  {label:15} | penalty={penalty:.3f} | {status}")
            
        except Exception as e:
            logger.warning(f"Model testing failed: {e}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Model saved to: {args.output}")
    print(f"AUC Score: {metrics['auc']:.3f}")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)