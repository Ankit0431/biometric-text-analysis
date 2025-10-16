"""
Training script for ML-based LLM detector (future enhancement).

This script is a placeholder for training a more sophisticated detector
using machine learning. The current implementation uses heuristics.

For production, this could train a small MLP or logistic regression model on:
- Human-written text samples
- LLM-generated text samples
- Features extracted by llm_detector._extract_features()

The trained model would replace the heuristic scoring with learned weights.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from llm_detector import _extract_features


def load_training_data(human_file: str, llm_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from files.

    Args:
        human_file: Path to file with human-written samples (one per line)
        llm_file: Path to file with LLM-generated samples (one per line)

    Returns:
        Tuple of (features, labels) where:
        - features: (n_samples, n_features) array
        - labels: (n_samples,) array (0=human, 1=LLM)
    """
    print(f"Loading human samples from {human_file}...")
    with open(human_file, 'r', encoding='utf-8') as f:
        human_texts = [line.strip() for line in f if line.strip()]

    print(f"Loading LLM samples from {llm_file}...")
    with open(llm_file, 'r', encoding='utf-8') as f:
        llm_texts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(human_texts)} human and {len(llm_texts)} LLM samples")

    # Extract features
    print("Extracting features...")
    all_texts = human_texts + llm_texts
    labels = np.array([0] * len(human_texts) + [1] * len(llm_texts))

    feature_dicts = [_extract_features(text) for text in all_texts]

    # Convert to numpy array
    feature_names = list(feature_dicts[0].keys())
    features = np.array([[fd[name] for name in feature_names] for fd in feature_dicts])

    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature names: {feature_names}")

    return features, labels, feature_names


def train_model(features: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Train a simple logistic regression model.

    For production, this could be replaced with:
    - MLP (small neural network)
    - Random Forest
    - Gradient Boosting
    - Fine-tuned transformer

    Args:
        features: (n_samples, n_features) feature matrix
        labels: (n_samples,) binary labels

    Returns:
        Dict with model weights and metadata
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("ERROR: scikit-learn not installed. Install with: pip install scikit-learn")
        sys.exit(1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("\nTraining logistic regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(f"\nTraining accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'LLM']))

    # Package model
    model_dict = {
        'weights': model.coef_[0].tolist(),
        'bias': float(model.intercept_[0]),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'test_roc_auc': roc_auc_score(y_test, y_proba),
    }

    return model_dict


def save_model(model_dict: Dict, feature_names: List[str], output_path: str):
    """Save trained model to JSON file."""
    model_dict['feature_names'] = feature_names
    model_dict['version'] = '1.0'
    model_dict['description'] = 'Logistic regression LLM detector'

    with open(output_path, 'w') as f:
        json.dump(model_dict, f, indent=2)

    print(f"\nModel saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train ML-based LLM detector')
    parser.add_argument('--human', type=str, required=True,
                      help='Path to file with human-written samples')
    parser.add_argument('--llm', type=str, required=True,
                      help='Path to file with LLM-generated samples')
    parser.add_argument('--output', type=str, default='llm_detector_model.json',
                      help='Output path for trained model')

    args = parser.parse_args()

    # Load data
    features, labels, feature_names = load_training_data(args.human, args.llm)

    # Train model
    model_dict = train_model(features, labels)

    # Save model
    save_model(model_dict, feature_names, args.output)

    print("\n✅ Training complete!")
    print("\nTo use this model in production:")
    print("1. Update llm_detector.py to load and use this model")
    print("2. Replace heuristic scoring with model predictions")
    print("3. Add model file to backend/data/ directory")


if __name__ == '__main__':
    # Example usage
    print("=" * 70)
    print("LLM Detector Training Script")
    print("=" * 70)
    print("\nThis is a placeholder for future ML-based detector training.")
    print("Current implementation uses heuristic-based detection.")
    print("\nTo train a model, you need:")
    print("  1. human_samples.txt - Human-written text (one sample per line)")
    print("  2. llm_samples.txt - LLM-generated text (one sample per line)")
    print("\nExample:")
    print("  python train_llm_detector.py --human data/human.txt --llm data/llm.txt")
    print("=" * 70)
    print()

    # Check if being run directly
    if len(sys.argv) > 1:
        main()
    else:
        print("Run with --help for usage information.")
