# LLM Detection Model Training

This directory contains comprehensive training scripts for the hybrid LLM detection system. The training scripts can work with various dataset formats and create machine learning models to enhance AI text detection accuracy.

## Quick Start

### Option 1: Simple Training Script (Recommended)
```bash
# Install required dependencies
pip install scikit-learn

# Train with generated sample data
python simple_train_llm_detector.py
```

### Option 2: Advanced Training Script
```bash
# Install all dependencies
pip install scikit-learn pandas

# Train with sample data
python train_llm_detector.py --dataset-type sample --n-samples 1000 --test-model

# Train with CSV dataset
python train_llm_detector.py --dataset path/to/dataset.csv --dataset-type csv --test-model

# Train with HC3 dataset
python train_llm_detector.py --dataset path/to/hc3/ --dataset-type hc3 --test-model
```

## Training Scripts

### 1. `simple_train_llm_detector.py`
- **Purpose**: Easy-to-use training script with enhanced sample data generation
- **Dependencies**: scikit-learn (automatically detected)
- **Features**:
  - Generates 1000 diverse sample texts (500 human, 500 AI)
  - Uses realistic human and AI writing patterns
  - Automatic model saving and testing
  - Progress reporting and evaluation metrics

### 2. `train_llm_detector.py`
- **Purpose**: Advanced training script with multiple dataset format support
- **Dependencies**: scikit-learn, pandas (optional for CSV)
- **Features**:
  - Support for CSV, HC3, and generated datasets
  - Comprehensive evaluation with cross-validation
  - Feature importance analysis
  - Configurable hyperparameters
  - Detailed logging and progress tracking

## Dataset Formats

### CSV Format
```csv
text,label
"hey what's up? just chilling here",0
"Furthermore, this approach demonstrates significant advantages.",1
```
- `text`: Text content
- `label`: 0 = human, 1 = AI

### HC3 Format
JSON files with structure:
```json
[
  {
    "human_answers": ["Human written text..."],
    "chatgpt_answers": ["AI generated text..."]
  }
]
```

### Sample Data Generation
Both scripts can generate synthetic training data:
- **Human patterns**: Casual, conversational, with informal language
- **AI patterns**: Formal, structured, with transitional phrases

## Model Architecture

The training creates a **GradientBoostingClassifier** with:
- **Features**: 20-dimensional linguistic feature vectors
- **Architecture**: Ensemble of decision trees
- **Training**: 80/20 train/test split with cross-validation
- **Output**: Probability scores for AI likelihood

### Feature Vector (20 dimensions)
1. Average sentence length
2. Average word length  
3. Vocabulary richness (unique words / total words)
4. Formal word frequency
5. Transition phrase frequency
6. Question mark frequency
7. Exclamation mark frequency
8. Comma frequency
9. Semicolon frequency
10. Adverb frequency (words ending in 'ly')
11. Superlative frequency ('most', 'best', etc.)
12. Passive voice indicators
13. First person pronoun frequency
14. Complex sentence ratio
15. Repetitive phrase detection
16. Perplexity score (if transformers available)
17. Part-of-speech diversity (if NLTK available)
18. Average dependency depth (if NLTK available)
19. Readability score approximation
20. Discourse marker frequency

## Training Process

1. **Data Loading**: Load from CSV, HC3, or generate samples
2. **Feature Extraction**: Extract 20-dimensional feature vectors
3. **Data Splitting**: 80% training, 20% testing
4. **Model Training**: GradientBoostingClassifier with hyperparameters:
   - `n_estimators`: 200
   - `learning_rate`: 0.05
   - `max_depth`: 4
   - Early stopping with validation
5. **Evaluation**: Accuracy, Precision, Recall, F1, AUC
6. **Model Saving**: Save to `models/llm_detector.pkl`

## Integration

Once trained, the model integrates automatically with the hybrid detection system:

```python
from llm_detection import detect_llm_likeness, load_llm_model

# Load your trained model
load_llm_model("models/llm_detector.pkl")

# Use hybrid detection (ML + heuristics)
penalty, is_llm = detect_llm_likeness("Your text here", use_ml=True)
```

## Performance Expectations

With sample data:
- **Accuracy**: 85-95%
- **AUC**: 0.90-0.98
- **Precision**: 85-95%
- **Recall**: 85-95%

With real-world data:
- **Accuracy**: 75-90% (depends on data quality)
- **AUC**: 0.80-0.95
- **Note**: Performance varies significantly with dataset diversity

## Hyperparameter Tuning

Customize the model in `train_llm_detector.py`:

```python
model_params = {
    'n_estimators': 200,      # Number of trees
    'learning_rate': 0.05,    # Learning rate
    'max_depth': 4,           # Tree depth
    'random_state': 42,       # Reproducibility
    'validation_fraction': 0.1, # Early stopping
    'n_iter_no_change': 10,   # Early stopping patience
}
```

## Troubleshooting

### Missing Dependencies
```bash
# Core requirements
pip install scikit-learn

# Full features
pip install scikit-learn pandas nltk transformers torch
```

### Low Model Performance
1. **Increase dataset size**: More diverse training data
2. **Balance classes**: Equal human/AI samples
3. **Feature engineering**: Add domain-specific features
4. **Hyperparameter tuning**: Adjust model parameters
5. **Data quality**: Remove low-quality samples

### Memory Issues
- Reduce batch size in feature extraction
- Use smaller datasets for testing
- Process data in chunks

## Example Usage

### Train and Test
```python
# Generate and train with sample data
texts, labels = generate_enhanced_sample_data(1000)
metrics = train_llm_detector(texts, labels, "models/llm_detector.pkl")

# Load and test the model
load_llm_model("models/llm_detector.pkl")
penalty, is_llm = detect_llm_likeness("Test text", use_ml=True)
```

### Production Pipeline
```python
# 1. Collect labeled data (human vs AI texts)
# 2. Train model with your data
# 3. Evaluate performance
# 4. Deploy model
# 5. Monitor and retrain as needed
```

## Best Practices

1. **Data Quality**: Use high-quality, diverse training data
2. **Balanced Classes**: Ensure equal representation of human/AI texts
3. **Regular Retraining**: Update model with new data
4. **Performance Monitoring**: Track accuracy over time
5. **Domain Adaptation**: Train on domain-specific data when possible
6. **Feature Engineering**: Add domain-specific linguistic features
7. **Cross-Validation**: Use k-fold validation for robust evaluation
8. **Model Versioning**: Keep track of model versions and performance

## References

- **HC3 Dataset**: [Human ChatGPT Comparison Corpus](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)
- **Feature Engineering**: Linguistic analysis techniques for AI detection
- **Gradient Boosting**: Ensemble learning for text classification
- **Hybrid Detection**: Combining ML classification with heuristic rules