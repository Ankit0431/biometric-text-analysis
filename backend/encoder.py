"""
Text encoder wrapper for biometric text authentication.

This module provides two encoding options:
1. Transformer-based encoder (xlm-roberta-base) with projection head
2. Fine-tuned sentence transformer for authorship embedding

Features:
- Mean-pooling of last hidden states (transformer mode)
- Linear projection to 512 dimensions (transformer mode) 
- Sentence transformer fine-tuned for authorship (sentence mode)
- L2 normalization
- Micro-batching for efficiency
- Optional quantization for faster inference
- Automatic fallback between modes
"""
import os
import time
import threading
from typing import List, Optional, Tuple
from queue import Queue, Empty
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


# Configuration
DEFAULT_MODEL_NAME = "xlm-roberta-base"
DEFAULT_SENTENCE_MODEL = "paraphrase-MiniLM-L6-v2"
TARGET_DIM = 512
MAX_LENGTH = 256  # Maximum sequence length
BATCH_TIMEOUT_MS = 20  # Micro-batch timeout in milliseconds
MAX_BATCH_SIZE = 32  # Maximum batch size

# Paths for model storage
MODELS_DIR = Path(__file__).parent / "models"
PROJECTION_WEIGHTS_DIR = Path(__file__).parent / "model_weights"
PROJECTION_WEIGHTS_FILE = PROJECTION_WEIGHTS_DIR / "projection.pt"
AUTHORSHIP_ENCODER_PATH = MODELS_DIR / "authorship_encoder"
SENTENCE_ENCODER_PATH = MODELS_DIR / "sentence_encoder"


@dataclass
class EncoderRequest:
    """Request for encoding text."""
    texts: List[str]
    result_queue: Queue


class TextEncoder:
    """
    Dual-mode text encoder supporting both transformer and sentence transformer models.

    Modes:
    1. Transformer mode: xlm-roberta-base with projection head
    2. Sentence transformer mode: Fine-tuned sentence transformer for authorship
    
    Automatically selects the best available model with fallback support.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        target_dim: int = TARGET_DIM,
        device: Optional[str] = None,
        use_quantization: bool = False,
        prefer_sentence_transformer: bool = True,
    ):
        """
        Initialize the encoder with automatic model selection.

        Args:
            model_name: Hugging Face model name (transformer mode)
            target_dim: Target embedding dimension
            device: Device to use ('cpu', 'cuda', or None for auto)
            use_quantization: Whether to use dynamic int8 quantization
            prefer_sentence_transformer: Whether to prefer sentence transformer if available
        """
        self.model_name = model_name
        self.target_dim = target_dim
        self.use_quantization = use_quantization
        self.prefer_sentence_transformer = prefer_sentence_transformer

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize encoder mode
        self.mode = None
        self.sentence_model = None
        self.tokenizer = None
        self.base_model = None
        self.projection = None
        self.hidden_size = None

        # Try to load sentence transformer first if preferred
        if prefer_sentence_transformer and SENTENCE_TRANSFORMERS_AVAILABLE:
            if self._load_sentence_transformer():
                self.mode = "sentence_transformer"
                print(f"✅ Using sentence transformer mode")
                return

        # Fallback to transformer mode
        self._load_transformer_model()
        self.mode = "transformer"
        print(f"✅ Using transformer mode: {model_name} on {self.device}")

    def _load_sentence_transformer(self) -> bool:
        """
        Try to load fine-tuned sentence transformer model.
        
        Returns:
            True if successful, False otherwise
        """
        # Check for authorship encoder first, then sentence encoder, then default
        search_paths = [
            AUTHORSHIP_ENCODER_PATH,
            SENTENCE_ENCODER_PATH,
        ]
        
        for model_path in search_paths:
            if model_path.exists():
                try:
                    print(f"Loading sentence transformer from: {model_path}")
                    if SentenceTransformer is not None:
                        self.sentence_model = SentenceTransformer(str(model_path), device=str(self.device))
                        
                        # Test the model
                        test_embedding = self.sentence_model.encode("Test sentence")
                        
                        # Check if embedding dimension matches target
                        if len(test_embedding) != self.target_dim:
                            print(f"⚠️  Sentence model embedding dim {len(test_embedding)} != target {self.target_dim}")
                            # Could add padding/truncation here if needed
                        
                        print(f"✅ Loaded sentence transformer: embedding dim {len(test_embedding)}")
                        return True
                    
                except Exception as e:
                    print(f"❌ Failed to load sentence transformer from {model_path}: {e}")
                    continue
        
        # Try loading default sentence transformer
        if not self.sentence_model and SentenceTransformer is not None:
            try:
                print(f"Loading default sentence transformer: {DEFAULT_SENTENCE_MODEL}")
                self.sentence_model = SentenceTransformer(DEFAULT_SENTENCE_MODEL, device=str(self.device))
                test_embedding = self.sentence_model.encode("Test sentence")
                print(f"✅ Loaded default sentence transformer: embedding dim {len(test_embedding)}")
                return True
            except Exception as e:
                print(f"❌ Failed to load default sentence transformer: {e}")
        
        return False

    def _load_transformer_model(self):
        """Load transformer model with projection head."""
        print(f"Loading transformer model: {self.model_name}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base_model = AutoModel.from_pretrained(self.model_name)

        # Get model hidden size
        self.hidden_size = self.base_model.config.hidden_size

        # Create projection head
        self.projection = nn.Linear(self.hidden_size, self.target_dim)
        
        # Try to load existing projection weights, otherwise initialize randomly
        if self._load_projection_weights():
            print(f"✅ Loaded existing projection weights from {PROJECTION_WEIGHTS_FILE}")
        else:
            print(f"⚠️  No existing projection weights found. Initializing randomly.")
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
            # Save the newly initialized weights
            self._save_projection_weights()
            print(f"💾 Saved new projection weights to {PROJECTION_WEIGHTS_FILE}")

        # Move to device
        self.base_model.to(self.device)
        self.projection.to(self.device)

        # Set to evaluation mode
        self.base_model.eval()
        self.projection.eval()

        # Apply quantization if requested (CPU only)
        if self.use_quantization and self.device.type == 'cpu':
            print("Applying dynamic int8 quantization...")
            self.base_model = torch.quantization.quantize_dynamic(
                self.base_model,
                {nn.Linear},
                dtype=torch.qint8
            )

        # Micro-batching state
        self.request_queue: Queue = Queue()
        self.batch_thread: Optional[threading.Thread] = None
        self.batch_thread_running = False

        print(f"Encoder initialized: {self.hidden_size} -> {self.target_dim} dims")

    def _save_projection_weights(self) -> bool:
        """
        Save projection layer weights to disk.
        
        Returns:
            True if successful, False otherwise
        """
        if self.projection is None:
            return False
            
        try:
            # Create directory if it doesn't exist
            PROJECTION_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save projection weights
            torch.save({
                'weight': self.projection.weight.cpu(),
                'bias': self.projection.bias.cpu(),
                'hidden_size': self.hidden_size,
                'target_dim': self.target_dim,
                'model_name': self.model_name,
            }, PROJECTION_WEIGHTS_FILE)
            
            return True
        except Exception as e:
            print(f"❌ Error saving projection weights: {e}")
            return False
    
    def _load_projection_weights(self) -> bool:
        """
        Load projection layer weights from disk if they exist.
        
        Returns:
            True if weights were loaded, False otherwise
        """
        if self.projection is None:
            return False
            
        try:
            if not PROJECTION_WEIGHTS_FILE.exists():
                return False
            
            # Load weights
            checkpoint = torch.load(PROJECTION_WEIGHTS_FILE, map_location='cpu')
            
            # Validate dimensions match
            if checkpoint['hidden_size'] != self.hidden_size or checkpoint['target_dim'] != self.target_dim:
                print(f"⚠️  Projection dimensions mismatch. Expected ({self.hidden_size}, {self.target_dim}), "
                      f"got ({checkpoint['hidden_size']}, {checkpoint['target_dim']}). Ignoring saved weights.")
                return False
            
            # Load weights into projection layer
            self.projection.weight.data = checkpoint['weight'].to(self.device)
            self.projection.bias.data = checkpoint['bias'].to(self.device)
            
            return True
        except Exception as e:
            print(f"❌ Error loading projection weights: {e}")
            return False

    def mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean-pool hidden states with attention mask.

        Args:
            hidden_states: Shape (batch, seq_len, hidden_size)
            attention_mask: Shape (batch, seq_len)

        Returns:
            Pooled embeddings of shape (batch, hidden_size)
        """
        # Expand attention mask to match hidden states
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

        # Sum embeddings
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)

        # Sum attention mask (count tokens)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        # Calculate mean
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings

    def l2_normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        L2 normalize embeddings.

        Args:
            embeddings: Shape (batch, dim)

        Returns:
            L2-normalized embeddings
        """
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-9)  # Avoid division by zero
        return embeddings / norms

    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts using the appropriate model.

        Args:
            texts: List of text strings

        Returns:
            Embeddings of shape (n, target_dim) with L2 norm ≈ 1
        """
        if not texts:
            return np.zeros((0, self.target_dim), dtype=np.float32)

        if self.mode == "sentence_transformer" and self.sentence_model is not None:
            return self._encode_with_sentence_transformer(texts)
        elif self.mode == "transformer":
            return self._encode_with_transformer(texts)
        else:
            raise RuntimeError("No valid encoder model loaded")

    def _encode_with_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Encode texts using sentence transformer."""
        if self.sentence_model is None:
            raise RuntimeError("Sentence transformer not loaded")
            
        # Encode with sentence transformer
        embeddings = self.sentence_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize
        )
        
        # Ensure correct dtype
        embeddings = embeddings.astype(np.float32)
        
        # Handle dimension mismatch by padding or truncating
        if embeddings.shape[1] != self.target_dim:
            if embeddings.shape[1] < self.target_dim:
                # Pad with zeros
                padding = np.zeros((embeddings.shape[0], self.target_dim - embeddings.shape[1]), dtype=np.float32)
                embeddings = np.concatenate([embeddings, padding], axis=1)
            else:
                # Truncate
                embeddings = embeddings[:, :self.target_dim]
            
            # Re-normalize after padding/truncating
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-9)  # Avoid division by zero
            embeddings = embeddings / norms
        
        return embeddings

    def _encode_with_transformer(self, texts: List[str]) -> np.ndarray:
        """Encode texts using transformer model with projection."""
        if self.tokenizer is None or self.base_model is None or self.projection is None:
            raise RuntimeError("Transformer model not properly loaded")

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Mean pool
        hidden_states = outputs.last_hidden_state
        pooled = self.mean_pool(hidden_states, attention_mask)

        # Project to target dimension
        projected = self.projection(pooled)

        # L2 normalize
        normalized = self.l2_normalize(projected)

        # Convert to numpy
        embeddings = normalized.cpu().numpy().astype(np.float32)

        return embeddings

    def encode(
        self,
        texts: List[str],
        lang: Optional[str] = None,
        use_batching: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings
            lang: Language code (unused for now, for future language-specific models)
            use_batching: Whether to use micro-batching (currently not implemented)

        Returns:
            Embeddings of shape (n, target_dim) with L2 norm ≈ 1
        """
        return self.encode_batch(texts)

    def start_batch_processor(self):
        """
        Start the micro-batching background thread.

        This collects requests for BATCH_TIMEOUT_MS and processes them as a batch.
        """
        if self.batch_thread_running:
            return

        self.batch_thread_running = True
        self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.batch_thread.start()
        print("Batch processor started")

    def stop_batch_processor(self):
        """Stop the micro-batching background thread."""
        self.batch_thread_running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=1.0)
        print("Batch processor stopped")

    def _batch_processor(self):
        """
        Background thread that processes batched requests.

        Collects requests for up to BATCH_TIMEOUT_MS milliseconds,
        then processes them as a single batch.
        """
        while self.batch_thread_running:
            requests: List[EncoderRequest] = []
            deadline = time.time() + (BATCH_TIMEOUT_MS / 1000.0)

            # Collect requests until timeout or max batch size
            while time.time() < deadline and len(requests) < MAX_BATCH_SIZE:
                timeout_remaining = max(0, deadline - time.time())
                try:
                    req = self.request_queue.get(timeout=timeout_remaining)
                    requests.append(req)
                except Empty:
                    break

            # Process batch if we have requests
            if requests:
                # Flatten all texts
                all_texts = []
                text_counts = []
                for req in requests:
                    text_counts.append(len(req.texts))
                    all_texts.extend(req.texts)

                # Encode all texts
                embeddings = self.encode_batch(all_texts)

                # Split results and send to requesters
                start_idx = 0
                for req, count in zip(requests, text_counts):
                    result = embeddings[start_idx:start_idx + count]
                    req.result_queue.put(result)
                    start_idx += count

    def encode_with_batching(self, texts: List[str], timeout: float = 1.0) -> np.ndarray:
        """
        Encode texts using micro-batching.

        Args:
            texts: List of text strings
            timeout: Timeout in seconds

        Returns:
            Embeddings of shape (n, target_dim)
        """
        result_queue = Queue()
        request = EncoderRequest(texts=texts, result_queue=result_queue)

        # Submit request
        self.request_queue.put(request)

        # Wait for result
        try:
            result = result_queue.get(timeout=timeout)
            return result
        except Empty:
            raise TimeoutError(f"Encoding timed out after {timeout}s")


# Global encoder instance
_encoder: Optional[TextEncoder] = None
_encoder_lock = threading.Lock()


def get_encoder(
    model_name: str = DEFAULT_MODEL_NAME,
    target_dim: int = TARGET_DIM,
    device: Optional[str] = None,
    use_quantization: bool = False,
) -> TextEncoder:
    """
    Get or create global encoder instance.

    Args:
        model_name: Hugging Face model name
        target_dim: Target embedding dimension
        device: Device to use
        use_quantization: Whether to use quantization

    Returns:
        TextEncoder instance
    """
    global _encoder

    with _encoder_lock:
        if _encoder is None:
            _encoder = TextEncoder(
                model_name=model_name,
                target_dim=target_dim,
                device=device,
                use_quantization=use_quantization,
            )

    return _encoder


def encode(
    texts: List[str],
    lang: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> np.ndarray:
    """
    Convenience function to encode texts.

    Args:
        texts: List of text strings
        lang: Language code (unused for now)
        model_name: Model to use

    Returns:
        Embeddings of shape (n, target_dim) with L2 norm ≈ 1
    """
    encoder = get_encoder(model_name=model_name)
    return encoder.encode(texts, lang=lang)
