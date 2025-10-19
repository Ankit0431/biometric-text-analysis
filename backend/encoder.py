"""
Text encoder wrapper for biometric text authentication.

This module wraps a pretrained Hugging Face transformer model (xlm-roberta-base)
and provides:
- Mean-pooling of last hidden states
- Linear projection to 512 dimensions
- L2 normalization
- Micro-batching for efficiency
- Optional quantization for faster inference
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


# Configuration
DEFAULT_MODEL_NAME = "xlm-roberta-base"
TARGET_DIM = 512
MAX_LENGTH = 256  # Maximum sequence length
BATCH_TIMEOUT_MS = 20  # Micro-batch timeout in milliseconds
MAX_BATCH_SIZE = 32  # Maximum batch size

# Path to save/load projection weights
PROJECTION_WEIGHTS_DIR = Path(__file__).parent / "model_weights"
PROJECTION_WEIGHTS_FILE = PROJECTION_WEIGHTS_DIR / "projection.pt"


@dataclass
class EncoderRequest:
    """Request for encoding text."""
    texts: List[str]
    result_queue: Queue


class TextEncoder:
    """
    Text encoder with projection head and L2 normalization.

    Uses xlm-roberta-base (or similar) with:
    - Mean-pooling of last hidden states
    - Linear projection to 512 dims
    - L2 normalization
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        target_dim: int = TARGET_DIM,
        device: Optional[str] = None,
        use_quantization: bool = False,
    ):
        """
        Initialize the encoder.

        Args:
            model_name: Hugging Face model name
            target_dim: Target embedding dimension
            device: Device to use ('cpu', 'cuda', or None for auto)
            use_quantization: Whether to use dynamic int8 quantization
        """
        self.model_name = model_name
        self.target_dim = target_dim
        self.use_quantization = use_quantization

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Loading encoder model: {model_name} on {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)

        # Get model hidden size
        self.hidden_size = self.base_model.config.hidden_size

        # Create projection head
        self.projection = nn.Linear(self.hidden_size, target_dim)
        
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
        if use_quantization and self.device.type == 'cpu':
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

        print(f"Encoder initialized: {self.hidden_size} -> {target_dim} dims")

    def _save_projection_weights(self) -> bool:
        """
        Save projection layer weights to disk.
        
        Returns:
            True if successful, False otherwise
        """
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
        Encode a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            Embeddings of shape (n, target_dim) with L2 norm ≈ 1
        """
        if not texts:
            return np.zeros((0, self.target_dim), dtype=np.float32)

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
