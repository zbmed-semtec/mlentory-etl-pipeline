import time
from typing import List, Dict, Any, Optional
import numpy as np

# We'll use sentence-transformers library for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Run: pip install sentence-transformers")

from .config import config

class EmbeddingService:
    """
    Service for generating text embeddings using sentence transformers.
    
    This class handles:
    - Loading the embedding model
    - Converting text to vectors
    - Batch processing for efficiency
    - Error handling and logging
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model to use. 
                       If None, uses the model from config.
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = None
        self.embedding_dimension = config.EMBEDDING_DIMENSION
        self.max_sequence_length = config.MAX_SEQUENCE_LENGTH
        
        # Performance tracking
        self.total_embeddings_generated = 0
        self.total_processing_time = 0.0
        
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """
        Load the sentence transformer model.
        
        This method:
        1. Downloads the model if not already cached
        2. Loads it into memory
        3. Validates the model works
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers library not available. "
                "Install it with: pip install sentence-transformers"
            )
        
        try:
            start_time = time.time()
            
            # Load the model (this will download it if not cached)
            self.model = SentenceTransformer(self.model_name)
            
            load_time = time.time() - start_time
            
            # Validate the model works
            self._validate_model()
            
        except Exception as e:
            print(f"❌ Failed to load model {self.model_name}: {e}")
            raise
    
    def _validate_model(self):
        """
        Validate that the loaded model works correctly.
        
        This method:
        1. Tests the model with a simple sentence
        2. Checks the output dimensions
        3. Ensures the model is ready for use
        """
        try:
            
            # Test with a simple sentence
            test_text = "This is a test sentence for validation"
            test_embedding = self.model.encode([test_text])
            
            # Check dimensions
            actual_dimension = len(test_embedding[0])
            if actual_dimension != self.embedding_dimension:
                print(
                    f"⚠️  Expected dimension {self.embedding_dimension}, "
                    f"but got {actual_dimension}. Updating config..."
                )
                self.embedding_dimension = actual_dimension
            
            
        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            raise
    
    def encode_text(self, text: str) -> List[float]:
        """
        Convert a single text string to an embedding vector.
        
        Args:
            text: The text to convert to embedding
            
        Returns:
            List[float]: The embedding vector (list of numbers)
            
        Example:
            >>> service = EmbeddingService()
            >>> vector = service.encode_text("BERT model for language understanding")
            >>> print(len(vector))  # Should be 384
            >>> print(vector[:5])   # First 5 numbers: [0.1, 0.8, 0.3, 0.2, 0.9]
        """
        if not text or not text.strip():
            return [0.0] * self.embedding_dimension
        
        try:
            start_time = time.time()
            
            # Clean and prepare the text
            cleaned_text = self._clean_text(text)
            
            # Generate embedding
            embedding = self.model.encode([cleaned_text])
            vector = embedding[0].tolist()  # Convert numpy array to list
            
            # Track performance
            processing_time = time.time() - start_time
            self.total_embeddings_generated += 1
            self.total_processing_time += processing_time
            
            
            return vector
            
        except Exception as e:
            print(f"❌ Failed to encode text: {e}")
            raise
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts to embeddings in batch (more efficient).
        
        Args:
            texts: List of texts to convert
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Example:
            >>> service = EmbeddingService()
            >>> texts = [
            ...     "BERT model for language understanding",
            ...     "GPT model for text generation",
            ...     "ResNet model for image classification"
            ... ]
            >>> vectors = service.encode_batch(texts)
            >>> print(len(vectors))  # Should be 3
            >>> print(len(vectors[0]))  # Should be 384
        """
        if not texts:
            return []
        
        try:
            start_time = time.time()
            
            # Clean all texts (keep all texts, even empty ones)
            cleaned_texts = [self._clean_text(text) for text in texts]
            
            if not cleaned_texts:
                return []
            
            # Generate embeddings in batch
            embeddings = self.model.encode(cleaned_texts)
            vectors = [embedding.tolist() for embedding in embeddings]
            
            # Track performance
            processing_time = time.time() - start_time
            self.total_embeddings_generated += len(vectors)
            self.total_processing_time += processing_time
            
            
            return vectors
            
        except Exception as e:
            print(f"❌ Failed to encode batch: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and prepare text for embedding.
        
        Args:
            text: Raw text to clean
            
        Returns:
            str: Cleaned text ready for embedding
        """
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = text.strip()
        
        # Truncate if too long (models have limits)
        if len(cleaned) > self.max_sequence_length:
            cleaned = cleaned[:self.max_sequence_length]
        
        return cleaned
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dict: Model information including name, dimensions, performance stats
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_sequence_length": self.max_sequence_length,
            "total_embeddings_generated": self.total_embeddings_generated,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": (
                self.total_processing_time / max(1, self.total_embeddings_generated)
            ),
            "embeddings_per_second": (
                self.total_embeddings_generated / max(0.001, self.total_processing_time)
            )
        }
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            int: The embedding dimension (e.g., 384)
        """
        return self.embedding_dimension
