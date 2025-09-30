"""
Configuration settings for Vector Similarity POC
This file contains all the settings we need for our vector similarity system.
Think of it as the "control panel" - we can change settings here without touching the code.
"""

import os
from typing import List, Dict, Any

class VectorSimilarityConfig:
    """
    Configuration class for vector similarity system.
    
    This class holds all the settings we need:
    - Which embedding model to use
    - How to connect to Elasticsearch
    - What indices to work with
    - Search parameters
    """
    
    # ==================== EMBEDDING MODEL SETTINGS ====================
    # The embedding model we'll use to convert text to vectors
    # We're starting with a simple, fast model that's easy to replace

    # EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384D, Very Fast
    # EMBEDDING_MODEL = "sentence-transfEMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2" ormers/all-MiniLM-L12-v2"  # 384D, Better quality
     # 768D, Best quality

    EMBEDDING_DIMENSION = 768  # This model produces 768-dimensional vectors
    MAX_SEQUENCE_LENGTH = 512  # Maximum text length the model can handle
    
    # ==================== MULTI-MODEL EMBEDDING SETTINGS ====================
    # Configuration for multiple embedding models
    MULTI_MODEL_CONFIGS = {
        "mpnet": {
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "dimension": 768,
            "index_suffix": "mpnet",
            "description": "MPNet - High quality general purpose embeddings"
        },
        "e5": {
            "model_name": "intfloat/e5-base-v2", 
            "dimension": 768,
            "index_suffix": "e5",
            "description": "E5 - Optimized for retrieval tasks"
        },
        "bge": {
            "model_name": "BAAI/bge-base-en-v1.5",
            "dimension": 768,
            "index_suffix": "bge", 
            "description": "BGE - Balanced general embeddings"
        }
    }

    # ==================== ELASTICSEARCH SETTINGS ====================
    # How to connect to our existing Elasticsearch container
    ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")  # Use localhost when running outside Docker
    ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
    ELASTICSEARCH_SCHEME = "http"
    
    # Authentication (we have security disabled, but this is for future use)
    ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", "elastic")
    ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "changeme")
    
    # ==================== INDEX SETTINGS ====================
    # Names of our existing indices (where the original data is stored)
    SOURCE_INDICES = [
        "hf_models",      # HuggingFace models
        "openml_models",  # OpenML models  
        "ai4life_models"  # AI4Life models
    ]
    
    # Names of our new vector indices (where we'll store the vector embeddings)
    VECTOR_INDEX_PREFIX = "vector_"
    VECTOR_INDICES = {
        "hf_models": "vector_multi_hf_models",  # Use multi-vector index
        "openml_models": "vector_openml_models", 
        "ai4life_models": "vector_ai4life_models"
    }
    
    # ==================== SEARCH SETTINGS ====================
    # How many results to return by default
    DEFAULT_SEARCH_SIZE = 10
    
    # Maximum number of results we can return
    MAX_SEARCH_SIZE = 100
    
    # Similarity threshold (0.0 = return everything, 1.0 = only identical matches)
    SIMILARITY_THRESHOLD = 0.3
    
    # Vector search candidates (higher = better quality, slower)
    VECTOR_SEARCH_CANDIDATES = 100
    
    # ==================== TEXT PROCESSING SETTINGS ====================
    # Which fields from the original models we want to include in our vector search
    # These fields will be combined into one text that gets converted to a vector
    EMBEDDING_FIELDS = [
        "name",           # Model name
        "description",    # Model description
        "mlTask",         # ML tasks (like "text-classification", "image-generation")
        "keywords",       # Keywords/tags
        "sharedBy"        # Who created/shared the model
    ]
    
    # Field weights for search and embedding generation
    # Higher weights = more important in search results
    FIELD_WEIGHTS = {
        "name": 3.0,        # Model name is most important
        "description": 2.5,  # Description is very important
        "mlTask": 2.0,      # ML tasks are important
        "keywords": 1.5,    # Keywords are somewhat important
        "sharedBy": 0.5     # Creator is least important
    }
    
    # How to combine these fields into one searchable text
    FIELD_SEPARATOR = " | "  # Separator between fields
    
    # ==================== BATCH PROCESSING SETTINGS ====================
    # How many models to process at once (for efficiency)
    BATCH_SIZE = 100
    
    # How many models to test with initially (for quick testing)
    SAMPLE_SIZE = 50
    
    # ==================== TESTING SETTINGS ====================
    # Test queries to validate our system works
    TEST_QUERIES = [
        "BERT model for language understanding",
        "computer vision image classification", 
        "transformer model for NLP",
        "deep learning model for text generation",
        "neural network for image recognition"
    ]
    
    # ==================== LOGGING SETTINGS ====================
    # How detailed our logs should be
    LOG_LEVEL = "INFO"
    
    # ==================== HELPER METHODS ====================
    
    @classmethod
    def get_elasticsearch_url(cls) -> str:
        """
        Get the full Elasticsearch URL for connection.
        
        Returns:
            str: Complete URL like "http://elastic_db:9200"
        """
        return f"{cls.ELASTICSEARCH_SCHEME}://{cls.ELASTICSEARCH_HOST}:{cls.ELASTICSEARCH_PORT}"
    
    @classmethod
    def get_vector_index_name(cls, source_index: str) -> str:
        """
        Get the vector index name for a given source index.
        
        Args:
            source_index: The original index name (e.g., "hf_models")
            
        Returns:
            str: The vector index name (e.g., "vector_hf_models")
        """
        return cls.VECTOR_INDICES.get(source_index, f"vector_{source_index}")
    
    @classmethod
    def get_embedding_model_info(cls) -> Dict[str, Any]:
        """
        Get information about the current embedding model.
        
        Returns:
            Dict: Model information including name, dimensions, etc.
        """
        return {
            "name": cls.EMBEDDING_MODEL,
            "dimension": cls.EMBEDDING_DIMENSION,
            "max_length": cls.MAX_SEQUENCE_LENGTH,
            "description": "Sentence transformer model for general text embeddings"
        }
    
    @classmethod
    def get_multi_model_config(cls, model_key: str) -> Dict[str, Any]:
        """
        Get configuration for a specific multi-model.
        
        Args:
            model_key: Key for the model (e.g., "mpnet", "e5", "bge")
            
        Returns:
            Dict: Model configuration
        """
        return cls.MULTI_MODEL_CONFIGS.get(model_key, {})
    
    @classmethod
    def get_all_multi_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all multi-model configurations.
        
        Returns:
            Dict: All model configurations
        """
        return cls.MULTI_MODEL_CONFIGS
    
    @classmethod
    def get_multi_model_index_name(cls, model_key: str, source_index: str = "hf_models") -> str:
        """
        Get the vector index name for a specific multi-model.
        
        Args:
            model_key: Key for the model (e.g., "mpnet", "e5", "bge")
            source_index: The source index name
            
        Returns:
            str: The vector index name (e.g., "vector_mpnet_hf_models")
        """
        model_config = cls.get_multi_model_config(model_key)
        if not model_config:
            raise ValueError(f"Unknown model key: {model_key}")
        
        suffix = model_config.get("index_suffix", model_key)
        return f"vector_{suffix}_{source_index}"
    
    @classmethod
    def get_field_weight(cls, field_name: str) -> float:
        """
        Get the weight for a specific field.
        
        Args:
            field_name: Name of the field
            
        Returns:
            float: Field weight (default 1.0 if not configured)
        """
        return cls.FIELD_WEIGHTS.get(field_name, 1.0)
    
    @classmethod
    def get_weighted_fields_for_search(cls) -> List[str]:
        """
        Get fields with their weights for Elasticsearch search queries.
        
        Returns:
            List[str]: Fields with weight notation (e.g., ["name^3.0", "description^2.5"])
        """
        weighted_fields = []
        for field in cls.EMBEDDING_FIELDS:
            weight = cls.get_field_weight(field)
            weighted_fields.append(f"{field}^{weight}")
        return weighted_fields
    
    @classmethod
    def prepare_weighted_searchable_text(cls, model_data: Dict[str, Any]) -> str:
        """
        Prepare searchable text with field weights applied.
        
        Args:
            model_data: Dictionary containing model information
            
        Returns:
            str: Weighted searchable text
        """
        text_parts = []
        
        for field in cls.EMBEDDING_FIELDS:
            value = model_data.get(field, "")
            weight = cls.get_field_weight(field)
            
            if isinstance(value, list):
                # Join list items with spaces
                field_text = " ".join(str(item) for item in value if item)
            elif value:
                field_text = str(value)
            else:
                continue
            
            # Repeat text based on weight (simple weighting approach)
            # Higher weight = more repetitions in the text
            repetitions = max(1, int(weight))
            weighted_text = " ".join([field_text] * repetitions)
            text_parts.append(weighted_text)
        
        # Combine all text parts
        combined_text = " ".join(text_parts)
        return combined_text.strip()

# Create a global config instance that we can import elsewhere
config = VectorSimilarityConfig()

# ==================== VALIDATION ====================
def validate_config():
    """
    Validate that our configuration is correct.
    This helps catch errors early.
    """
    errors = []
    
    # Check if embedding model name is valid
    if not config.EMBEDDING_MODEL:
        errors.append("EMBEDDING_MODEL cannot be empty")
    
    # Check if we have source indices
    if not config.SOURCE_INDICES:
        errors.append("SOURCE_INDICES cannot be empty")
    
    # Check if embedding dimension is positive
    if config.EMBEDDING_DIMENSION <= 0:
        errors.append("EMBEDDING_DIMENSION must be positive")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True

# Validate configuration when this file is imported
if __name__ == "__main__":
    validate_config()
