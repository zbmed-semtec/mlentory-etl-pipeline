"""
Vector Index Management for ETL Pipeline

Self-contained module for creating and populating vector indices with embeddings.
Includes integrated embedding functionality (no external EmbeddingService dependency).

Location: mlentory-etl-pipeline/code/load/mlentory_load/core/vectors.py

Usage:
    from mlentory_load.core.vectors import VectorIndexManager
    
    manager = VectorIndexManager(
        index_handler=indexHandler,
        platform="hf"  # or "openml" or "ai4life"
    )
    manager.initialize_vector_index()  # Create index
    manager.update_vector_index()      # Populate with embeddings
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional

try:
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import login
except ImportError:
    SentenceTransformer = None
    login = None
    print("Warning: sentence-transformers not installed. Vector indexing will be disabled.")

# Metadata extraction is optional - if not available, we'll work with original data only
run_extraction = None


class VectorIndexManager:
    """
    Manager for creating and updating vector indices in the ETL pipeline.
    
    This class handles:
    - Creating vector indices from source indices
    - Generating embeddings for models using integrated embedding model
    - Incremental updates (only new/changed models)
    
    The embedding model is initialized directly in this class (no external dependency).
    """
    
    # Default embedding configuration
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    DEFAULT_EMBEDDING_DIMENSION = 768  # MPNet default
    DEFAULT_MAX_SEQUENCE_LENGTH = 512
    
    def __init__(self, index_handler, platform: str, logger=None):
        """
        Initialize the vector index manager.
        
        Args:
            index_handler: ETL's IndexHandler instance
            platform: Platform name ("hf", "openml", or "ai4life")
            logger: Optional logger instance
        """
        self.index_handler = index_handler
        self.platform = platform
        self.logger = logger or self._default_logger
        
        # Map platform to source index name
        self.source_index_map = {
            "hf": "hf_models",
            "openml": "openml_models",
            "ai4life": "ai4life_models"
        }
        
        self.source_index = self.source_index_map.get(platform, f"{platform}_models")
        self.vector_index = self.source_index.replace("_models", "_vector_models")
        
        # Initialize embedding model directly
        self.model = None
        self.embedding_model = self.DEFAULT_EMBEDDING_MODEL
        self.embedding_dimension = self.DEFAULT_EMBEDDING_DIMENSION
        self.max_sequence_length = self.DEFAULT_MAX_SEQUENCE_LENGTH
        self.device = self._resolve_device()
        self._load_embedding_model()
    
    def _default_logger(self, message: str, level: str = "info"):
        """Default logger that prints to console."""
        print(f"[{level.upper()}] {message}")
    
    def _log(self, message: str, level: str = "info"):
        """
        Unified logging method that handles both Logger objects and callable loggers.
        
        Args:
            message: Message to log
            level: Log level (info, warning, error)
        """
        if hasattr(self.logger, level):
            # It's a Logger object, use the appropriate method
            log_method = getattr(self.logger, level)
            log_method(message)
        elif callable(self.logger):
            # It's a callable function
            self.logger(message, level)
        else:
            # Fallback to print
            print(f"[{level.upper()}] {message}")
    
    def _resolve_device(self) -> str:
        """Resolve the device to use for embeddings (CUDA if available, else CPU)."""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    
    def _load_embedding_model(self):
        """Load the sentence transformer model for generating embeddings."""
        if not SentenceTransformer:
            self._log("sentence-transformers not available. Vector indexing disabled.", "warning")
            return
        
        # Authenticate with HuggingFace if token is provided
        self._authenticate_huggingface()
        
        # Set PyTorch environment variables
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        try:
            start_time = time.time()
            token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
            
            # Load the model
            try:
                # Handle special cases for certain models
                if "google/embeddinggemma" in self.embedding_model:
                    import torch
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                    torch.backends.cudnn.enabled = False
                
                self.model = SentenceTransformer(
                    self.embedding_model,
                    device=self.device,
                    trust_remote_code=True,
                    token=token if token else None
                )
            except Exception as meta_error:
                if "meta tensor" in str(meta_error).lower():
                    self._log("Meta tensor issue detected, trying alternative loading method...", "warning")
                    import torch
                    torch.backends.cudnn.enabled = False
                    self.model = SentenceTransformer(
                        self.embedding_model,
                        device=self.device,
                        trust_remote_code=True,
                        use_auth_token=False
                    )
                else:
                    raise meta_error
            
            load_time = time.time() - start_time
            
            # Validate the model
            self._validate_model()
            
            self._log(f"Embedding model initialized: {self.embedding_model} (loaded in {load_time:.2f}s)", "info")
            
        except Exception as e:
            self._log(f"Failed to load embedding model {self.embedding_model}: {e}", "warning")
            # Try minimal configuration as fallback
            try:
                self._log("Retrying with minimal configuration...", "info")
                self.model = SentenceTransformer(self.embedding_model)
                self._validate_model()
                self._log("Model loaded with minimal configuration", "info")
            except Exception as retry_error:
                self._log(f"Failed to load embedding model: {retry_error}", "error")
                self.model = None
    
    def _authenticate_huggingface(self):
        """Authenticate with HuggingFace if token is available."""
        if not login:
            return
        
        if not hasattr(self, '_hf_authenticated'):
            token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
            if token:
                try:
                    login(token=token)
                    self._log("Authenticated with Hugging Face", "info")
                except Exception as e:
                    self._log(f"Hugging Face authentication failed: {e}", "warning")
            self._hf_authenticated = True
    
    def _validate_model(self):
        """Validate that the loaded model works correctly."""
        if not self.model:
            return
        
        try:
            # Test with a simple sentence
            test_text = "This is a test sentence for validation"
            test_embedding = self.model.encode(
                [test_text],
                show_progress_bar=False,
                normalize_embeddings=False
            )
            
            # Check dimensions
            actual_dimension = len(test_embedding[0])
            if actual_dimension != self.embedding_dimension:
                self._log(
                    f"Expected dimension {self.embedding_dimension}, "
                    f"but got {actual_dimension}. Updating...",
                    "warning"
                )
                self.embedding_dimension = actual_dimension
        except Exception as e:
            self._log(f"Model validation failed: {e}", "error")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for embedding."""
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = text.strip()
        
        # Truncate if too long (models have limits)
        if len(cleaned) > self.max_sequence_length:
            cleaned = cleaned[:self.max_sequence_length]
        
        return cleaned
    
    def _encode_text(self, text: str) -> List[float]:
        """
        Encode text to vector using the loaded embedding model.
        
        Args:
            text: Text to encode
            
        Returns:
            List of floats representing the embedding vector
        """
        if not self.model:
            return [0.0] * self.embedding_dimension
        
        if not text or not text.strip():
            return [0.0] * self.embedding_dimension
        
        try:
            # Clean and prepare the text
            cleaned_text = self._clean_text(text)
            
            # Generate embedding
            embedding = self.model.encode(
                [cleaned_text],
                show_progress_bar=False,
                normalize_embeddings=False
            )
            vector = embedding[0].tolist()  # Convert numpy array to list
            
            return vector
            
        except Exception as e:
            self._log(f"Failed to encode text: {e}", "error")
            return [0.0] * self.embedding_dimension
    
    def initialize_vector_index(self):
        """
        Create the vector index if it doesn't exist.
        This matches the structure from SEA-App/scripts/create_indices.py.
        """
        if not self.model:
            self._log("Embedding model not available, skipping vector index creation", "warning")
            return False
        
        try:
            es = self.index_handler.es
            
            # Check if vector index already exists
            if es.indices.exists(index=self.vector_index):
                self._log(f"Vector index {self.vector_index} already exists", "info")
                return True
            
            # Create index mapping (matches create_indices.py structure)
            index_mapping = {
                "mappings": {
                    "properties": {
                        # Original model fields
                        "db_identifier": {"type": "keyword"},
                        "name": {"type": "text"},
                        "description": {"type": "text"},
                        "license": {"type": "keyword"},
                        "sharedBy": {"type": "text"},
                        "mlTask": {"type": "keyword"},
                        "keywords": {"type": "keyword"},
                        "relatedDatasets": {"type": "text"},
                        "baseModels": {"type": "text"},
                        "platform": {"type": "keyword"},
                        "dateCreated": {"type": "date"},
                        
                        # Extracted fields from description
                        "version": {"type": "keyword"},
                        "modalities": {"type": "keyword"},
                        "domain": {"type": "keyword"},
                        "architecture": {"type": "text"},
                        "modelSize": {"type": "keyword"},
                        "dataset": {"type": "text"},
                        "trainingType": {"type": "keyword"},
                        
                        # MPNet vector field (single model_vector like create_indices.py)
                        "model_vector": {
                            "type": "dense_vector",
                            "dims": self.embedding_dimension,
                            "index": True,
                            "similarity": "cosine"
                        },
                        
                        # Searchable text field
                        "searchable_text": {"type": "text"},
                        
                        # Metadata fields
                        "vector_created_at": {"type": "date"},
                        "embedding_model": {"type": "keyword"},
                        "source_index": {"type": "keyword"}
                    }
                }
            }
            
            es.indices.create(index=self.vector_index, body=index_mapping)
            self._log(f"Created vector index: {self.vector_index}", "info")
            return True
            
        except Exception as e:
            self._log(f"Failed to create vector index: {e}", "error")
            import traceback
            traceback.print_exc()
            return False
    
    def prepare_searchable_text(self, model_data: Dict[str, Any], extracted_data: Dict[str, Any]) -> str:
        """
        Prepare structured searchable text from model data for embedding.
        
        This matches the logic from SEA-App/scripts/create_indices.py.
        
        STRICT FIELD SEPARATION:
        - Original data: db_identifier, name, sharedBy, mlTask, keywords, etc.
        - Extracted metadata: version, modalities, domain, architecture, modelSize, dataset, trainingType
        
        Args:
            model_data: Dictionary containing ORIGINAL model information from source index
            extracted_data: Dictionary containing EXTRACTED fields from description only
            
        Returns:
            str: Structured text for embedding and storage
        """
        text_parts = []
        
        # === ORIGINAL DATA FIELDS ===
        
        # Model name (original only)
        name = model_data.get('name')
        if name:
            text_parts.append(f"The model name is {name}.")
        
        # Shared by (original only)
        shared_by = model_data.get('sharedBy')
        if shared_by:
            text_parts.append(f"It is shared by {shared_by}.")
        
        # ML Task (original only)
        ml_task = model_data.get('mlTask')
        if ml_task:
            if isinstance(ml_task, list):
                ml_task_str = ', '.join(str(t) for t in ml_task if t)
            else:
                ml_task_str = str(ml_task)
            text_parts.append(f"The model performs the {ml_task_str} task.")
        
        # Keywords (original only)
        keywords = model_data.get('keywords')
        if keywords and isinstance(keywords, list) and keywords:
            keywords_str = ', '.join(str(k) for k in keywords if k)
            text_parts.append(f"Keywords: {keywords_str}.")
                
        # Base Models (original only - for reference)
        base_models = model_data.get('baseModels')
        if base_models and isinstance(base_models, list) and base_models:
            base_models_str = ', '.join(str(b) for b in base_models if b)
            text_parts.append(f"Base models: {base_models_str}.")
        
        # === EXTRACTED FIELDS ===
        
        # Version (extracted)
        version = extracted_data.get('version')
        if version:
            text_parts.append(f"The version is {version}.")
        
        # Modalities (extracted)
        modalities = extracted_data.get('modalities')
        if modalities and isinstance(modalities, list) and modalities:
            modalities_str = ', '.join(str(m) for m in modalities if m)
            text_parts.append(f"The model works with {modalities_str} modalities.")
        
        # Domain (extracted)
        domain = extracted_data.get('domain')
        if domain:
            text_parts.append(f"The domain of the model is {domain}.")
        
        # Training type (extracted)
        training_type = extracted_data.get('trainingType')
        if training_type:
            text_parts.append(f"The training type is {training_type}.")
        
        # Architecture (extracted)
        architecture = extracted_data.get('architecture')
        if architecture:
            text_parts.append(f"The architecture is {architecture}.")
        
        # Model size (extracted)
        model_size = extracted_data.get('modelSize')
        if model_size:
            text_parts.append(f"The model size is {model_size}.")
        
        # Datasets: combine original relatedDatasets and extracted dataset into one, deduplicated output
        combined_datasets = []
        seen_lower = set()

        # From original field relatedDatasets
        related_datasets = model_data.get('relatedDatasets')
        if related_datasets:
            if isinstance(related_datasets, list):
                candidates = related_datasets
            else:
                candidates = [str(related_datasets)]
            for ds in candidates:
                if not ds:
                    continue
                ds_str = str(ds).strip()
                if not ds_str or ds_str.lower() == 'information not found':
                    continue
                key = ds_str.lower()
                if key not in seen_lower:
                    seen_lower.add(key)
                    combined_datasets.append(ds_str)

        # From extracted field dataset
        extracted_ds = extracted_data.get('dataset')
        if extracted_ds:
            ds_str = str(extracted_ds).strip()
            if ds_str:
                key = ds_str.lower()
                if key not in seen_lower:
                    seen_lower.add(key)
                    combined_datasets.append(ds_str)

        if combined_datasets:
            if len(combined_datasets) == 1:
                text_parts.append(f"It was trained on the {combined_datasets[0]} dataset.")
            else:
                datasets_str = ', '.join(combined_datasets)
                text_parts.append(f"It was trained on the {datasets_str} datasets.")
        
        # Join all parts
        return ' '.join(text_parts)
    
    def update_vector_index(self, model_ids: Optional[List[str]] = None, batch_size: int = 50):
        """
        Populate vector index with embeddings from the source index.
        Matches the structure from SEA-App/scripts/create_indices.py.
        
        Args:
            model_ids: Optional list of specific model IDs to update.
                      If None, updates all models from source index.
            batch_size: Number of models to process in each batch
        """
        if not self.model:
            self._log("Embedding model not available", "warning")
            return
        
        try:
            # Ensure vector index exists
            if not self.initialize_vector_index():
                return
            
            es = self.index_handler.es
            
            # Get models to process
            if model_ids:
                # Process specific models
                query = {"query": {"terms": {"_id": model_ids}}}
            else:
                # Process all models
                query = {"query": {"match_all": {}}}
            
            # Use scroll API for large datasets
            response = es.search(
                index=self.source_index,
                body=query,
                scroll='5m',
                size=batch_size,
                _source=True
            )
            
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
            total_processed = 0
            
            while hits:
                self._log(f"Processing batch of {len(hits)} models...", "info")
                
                # Process batch
                for hit in hits:
                    model_data = hit["_source"]
                    model_id = hit["_id"]
                    
                    try:
                        # Extract additional fields from description (optional - if run_extraction not available, use empty dict)
                        extracted_data = {}
                        if run_extraction:
                            try:
                                extracted_data = run_extraction(model_data)
                            except Exception as e:
                                self._log(f"Extraction failed for model {model_id}, using original data only: {e}", "warning")
                                extracted_data = {}
                        
                        # Prepare searchable text using structured format
                        searchable_text = self.prepare_searchable_text(model_data, extracted_data)
                        
                        if not searchable_text:
                            self._log(f"Skipping model {model_id} - no searchable text", "warning")
                            continue
                        
                        # Generate embedding using integrated model
                        model_vector = self._encode_text(searchable_text)
                        
                        # Create document with original data, extracted fields, vector, and searchable text
                        doc_body = {
                            **model_data,  # Original fields
                            **extracted_data,  # Extracted fields
                            "model_vector": model_vector,
                            "searchable_text": searchable_text,  # Store the text used for embedding
                            "vector_created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "embedding_model": self.embedding_model,
                            "source_index": self.source_index
                        }
                        
                        # Save to Elasticsearch
                        es.index(
                            index=self.vector_index,
                            id=model_id,
                            body=doc_body
                        )
                        total_processed += 1
                        
                    except Exception as e:
                        self._log(f"Error processing model {model_id}: {e}", "warning")
                        continue
                
                # Get next batch
                response = es.scroll(scroll_id=scroll_id, scroll='5m')
                scroll_id = response['_scroll_id']
                hits = response['hits']['hits']
                
                self._log(f"Processed {total_processed} models so far...", "info")
            
            # Clear scroll and refresh
            es.clear_scroll(scroll_id=scroll_id)
            es.indices.refresh(index=self.vector_index)
            
            self._log(f"Successfully processed {total_processed} models for vector index", "info")
            
        except Exception as e:
            self._log(f"Vector index update failed: {e}", "error")
            import traceback
            traceback.print_exc()

