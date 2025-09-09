import time
from typing import List, Dict, Any, Optional, Tuple
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch_dsl import Document, Text, Keyword, DenseVector, Date, Integer

from .config import config
from .embedding_service import EmbeddingService

class VectorModelDocument(Document):
    """
    Elasticsearch document structure for storing models with vector embeddings.
    
    This defines how our vector data will be stored in Elasticsearch:
    - Original model fields (name, description, etc.)
    - Vector embedding field (384-dimensional vector)
    - Metadata fields (platform, creation date, etc.)
    """
    
    # Original model fields (copied from source)
    db_identifier = Keyword()
    name = Text()
    description = Text()
    license = Keyword()
    sharedBy = Text()
    mlTask = Keyword(multi=True)
    keywords = Keyword(multi=True)
    relatedDatasets = Text(multi=True)
    baseModels = Text(multi=True)
    platform = Keyword()
    dateCreated = Date()
    
    # Vector embedding field (the magic happens here!)
    model_vector = DenseVector(
        dims=config.EMBEDDING_DIMENSION,  # 384 dimensions
        index=True,                      # Enable vector search
        similarity="cosine"              # Use cosine similarity for search
    )
    
    # Metadata fields
    vector_created_at = Date()
    embedding_model = Keyword()
    source_index = Keyword()
    
    class Meta:
        # This will be set dynamically for each index
        index = "vector_models"
        doc_type = "_doc"

class VectorIndexHandler:
    """
    Handler for managing vector indices in Elasticsearch.
    
    This class handles:
    - Creating vector indices with proper mappings
    - Storing vector embeddings alongside model data
    - Performing vector similarity searches
    - Managing bulk operations for efficiency
    """
    
    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize the vector index handler.
        
        Args:
            embedding_service: The embedding service for generating vectors
        """
        self.embedding_service = embedding_service
        self.es = None
        self.vector_indices = {}
        
        
        # Connect to Elasticsearch
        self._connect_to_elasticsearch()
        
        # Initialize vector indices
        self._initialize_vector_indices()
    
    def _connect_to_elasticsearch(self):
        """
        Connect to our existing Elasticsearch container.
        
        This connects to the same Elasticsearch instance that stores our original model data.
        """
        try:
            
            # Create Elasticsearch client
            self.es = Elasticsearch(
                [config.get_elasticsearch_url()],
                basic_auth=(config.ELASTICSEARCH_USERNAME, config.ELASTICSEARCH_PASSWORD),
                request_timeout=30,
                max_retries=3,
                retry_on_timeout=True,
            )
            
            # Test connection
            if not self.es.ping():
                raise ConnectionError("Failed to ping Elasticsearch")
                
        except Exception as e:
            print(f"❌ Failed to connect to Elasticsearch: {e}")
            raise
    
    def _initialize_vector_indices(self):
        """
        Initialize vector indices for each source index.
        
        This creates the vector indices with the proper structure for storing embeddings.
        """
        
        for source_index in config.SOURCE_INDICES:
            vector_index_name = config.get_vector_index_name(source_index)
            
            try:
                # Create the vector index if it doesn't exist
                if not self.es.indices.exists(index=vector_index_name):
                    self._create_vector_index(vector_index_name, source_index)
                
                # Store the index name for later use
                self.vector_indices[source_index] = vector_index_name
                
            except Exception as e:
                print(f"❌ Failed to initialize vector index {vector_index_name}: {e}")
                raise
    
    def _create_vector_index(self, index_name: str, source_index: str):
        """
        Create a new vector index with the proper mapping.
        
        Args:
            index_name: Name of the vector index to create
            source_index: Name of the source index (for metadata)
        """
        # Define the mapping for our vector index
        mapping = {
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
                    
                    # Vector embedding field (the key part!)
                    "model_vector": {
                        "type": "dense_vector",
                        "dims": config.EMBEDDING_DIMENSION,
                        "index": True,
                        "similarity": "cosine"
                    },
                    
                    # Metadata fields
                    "vector_created_at": {"type": "date"},
                    "embedding_model": {"type": "keyword"},
                    "source_index": {"type": "keyword"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }
        
        # Create the index
        self.es.indices.create(index=index_name, body=mapping)
    
    def index_model_with_vector(self, model_data: Dict[str, Any], source_index: str) -> bool:
        """
        Index a single model with its vector embedding.
        
        Args:
            model_data: The model data from the source index
            source_index: The source index name (e.g., "hf_models")
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            vector_index_name = self.vector_indices[source_index]
            
            # Prepare the text for embedding
            searchable_text = self._prepare_searchable_text(model_data)
            
            # Generate vector embedding
            vector = self.embedding_service.encode_text(searchable_text)
            
            # Create document for vector index
            vector_doc = {
                # Copy original fields
                "db_identifier": model_data.get("db_identifier", ""),
                "name": model_data.get("name", ""),
                "description": model_data.get("description", ""),
                "license": model_data.get("license", ""),
                "sharedBy": model_data.get("sharedBy", ""),
                "mlTask": model_data.get("mlTask", []),
                "keywords": model_data.get("keywords", []),
                "relatedDatasets": model_data.get("relatedDatasets", []),
                "baseModels": model_data.get("baseModels", []),
                "platform": model_data.get("platform", ""),
                "dateCreated": model_data.get("dateCreated") or "2024-01-01",  # Default date if empty
                
                # Add vector embedding
                "model_vector": vector,
                
                # Add metadata
                "vector_created_at": int(time.time() * 1000),  # Current timestamp in milliseconds
                "embedding_model": config.EMBEDDING_MODEL,
                "source_index": source_index
            }
            
            # Index the document
            self.es.index(
                index=vector_index_name,
                id=model_data.get("db_identifier", ""),
                body=vector_doc
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to index model with vector: {e}")
            return False
    
    def bulk_index_models_with_vectors(self, models_data: List[Dict[str, Any]], source_index: str) -> Dict[str, int]:
        """
        Bulk index multiple models with their vector embeddings.
        
        Args:
            models_data: List of model data from the source index
            source_index: The source index name
            
        Returns:
            Dict: Statistics about the indexing operation
        """
        if not models_data:
            return {"success": 0, "failed": 0}
        
        vector_index_name = self.vector_indices[source_index]
        success_count = 0
        failed_count = 0
        
        
        # Process in batches for efficiency
        batch_size = config.BATCH_SIZE
        for i in range(0, len(models_data), batch_size):
            batch = models_data[i:i + batch_size]
            
            try:
                # Prepare batch data
                batch_docs = []
                batch_texts = []
                
                for model_data in batch:
                    # Prepare searchable text
                    searchable_text = self._prepare_searchable_text(model_data)
                    batch_texts.append(searchable_text)
                
                # Generate embeddings for the batch
                vectors = self.embedding_service.encode_batch(batch_texts)
                
                # Ensure we have the same number of vectors as models
                if len(vectors) != len(batch):
                    # Skip this batch if there's a mismatch
                    failed_count += len(batch)
                    continue
                
                # Create documents for the batch
                for model_data, vector in zip(batch, vectors):
                    # Create a unique ID if db_identifier is empty
                    doc_id = model_data.get("db_identifier", f"test_{time.time()}")
                    
                    doc = {
                        "_index": vector_index_name,
                        "_id": doc_id,
                        "_source": {
                            # Copy original fields
                            "db_identifier": doc_id,
                            "name": model_data.get("name", ""),
                            "description": model_data.get("description", ""),
                            "license": model_data.get("license", ""),
                            "sharedBy": model_data.get("sharedBy", ""),
                            "mlTask": model_data.get("mlTask", []),
                            "keywords": model_data.get("keywords", []),
                            "relatedDatasets": model_data.get("relatedDatasets", []),
                            "baseModels": model_data.get("baseModels", []),
                            "platform": model_data.get("platform", ""),
                            "dateCreated": model_data.get("dateCreated") or "2024-01-01",
                            
                            # Add vector embedding
                            "model_vector": vector,
                            
                            # Add metadata
                            "vector_created_at": int(time.time() * 1000),  # Convert to integer
                            "embedding_model": config.EMBEDDING_MODEL,
                            "source_index": source_index
                        }
                    }
                    batch_docs.append(doc)
                
                # Bulk index the batch
                try:
                    success, failed = bulk(self.es, batch_docs)
                    success_count += success
                    failed_count += len(failed)  # failed is a list, not a count
                    
                    
                    # Log any failed documents
                except Exception as bulk_error:
                    print(f"❌ Bulk indexing failed for batch {i//batch_size + 1}: {bulk_error}")
                    failed_count += len(batch_docs)
                
            except Exception as e:
                print(f"❌ Failed to process batch {i//batch_size + 1}: {e}")
                failed_count += len(batch)
        
        return {"success": success_count, "failed": failed_count}
    
    def _prepare_searchable_text(self, model_data: Dict[str, Any]) -> str:
        """
        Prepare searchable text from model data.
        
        Args:
            model_data: The model data dictionary
            
        Returns:
            str: Combined searchable text
        """
        text_parts = []
        
        # Add fields specified in config
        for field in config.EMBEDDING_FIELDS:
            value = model_data.get(field, "")
            
            if isinstance(value, list):
                # Handle list fields (like mlTask, keywords)
                text_parts.extend([str(v) for v in value if v])
            elif value:
                # Handle single values
                text_parts.append(str(value))
        
        # Join with separator
        return config.FIELD_SEPARATOR.join(text_parts)
    
    def vector_search(self, query_text: str, source_index: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            query_text: The search query text
            source_index: The source index to search in
            top_k: Number of results to return
            
        Returns:
            List[Dict]: Search results with similarity scores
        """
        try:
            vector_index_name = self.vector_indices[source_index]
            
            # Generate query vector
            query_vector = self.embedding_service.encode_text(query_text)
            
            # Build search query (Elasticsearch 8.x syntax) with improved settings
            search_body = {
                "size": top_k,
                "knn": {
                    "field": "model_vector",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": config.VECTOR_SEARCH_CANDIDATES  # Use config setting for better quality
                },
                "_source": [
                    "name", "description", "mlTask", "keywords", "platform", 
                    "db_identifier", "sharedBy", "license"
                ]
            }
            
            # Execute search
            response = self.es.search(index=vector_index_name, body=search_body)
            
            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "score": hit["_score"],
                    "model_data": hit["_source"],
                    "db_identifier": hit["_source"].get("db_identifier", "")
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Vector search failed: {e}")
            return []
    
    def get_index_stats(self, source_index: str) -> Dict[str, Any]:
        """
        Get statistics about a vector index.
        
        Args:
            source_index: The source index name
            
        Returns:
            Dict: Index statistics
        """
        try:
            vector_index_name = self.vector_indices[source_index]
            stats = self.es.indices.stats(index=vector_index_name)
            
            return {
                "index_name": vector_index_name,
                "source_index": source_index,
                "document_count": stats["indices"][vector_index_name]["total"]["docs"]["count"],
                "index_size": stats["indices"][vector_index_name]["total"]["store"]["size_in_bytes"]
            }
            
        except Exception as e:
            print(f"❌ Failed to get index stats: {e}")
            return {}

