#!/usr/bin/env python3
"""
Create Multi-Model Vector Indices

This script creates separate vector indices for different embedding models.
Each model gets its own index with model-specific vector dimensions and configurations.

Models supported:
- sentence-transformers/all-mpnet-base-v2 (768D)
- intfloat/e5-base-v2 (768D) 
- BAAI/bge-base-en-v1.5 (768D)

Usage:
    python3 create_multi_model_vector_indices.py
"""

import sys
import os
import time
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Document, Text, Keyword, DenseVector, Date, Index

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.embedding_service import EmbeddingService
from src.config import config

class MultiModelVectorDocument(Document):
    """
    Elasticsearch document structure for multi-model vector embeddings.
    """
    
    class Index:
        name = 'vector_multi_model_hf_models'  # Will be set dynamically
    
    # Original model fields
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
    
    # Model-specific vector field
    model_vector = DenseVector(
        dims=768,  # Will be set dynamically based on model
        index=True,
        similarity="cosine"
    )
    
    # Metadata fields
    vector_created_at = Date()
    embedding_model = Keyword()
    source_index = Keyword()
    
    class Meta:
        index = "vector_multi_model_hf_models"  # Will be set dynamically
        doc_type = "_doc"

class MultiModelVectorIndexManager:
    """
    Manager for creating and populating multiple model-specific vector indices.
    """
    
    def __init__(self):
        """Initialize the multi-model vector index manager."""
        self.embedding_services = {}
        self.es = None
        self.source_index = "hf_models"
        
        # Get model configurations from config
        self.model_configs = config.get_all_multi_models()
        
    def initialize_services(self):
        """Initialize embedding services and Elasticsearch connection."""
        print("🔧 Initializing services...")
        
        # Initialize embedding services for each model
        for model_key, model_config in self.model_configs.items():
            print(f"   📦 Loading {model_config['model_name']}...")
            try:
                self.embedding_services[model_key] = EmbeddingService(
                    model_name=model_config['model_name']
                )
                print(f"   ✅ {model_key.upper()} service ready")
            except Exception as e:
                print(f"   ❌ Failed to load {model_key}: {e}")
                raise
        
        # Connect to Elasticsearch
        self.es = Elasticsearch(
            [config.get_elasticsearch_url()],
            basic_auth=(config.ELASTICSEARCH_USERNAME, config.ELASTICSEARCH_PASSWORD),
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )
        
        if not self.es.ping():
            raise Exception("Cannot connect to Elasticsearch")
        print(f"✅ Elasticsearch connection ready")
        
    def create_vector_index(self, model_key: str):
        """Create a vector index for a specific model."""
        model_config = self.model_configs[model_key]
        index_name = config.get_multi_model_index_name(model_key, self.source_index)
        
        print(f"📝 Creating vector index: {index_name}")
        
        if self.es.indices.exists(index=index_name):
            print(f"✅ Vector index {index_name} already exists")
            return index_name
        
        # Create index with proper mapping
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
                    
                    # Model-specific vector field
                    "model_vector": {
                        "type": "dense_vector",
                        "dims": model_config['dimension'],
                        "index": True,
                        "similarity": "cosine"
                    },
                    
                    # Metadata fields
                    "vector_created_at": {"type": "date"},
                    "embedding_model": {"type": "keyword"},
                    "source_index": {"type": "keyword"}
                }
            }
        }
        
        # Create the index
        self.es.indices.create(index=index_name, body=index_mapping)
        print(f"✅ Created vector index: {index_name}")
        
        return index_name
        
    def get_source_model_count(self) -> int:
        """Get the total number of models in the source index."""
        try:
            response = self.es.count(index=self.source_index)
            return response['count']
        except Exception as e:
            print(f"❌ Failed to get source model count: {e}")
            return 0
    
    def get_vector_model_count(self, index_name: str) -> int:
        """Get the total number of models in a vector index."""
        try:
            response = self.es.count(index=index_name)
            return response['count']
        except Exception as e:
            print(f"❌ Failed to get vector model count for {index_name}: {e}")
            return 0
    
    def prepare_searchable_text(self, model_data):
        """
        Prepare searchable text from model data for embedding with field weights.
        
        Args:
            model_data: Dictionary containing model information
            
        Returns:
            str: Weighted combined searchable text
        """
        # Use the new weighted text preparation from config
        return config.prepare_weighted_searchable_text(model_data)
    
    def populate_vector_index(self, model_key: str, index_name: str, batch_size: int = 50):
        """
        Populate a vector index with embeddings from the source index.
        
        Args:
            model_key: Key for the embedding model to use
            index_name: Name of the vector index to populate
            batch_size: Number of models to process in each batch
        """
        print(f"🔄 Populating vector index {index_name} with {model_key.upper()} embeddings...")
        
        embedding_service = self.embedding_services[model_key]
        model_config = self.model_configs[model_key]
        
        # Get all models from source index
        print(f"📥 Fetching models from {self.source_index}...")
        
        # Use scroll API for large datasets
        response = self.es.search(
            index=self.source_index,
            body={"query": {"match_all": {}}},
            scroll='5m',
            size=batch_size
        )
        
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        total_processed = 0
        
        while hits:
            print(f"🔄 Processing batch of {len(hits)} models with {model_key.upper()}...")
            
            # Process batch
            for hit in hits:
                model_data = hit["_source"]
                model_id = hit["_id"]
                
                try:
                    # Prepare searchable text
                    searchable_text = self.prepare_searchable_text(model_data)
                    
                    if not searchable_text:
                        print(f"⚠️  Skipping model {model_id} - no searchable text")
                        continue
                    
                    # Generate embedding using the specific model
                    model_vector = embedding_service.encode_text(searchable_text)
                    
                    # Create document
                    doc_body = {
                        **model_data,
                        "model_vector": model_vector,
                        "vector_created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "embedding_model": model_config['model_name'],
                        "source_index": self.source_index
                    }
                    
                    # Save to Elasticsearch
                    self.es.index(
                        index=index_name,
                        id=model_id,
                        body=doc_body
                    )
                    total_processed += 1
                    
                except Exception as e:
                    print(f"❌ Error processing model {model_id} with {model_key}: {e}")
                    continue
            
            # Get next batch
            response = self.es.scroll(scroll_id=scroll_id, scroll='5m')
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
            
            print(f"📊 Processed {total_processed} models so far with {model_key.upper()}...")
        
        # Clear scroll
        self.es.clear_scroll(scroll_id=scroll_id)
        
        # Refresh index
        self.es.indices.refresh(index=index_name)
        print(f"✅ Successfully processed {total_processed} models with {model_key.upper()}")
        print(f"🔄 Index {index_name} refreshed")
    
    def run(self):
        """Run the complete process for all models."""
        print("🚀 Starting Multi-Model Vector Index Creation...")
        print("📋 Configuration:")
        print(f"   - Source index: {self.source_index}")
        print(f"   - Models: {len(self.model_configs)}")
        for model_key, model_config in self.model_configs.items():
            print(f"     • {model_key.upper()}: {model_config['model_name']} ({model_config['dimension']}D)")
        
        try:
            # Initialize services
            self.initialize_services()
            
            # Check source index
            print(f"\n📊 Checking source index...")
            source_count = self.get_source_model_count()
            print(f"   📥 Source models in {self.source_index}: {source_count}")
            
            if source_count == 0:
                print(f"❌ No models found in source index {self.source_index}")
                return
            
            # Process each model
            for model_key, model_config in self.model_configs.items():
                print(f"\n🔧 Processing {model_key.upper()} model...")
                print(f"   📦 Model: {model_config['model_name']}")
                print(f"   📏 Dimension: {model_config['dimension']}")
                print(f"   📝 Description: {model_config['description']}")
                
                # Create vector index
                index_name = self.create_vector_index(model_key)
                
                # Check if index needs population
                vector_count = self.get_vector_model_count(index_name)
                print(f"   📊 Current vector count: {vector_count}")
                
                if vector_count == 0:
                    print(f"   📝 Index is empty, populating with {model_key.upper()} embeddings...")
                    self.populate_vector_index(model_key, index_name)
                elif vector_count < source_count:
                    print(f"   ⚠️  Index has {vector_count} models, but source has {source_count}")
                    print(f"   🔄 Completing population with remaining models...")
                    self.populate_vector_index(model_key, index_name)
                else:
                    print(f"   ✅ Index already has {vector_count} models")
                    print(f"   ℹ️  Skipping population (index already populated)")
                
                # Final count for this model
                final_count = self.get_vector_model_count(index_name)
                print(f"   ✅ {model_key.upper()} completed! Final count: {final_count}")
            
            print(f"\n🎉 All models processed successfully!")
            print(f"📊 Summary:")
            for model_key, model_config in self.model_configs.items():
                index_name = config.get_multi_model_index_name(model_key, self.source_index)
                final_count = self.get_vector_model_count(index_name)
                print(f"   • {model_key.upper()}: {final_count} models in {index_name}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

def main():
    """Main function."""
    manager = MultiModelVectorIndexManager()
    manager.run()

if __name__ == "__main__":
    main()
