#!/usr/bin/env python3
"""
Create Multi-Vector Index for hf_models

This script creates and populates the vector_multi_hf_models index with multiple vector fields:
- name_vector: Vector for the model name only
- task_vector: Vector for mlTask + keywords combined  
- creator_vector: Vector for sharedBy field only

Usage:
    python3 create_multi_vector_index.py
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

class MultiVectorDocument(Document):
    """
    Elasticsearch document structure for multi-vector embeddings.
    """
    
    class Index:
        name = 'vector_multi_hf_models'
    
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
    
    # Multi-vector fields
    name_vector = DenseVector(
        dims=config.EMBEDDING_DIMENSION,
        index=True,
        similarity="cosine"
    )
    task_vector = DenseVector(
        dims=config.EMBEDDING_DIMENSION,
        index=True,
        similarity="cosine"
    )
    creator_vector = DenseVector(
        dims=config.EMBEDDING_DIMENSION,
        index=True,
        similarity="cosine"
    )
    
    # Metadata fields
    vector_created_at = Date()
    embedding_model = Keyword()
    source_index = Keyword()

class MultiVectorIndexCreator:
    """
    Creates and populates the multi-vector index for hf_models.
    """
    
    def __init__(self):
        """Initialize the multi-vector index creator."""
        self.embedding_service = None
        self.es = None
        self.source_index = "hf_models"
        self.vector_index = "vector_multi_hf_models"
        
    def initialize_services(self):
        """Initialize embedding service and Elasticsearch connection."""
        print("🔧 Initializing services...")
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService()
        print("✅ Embedding service ready")
        
        # Initialize Elasticsearch connection
        self.es = Elasticsearch(
            [config.get_elasticsearch_url()],
            basic_auth=(config.ELASTICSEARCH_USERNAME, config.ELASTICSEARCH_PASSWORD),
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )
        
        if not self.es.ping():
            raise Exception("Cannot connect to Elasticsearch")
        print("✅ Elasticsearch connection ready")
        
    def create_vector_index(self):
        """Create the vector index if it doesn't exist."""
        print(f"📝 Creating vector index: {self.vector_index}")
        
        if self.es.indices.exists(index=self.vector_index):
            print(f"✅ Vector index {self.vector_index} already exists")
            return
        
        # Create index with proper mapping
        MultiVectorDocument.init(using=self.es)
        print(f"✅ Created vector index: {self.vector_index}")
        
    def get_source_model_count(self) -> int:
        """Get the total number of models in the source index."""
        try:
            response = self.es.count(index=self.source_index)
            return response['count']
        except Exception as e:
            print(f"❌ Failed to get source model count: {e}")
            return 0
    
    def get_vector_model_count(self) -> int:
        """Get the total number of models in the vector index."""
        try:
            response = self.es.count(index=self.vector_index)
            return response['count']
        except Exception as e:
            print(f"❌ Failed to get vector model count: {e}")
            return 0
    
    def prepare_vector_texts(self, model_data):
        """
        Prepare the text content for each vector type.
        
        Args:
            model_data: The model data from the source index
            
        Returns:
            Dict with vector types as keys and text content as values
        """
        vector_texts = {}
        
        # 1. Name vector (exact model name)
        vector_texts['name_vector'] = model_data.get('name', '')
        
        # 2. Task vector (mlTask + keywords combined)
        mlTask = model_data.get('mlTask', '')
        keywords = model_data.get('keywords', '')
        task_text = f"{mlTask} {keywords}".strip()
        vector_texts['task_vector'] = task_text if task_text else ''
        
        # 3. Creator vector (sharedBy)
        vector_texts['creator_vector'] = model_data.get('sharedBy', '')
        
        return vector_texts
    
    def populate_vector_index(self):
        """Populate the vector index with embeddings."""
        print("🔄 Populating vector index with embeddings...")
        
        # Get all models from source index
        print("📥 Fetching models from hf_models...")
        response = self.es.search(
            index=self.source_index,
            body={"query": {"match_all": {}}, "size": 1000},
            scroll='5m'
        )
        
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        total_processed = 0
        
        while hits:
            print(f"🔄 Processing batch of {len(hits)} models...")
            
            for hit in hits:
                model_data = hit["_source"]
                model_id = model_data.get("db_identifier") or hit["_id"]
                
                try:
                    # Prepare vector texts for this model
                    vector_texts = self.prepare_vector_texts(model_data)
                    
                    # Generate embeddings for each vector type
                    embeddings = {}
                    for vector_type, text in vector_texts.items():
                        if text.strip():  # Only create embedding if text is not empty
                            try:
                                embedding = self.embedding_service.encode_text(text)
                                embeddings[vector_type] = embedding
                            except Exception as e:
                                print(f"⚠️  Failed to create {vector_type} embedding for {model_id}: {e}")
                                embeddings[vector_type] = None
                        else:
                            embeddings[vector_type] = None
                    
                    # Create document
                    doc_body = {
                        **model_data,
                        **embeddings,
                        "vector_created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "embedding_model": config.EMBEDDING_MODEL,
                        "source_index": self.source_index
                    }
                    
                    # Index the document
                    self.es.index(
                        index=self.vector_index,
                        id=model_id,
                        body=doc_body
                    )
                    total_processed += 1
                    
                except Exception as e:
                    print(f"❌ Error processing model {model_id}: {e}")
                    continue
            
            # Get next batch
            response = self.es.scroll(scroll_id=scroll_id, scroll='5m')
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
            
            print(f"📊 Processed {total_processed} models so far...")
        
        # Clear scroll
        self.es.clear_scroll(scroll_id=scroll_id)
        
        # Refresh index
        self.es.indices.refresh(index=self.vector_index)
        print(f"✅ Successfully processed {total_processed} models")
        print(f"🔄 Index refreshed")
    
    def run(self):
        """Run the complete process."""
        print("🚀 Starting Multi-Vector Index Creation...")
        print("📋 Configuration:")
        print(f"   - Source index: {self.source_index}")
        print(f"   - Vector index: {self.vector_index}")
        print(f"   - Embedding model: {config.EMBEDDING_MODEL}")
        print(f"   - Embedding dimension: {config.EMBEDDING_DIMENSION}")
        print(f"   - Vector types: name_vector, task_vector, creator_vector")
        
        try:
            # Initialize services
            self.initialize_services()
            
            # Create vector index
            self.create_vector_index()
            
            # Get source model count
            print(f"\n📊 Checking source index...")
            source_count = self.get_source_model_count()
            print(f"   📥 Source models in {self.source_index}: {source_count}")
            
            if source_count == 0:
                print("❌ No models found in source index. Exiting.")
                return
            
            # Check vector index
            print(f"\n🔍 Checking vector index...")
            vector_count = self.get_vector_model_count()
            print(f"   📊 Vector models in {self.vector_index}: {vector_count}")
            
            if vector_count == 0:
                print(f"   📝 Vector index is empty, populating with all models...")
                self.populate_vector_index()
            elif vector_count < source_count:
                print(f"   ⚠️  Vector index has {vector_count} models, but source has {source_count}")
                print(f"   🔄 Completing population with remaining models...")
                self.populate_vector_index()
            else:
                print(f"   ✅ Vector index already has {vector_count} models")
                print(f"   ℹ️  Skipping population (index already populated)")
            
            # Final count
            final_count = self.get_vector_model_count()
            print(f"\n✅ Process completed!")
            print(f"   📊 Final vector index count: {final_count}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

def main():
    """Main function."""
    creator = MultiVectorIndexCreator()
    creator.run()

if __name__ == "__main__":
    main()
