#!/usr/bin/env python3
"""
Create Single Vector Index for hf_models

This script creates and populates the vector_hf_models index with a single model_vector field.
It's separate from the multi-vector indices and serves a different purpose.

Usage:
    python3 create_single_vector_index.py
"""

import sys
import time
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Document, Text, Keyword, DenseVector, Date, Index
from src.embedding_service import EmbeddingService
from src.config import config

class SingleVectorDocument(Document):
    """
    Elasticsearch document structure for single vector embeddings.
    """
    
    class Index:
        name = 'vector_hf_models'
    
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
    
    # Single vector field
    model_vector = DenseVector(
        dims=config.EMBEDDING_DIMENSION,
        index=True,
        similarity="cosine"
    )
    
    # Metadata fields
    vector_created_at = Date()
    embedding_model = Keyword()
    source_index = Keyword()
    
    class Meta:
        index = "vector_hf_models"
        doc_type = "_doc"

class SingleVectorIndexManager:
    """
    Manager for creating and populating the single vector index.
    """
    
    def __init__(self):
        """Initialize the single vector index manager."""
        self.embedding_service = None
        self.es = None
        self.source_index = "hf_models"
        self.vector_index = "vector_hf_models"
        
    def initialize_services(self):
        """Initialize embedding service and Elasticsearch connection."""
        print("🔧 Initializing services...")
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService()
        print(f"✅ Embedding service ready")
        
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
        
    def create_vector_index(self):
        """Create the vector index if it doesn't exist."""
        print(f"📝 Creating vector index: {self.vector_index}")
        
        if self.es.indices.exists(index=self.vector_index):
            print(f"✅ Vector index {self.vector_index} already exists")
            return
        
        # Create index with proper mapping
        SingleVectorDocument.init(using=self.es)
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
    
    def prepare_searchable_text(self, model_data):
        """
        Prepare searchable text from model data for single vector embedding.
        
        Args:
            model_data: Dictionary containing model information
            
        Returns:
            str: Combined searchable text
        """
        text_parts = []
        
        # Add fields specified in config
        for field in config.EMBEDDING_FIELDS:
            value = model_data.get(field, "")
            
            if isinstance(value, list):
                # Join list items with spaces
                text_parts.append(" ".join(str(item) for item in value if item))
            elif value:
                text_parts.append(str(value))
        
        # Combine all text parts
        combined_text = " ".join(text_parts)
        return combined_text.strip()
    
    def populate_vector_index(self, batch_size: int = 50):
        """
        Populate the vector index with embeddings from the source index.
        
        Args:
            batch_size: Number of models to process in each batch
        """
        print(f"🔄 Populating vector index with embeddings...")
        
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
            print(f"🔄 Processing batch of {len(hits)} models...")
            
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
                    
                    # Generate embedding
                    model_vector = self.embedding_service.encode_text(searchable_text)
                    
                    # Create document
                    doc = SingleVectorDocument(
                        meta={'id': model_id},
                        **model_data,
                        model_vector=model_vector,
                        vector_created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        embedding_model=config.EMBEDDING_MODEL,
                        source_index=self.source_index
                    )
                    
                    # Save to Elasticsearch
                    doc.save(using=self.es)
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
        print("🚀 Starting Single Vector Index Creation...")
        print("📋 Configuration:")
        print(f"   - Source index: {self.source_index}")
        print(f"   - Vector index: {self.vector_index}")
        print(f"   - Embedding model: {config.EMBEDDING_MODEL}")
        print(f"   - Embedding dimension: {config.EMBEDDING_DIMENSION}")
        
        try:
            # Initialize services
            self.initialize_services()
            
            # Create vector index
            self.create_vector_index()
            
            # Check source index
            print(f"\n📊 Checking source index...")
            source_count = self.get_source_model_count()
            print(f"   📥 Source models in {self.source_index}: {source_count}")
            
            if source_count == 0:
                print(f"❌ No models found in source index {self.source_index}")
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
            sys.exit(1)

def main():
    """Main function."""
    manager = SingleVectorIndexManager()
    manager.run()

if __name__ == "__main__":
    main()
