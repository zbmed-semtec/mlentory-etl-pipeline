#!/usr/bin/env python3
import sys
import os
from typing import List, Dict, Any, Set
from tqdm import tqdm
from datetime import datetime

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.embedding_service import EmbeddingService
from src.config import config

class NgramEnhancedVectorIndexManager:
    """
    Smart manager for n-gram enhanced multi-vector index creation and maintenance.
    Creates multiple n-gram enhanced vectors per model for better partial matching.
    """
    
    def __init__(self):
        """Initialize the n-gram enhanced vector index manager."""
        self.embedding_service = None
        self.es = None
        self.source_index = "hf_models"
        self.vector_index = "vector_multi_ngram_hf_models"
        self.known_models = set()
        
    def initialize_services(self):
        """Initialize embedding service and Elasticsearch connection."""
        print("🔧 Initializing services...")
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService()
        print(f"✅ Embedding service ready")
        
        # Initialize Elasticsearch connection
        from elasticsearch import Elasticsearch
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
        
        # Learn model names from data
        self._learn_model_names_from_data()
        
    def _learn_model_names_from_data(self):
        """
        Learn model names from actual data in Elasticsearch indices.
        This is the 'learn from data' approach for smart query detection.
        """
        print("🧠 Learning model names from your data...")
        
        try:
            # Get all model names from the source index
            response = self.es.search(
                index=self.source_index,
                body={
                    "size": 1000,  # Adjust based on your data size
                    "_source": ["name"],
                    "query": {"match_all": {}}
                }
            )
            
            count = 0
            for hit in response["hits"]["hits"]:
                name = hit["_source"].get("name", "")
                if name:
                    # Clean the name and add to our set
                    clean_name = name.strip().lower()
                    self.known_models.add(clean_name)
                    count += 1
            
            print(f"🎯 Learned {count} unique model names from {self.source_index}")
            
            # Show some examples
            if self.known_models:
                examples = list(self.known_models)[:5]
                print(f"📝 Examples: {', '.join(examples)}")
                        
        except Exception as e:
            print(f"❌ Error learning model names: {e}")
    
    def detect_query_type(self, query: str) -> str:
        """
        Detect query type using learned model names and heuristics.
        
        Args:
            query: The search query
            
        Returns:
            str: Query type ("lexical", "mixed", or "semantic")
        """
        query_clean = query.strip().lower()
        words = query_clean.split()
        word_count = len(words)
        char_count = len(query_clean)
        
        # Check against known models (the magic happens here!)
        first_word = words[0] if words else ""
        is_known_model = first_word in self.known_models
        
        if is_known_model:
            if word_count == 1:
                return "lexical"
            elif word_count <= 3:
                return "mixed"
        
        # Fallback heuristics for unknown models
        if char_count <= 6 and word_count <= 1:
            return "lexical"
        elif char_count <= 15 and word_count <= 3:
            return "mixed"
        else:
            return "semantic"
    
    def enhance_query_with_ngrams(self, query: str, min_gram: int = 3, max_gram: int = 10) -> str:
        """
        Add edge n-grams to the query to improve vector search for partial matches.
        
        Args:
            query: Original search query
            min_gram: Minimum n-gram size
            max_gram: Maximum n-gram size
            
        Returns:
            str: Enhanced query with n-grams
        """
        # Clean the query
        query_clean = query.strip().lower()
        
        # Generate edge n-grams
        ngrams = []
        for length in range(min_gram, min(max_gram + 1, len(query_clean) + 1)):
            ngrams.append(query_clean[:length])
        
        # Combine original query with n-grams
        enhanced_query = f"{query} {' '.join(ngrams)}"
        
        return enhanced_query
    
    def smart_enhance_query(self, query: str) -> str:
        """
        Smart query enhancement based on query type detection.
        
        Args:
            query: Original search query
            
        Returns:
            str: Enhanced query
        """
        query_type = self.detect_query_type(query)
        
        if query_type == "lexical":
            # Short queries benefit from n-grams
            enhanced = self.enhance_query_with_ngrams(query, min_gram=2, max_gram=8)
        elif query_type == "mixed":
            # Medium queries get light enhancement
            words = query.split()
            if words:
                model_part = words[0]
                enhanced_model = self.enhance_query_with_ngrams(model_part, min_gram=3, max_gram=6)
                enhanced = f"{enhanced_model} {' '.join(words[1:])}"
            else:
                enhanced = query
        else:  # semantic
            # Long queries use as-is (no n-grams)
            enhanced = query
        
        return enhanced, query_type
    
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
    
    def vector_index_exists(self) -> bool:
        """Check if the vector index exists."""
        try:
            return self.es.indices.exists(index=self.vector_index)
        except Exception as e:
            print(f"❌ Failed to check if vector index exists: {e}")
            return False
    
    def create_ngram_enhanced_vector_index_mapping(self):
        """Create the index mapping for n-gram enhanced multi-vector structure."""
        mapping = {
            "mappings": {
                "properties": {
                    # Original metadata fields
                    "db_identifier": {"type": "keyword"},
                    "name": {"type": "text", "analyzer": "standard"},
                    "description": {"type": "text", "analyzer": "standard"},
                    "license": {"type": "keyword"},
                    "sharedBy": {"type": "text", "analyzer": "standard"},
                    "mlTask": {"type": "text", "analyzer": "standard"},
                    "keywords": {"type": "text", "analyzer": "standard"},
                    "relatedDatasets": {"type": "text", "analyzer": "standard"},
                    "baseModels": {"type": "text", "analyzer": "standard"},
                    "platform": {"type": "keyword"},
                    "dateCreated": {"type": "date"},
                    
                    # N-gram enhanced multi-vector fields
                    "name_vector": {
                        "type": "dense_vector",
                        "dims": config.EMBEDDING_DIMENSION,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "task_vector": {
                        "type": "dense_vector",
                        "dims": config.EMBEDDING_DIMENSION,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "creator_vector": {
                        "type": "dense_vector",
                        "dims": config.EMBEDDING_DIMENSION,
                        "index": True,
                        "similarity": "cosine"
                    },
                    
                    # Metadata fields
                    "vector_created_at": {"type": "date"},
                    "embedding_model": {"type": "keyword"},
                    "source_index": {"type": "keyword"},
                    "ngram_enhanced": {"type": "keyword"}
                }
            }
        }
        
        return mapping
    
    def create_vector_index(self):
        """Create the n-gram enhanced vector index with proper mapping."""
        print(f"🔧 Creating n-gram enhanced vector index: {self.vector_index}")
        
        try:
            # Delete existing index if it exists
            if self.es.indices.exists(index=self.vector_index):
                print(f"  🗑️  Deleting existing index...")
                self.es.indices.delete(index=self.vector_index)
            
            # Create new index with mapping
            mapping = self.create_ngram_enhanced_vector_index_mapping()
            self.es.indices.create(index=self.vector_index, body=mapping)
            
            print(f"  ✅ Index {self.vector_index} created successfully")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to create index: {e}")
            return False
    
    def populate_vector_index(self, sample_size: int = None, batch_size: int = 100):
        """
        Populate the vector index with n-gram enhanced vectors.
        
        Args:
            sample_size: Number of models to process (None for all)
            batch_size: Batch size for processing
        """
        print(f"📊 Populating n-gram enhanced vector index...")
        
        try:
            # Get total count
            total_models = self.get_source_model_count()
            if total_models == 0:
                print("❌ No models found in source index")
                return False
            
            # Determine how many to process
            if sample_size:
                models_to_process = min(sample_size, total_models)
                print(f"  📝 Processing {models_to_process} models (sample)")
            else:
                models_to_process = total_models
                print(f"  📝 Processing all {models_to_process} models")
            
            # Get models from source index
            print("  📥 Fetching models from source index...")
            response = self.es.search(
                index=self.source_index,
                body={
                    "size": models_to_process,
                    "query": {"match_all": {}},
                    "_source": [
                        "db_identifier", "name", "description", "license", "sharedBy",
                        "mlTask", "keywords", "relatedDatasets", "baseModels", "platform", "dateCreated"
                    ]
                }
            )
            
            models = response["hits"]["hits"]
            print(f"  ✅ Fetched {len(models)} models")
            
            # Process models in batches
            print("  🔄 Processing models with n-gram enhancement...")
            
            from elasticsearch.helpers import bulk
            import time
            
            documents = []
            processed_count = 0
            
            for i, hit in enumerate(tqdm(models, desc="Processing models")):
                model_data = hit["_source"]
                
                # Prepare searchable text for each vector type
                name_text = model_data.get("name", "")
                task_text = " ".join(model_data.get("mlTask", []))
                creator_text = model_data.get("sharedBy", "")
                
                # Apply n-gram enhancement to each text
                enhanced_name, name_type = self.smart_enhance_query(name_text)
                enhanced_task, task_type = self.smart_enhance_query(task_text)
                enhanced_creator, creator_type = self.smart_enhance_query(creator_text)
                
                # Generate vectors from enhanced text
                name_vector = self.embedding_service.encode_text(enhanced_name)
                task_vector = self.embedding_service.encode_text(enhanced_task)
                creator_vector = self.embedding_service.encode_text(enhanced_creator)
                
                # Create document
                doc = {
                    "_index": self.vector_index,
                    "_id": hit["_id"],
                    "_source": {
                        **model_data,
                        "name_vector": name_vector,
                        "task_vector": task_vector,
                        "creator_vector": creator_vector,
                        "vector_created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "embedding_model": config.EMBEDDING_MODEL,
                        "source_index": self.source_index,
                        "ngram_enhanced": "true"
                    }
                }
                documents.append(doc)
                processed_count += 1
                
                # Bulk index when batch is full
                if len(documents) >= batch_size:
                    success_count, failed_items = bulk(self.es, documents, chunk_size=batch_size)
                    if failed_items:
                        print(f"    ⚠️  {len(failed_items)} documents failed in batch")
                    documents = []
            
            # Index remaining documents
            if documents:
                success_count, failed_items = bulk(self.es, documents, chunk_size=batch_size)
                if failed_items:
                    print(f"    ⚠️  {len(failed_items)} documents failed in final batch")
            
            # Refresh index
            self.es.indices.refresh(index=self.vector_index)
            
            print(f"  ✅ Successfully processed {processed_count} models")
            print(f"  📊 Final index contains {self.get_vector_model_count()} documents")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to populate index: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify_index(self):
        """Verify the created index and show sample data."""
        print(f"🔍 Verifying n-gram enhanced vector index...")
        
        try:
            # Check document count
            doc_count = self.get_vector_model_count()
            print(f"  📊 Document count: {doc_count}")
            
            if doc_count > 0:
                # Get sample documents
                response = self.es.search(
                    index=self.vector_index,
                    body={
                        "size": 3,
                        "_source": ["name", "ngram_enhanced", "embedding_model", "source_index"]
                    }
                )
                
                print(f"  📝 Sample documents:")
                for i, hit in enumerate(response["hits"]["hits"], 1):
                    doc = hit["_source"]
                    print(f"    {i}. {doc.get('name', 'N/A')} (ngram_enhanced: {doc.get('ngram_enhanced', 'N/A')})")
                
                # Test n-gram enhancement
                print(f"  🧪 Testing n-gram enhancement:")
                test_queries = ["bert", "gpt-2", "Language Understanding Model"]
                for query in test_queries:
                    enhanced, query_type = self.smart_enhance_query(query)
                    print(f"    '{query}' → Type: {query_type}, Enhanced: '{enhanced}'")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Verification failed: {e}")
            return False
    
    def run_full_process(self, sample_size: int = None):
        """Run the complete process: create index, populate, and verify."""
        print("🚀 Starting N-gram Enhanced Multi-Vector Index Creation")
        print("=" * 60)
        
        try:
            # Initialize services
            self.initialize_services()
            
            # Create index
            if not self.create_vector_index():
                return False
            
            # Populate index
            if not self.populate_vector_index(sample_size):
                return False
            
            # Verify index
            if not self.verify_index():
                return False
            
            print("\n🎉 N-gram Enhanced Multi-Vector Index Creation Complete!")
            print(f"📊 Index: {self.vector_index}")
            print(f"📊 Models: {self.get_vector_model_count()}")
            print(f"🧠 Learned model names: {len(self.known_models)}")
            print("\n💡 You can now use search_multi_cli_with_ngram.py for n-gram enhanced search!")
            
            return True
            
        except Exception as e:
            print(f"❌ Process failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create N-gram Enhanced Multi-Vector Indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 create_vector_indices_with_ngrams.py
  python3 create_vector_indices_with_ngrams.py --sample-size 100
  python3 create_vector_indices_with_ngrams.py --sample-size 50 --batch-size 25
        """
    )
    
    parser.add_argument(
        "--sample-size", "-s",
        type=int,
        help="Number of models to process (default: all models)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Create manager and run process
    manager = NgramEnhancedVectorIndexManager()
    success = manager.run_full_process(sample_size=args.sample_size)
    
    if success:
        print("\n✅ All done!")
        sys.exit(0)
    else:
        print("\n❌ Process failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
