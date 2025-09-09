import time
from datetime import datetime
from elasticsearch import Elasticsearch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.embedding_service import EmbeddingService
from src.vector_index_handler import VectorIndexHandler
from src.hybrid_search import HybridSearch
from src.config import config


def get_all_models_from_existing_indices():
    """
    Get ALL models from your existing Elasticsearch indices.
    """
    
    # Connect to Elasticsearch
    es = Elasticsearch([config.get_elasticsearch_url()])
    
    all_models = []
    
    for source_index in config.SOURCE_INDICES:
        try:
            
            # Check if index exists
            if not es.indices.exists(index=source_index):
                print(f"   ⚠️  Index {source_index} does not exist, skipping...")
                continue
            
            # Get ALL documents from the index
            search_body = {
                "size": 1000,  # Get ALL models from each index
                "query": {"match_all": {}},
                "_source": [
                    "name", "description", "mlTask", "keywords", "platform", 
                    "db_identifier", "sharedBy", "license", "dateCreated"
                ]
            }
            
            response = es.search(index=source_index, body=search_body)
            
            for hit in response["hits"]["hits"]:
                model_data = hit["_source"]
                model_data["source_index"] = source_index
                all_models.append(model_data)
            
            
        except Exception as e:
            print(f"   ❌ Error reading {source_index}: {e}")
            continue
    
    return all_models

def test_all_search_methods_with_real_data():
    """
    Test all search methods with real data from your existing indices.
    """
    print("🧪 Testing Search Methods with Real Data")
    
    try:
        # Step 1: Get ALL models from existing indices
        all_models = get_all_models_from_existing_indices()
        
        if not all_models:
            print("❌ No models found. Make sure your existing indices have data.")
            return
        
        # Step 2: Initialize services
        print(f"🧠 Using embedding model: {config.EMBEDDING_MODEL}")
        embedding_service = EmbeddingService()
        vector_handler = VectorIndexHandler(embedding_service)
        hybrid_search = HybridSearch(embedding_service, vector_handler)
        
        # Step 3: Index ALL models
        print(f"📊 Indexing {len(all_models)} models...")
        
        # Group models by source index
        models_by_index = {}
        for model in all_models:
            source_index = model.get("source_index", "hf_models")
            if source_index not in models_by_index:
                models_by_index[source_index] = []
            models_by_index[source_index].append(model)
        
        # Index models for each source index
        total_indexed = 0
        for source_index, models in models_by_index.items():
            print(f"   📁 Indexing {len(models)} models in {source_index}...")
            result = vector_handler.bulk_index_models_with_vectors(models, source_index)
            total_indexed += result.get("success", 0)
            print(f"   ✅ Indexed {result.get('success', 0)} models successfully")
        
        print(f"📊 Total models indexed: {total_indexed}")
        
        # Wait for indexing to complete
        print("⏳ Waiting for indexing to complete...")
        time.sleep(3)
        
        # Step 4: Test all search methods with various queries
        print("🔍 Testing all search methods with real data...")
        
        test_queries = [
            {
                "query": "language understanding model",
                "description": "Looking for models that understand human language",
                "expected": "Should find BERT, GPT, RoBERTa, etc."
            },
            {
                "query": "image classification model", 
                "description": "Looking for models that classify images",
                "expected": "Should find ResNet, VGG, etc."
            },
            {
                "query": "transformer model",
                "description": "Looking for transformer-based models",
                "expected": "Should find BERT, GPT, RoBERTa, etc."
            },
            {
                "query": "BERT model",
                "description": "Looking specifically for BERT models",
                "expected": "Should find BERT variants"
            },
            {
                "query": "computer vision model",
                "description": "Looking for computer vision models",
                "expected": "Should find image processing models"
            }
        ]
        
        for test_case in test_queries:
            query = test_case["query"]
            description = test_case["description"]
            expected = test_case["expected"]
            
            print("\n", "-" * 60, "\n")
            print(f"🔍 Testing: '{query}' \n")
            
            # Test on each available index
            for source_index in models_by_index.keys():
                
                # Compare all methods
                results = hybrid_search.compare_search_methods(query, source_index, top_k=10)
                
                # Display results for each method
                for method in ["text", "vector", "hybrid"]:
                    method_results = results[method]
                    print(f"📊 {method.upper()}: {len(method_results)} results")
                    
                    if method_results:
                        for i, result in enumerate(method_results):  # Show all results
                            name = result["model_data"]["name"]
                            ml_task = result["model_data"].get("mlTask", [])
                            
                            if method == "hybrid":
                                score = result["combined_score"]
                                print(f"   {i+1}. {name} ({score:.3f})")
                            else:
                                score = result["score"]
                                print(f"   {i+1}. {name} ({score:.3f})")
                            
                            if ml_task:
                                print(f"      Tasks: {', '.join(ml_task[:3])}")  # Show first 3 tasks
                
                # Show timing
                timing = results["timing"]
                print(f"⏱️  Text: {timing['text_search']:.2f}s, Vector: {timing['vector_search']:.2f}s, Hybrid: {timing['hybrid_search']:.2f}s")
                print("💡 Compare the tasks to see which method found the most relevant models!")
        
        # Step 5: Test hybrid search with different weights
        print("\n", "-" * 60, "\n")
        print("🎯 Testing Weight Combinations: \n")
        
        query = "language understanding model"
        weight_combinations = [
            (0.1, 0.9, "Mostly Vector"),
            (0.3, 0.7, "Balanced"),
            (0.5, 0.5, "Equal Weight"),
            (0.7, 0.3, "Mostly Text")
        ]
        
        for text_weight, vector_weight, description in weight_combinations:
            print(f"📊 {description}:")
            
            # Test on first available index
            test_index = list(models_by_index.keys())[0]
            results = hybrid_search.hybrid_search(
                query, test_index, top_k=2,
                text_weight=text_weight, vector_weight=vector_weight
            )
            
            for i, result in enumerate(results):
                name = result["model_data"]["name"]
                combined_score = result["combined_score"]
                print(f"   {i+1}. {name} ({combined_score:.3f})")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to run the test.
    """
    test_all_search_methods_with_real_data()

if __name__ == "__main__":
    main()
