#!/usr/bin/env python3
"""
Dimension-Grouped Model Evaluation
Tests models grouped by embedding dimensions to avoid conflicts.
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from elasticsearch import Elasticsearch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.embedding_service import EmbeddingService
from src.vector_index_handler import VectorIndexHandler
from src.hybrid_search import HybridSearch
from src.config import config

@dataclass
class ModelConfig:
    name: str
    dimensions: int
    description: str

@dataclass
class EvaluationResult:
    model_name: str
    search_method: str
    query: str
    relevance_score: float
    task_match_score: float
    total_score: float
    execution_time: float

class DimensionGroupedEvaluator:
    """Evaluates models grouped by embedding dimensions."""
    
    def __init__(self):
        self.es = Elasticsearch([config.get_elasticsearch_url()])
        
        self.test_queries = [
            {
                "query": "language understanding model",
                "expected_tasks": ["text generation", "text classification", "question answering", "natural language understanding", "sentiment analysis"],
            },
            {
                "query": "image classification model", 
                "expected_tasks": ["image classification", "object detection", "image segmentation", "computer vision"],
            },
            {
                "query": "transformer model",
                "expected_tasks": ["text generation", "text classification", "translation", "summarization"],
            },
            {
                "query": "BERT model",
                "expected_tasks": ["text classification", "question answering", "named entity recognition", "natural language understanding"],
            },
            {
                "query": "computer vision model",
                "expected_tasks": ["image classification", "object detection", "image segmentation", "computer vision", "image generation"],
            }
        ]
        
        # Group models by dimensions
        self.model_groups = {
            384: [
                ModelConfig("sentence-transformers/all-MiniLM-L6-v2", 384, "Very Fast, Good Quality"),
                ModelConfig("sentence-transformers/all-MiniLM-L12-v2", 384, "Fast, Better Quality"),
                ModelConfig("sentence-transformers/all-distilroberta-v1", 768, "Balanced Speed/Quality"),  # Actually 768, but let's test
            ],
            768: [
                ModelConfig("sentence-transformers/all-mpnet-base-v2", 768, "Best Quality"),
                ModelConfig("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 768, "Multilingual, High Quality"),
            ]
        }
        
        self.search_methods = ["text", "vector", "hybrid"]
        self.all_results = []
    
    def get_vector_index_name(self, model_name: str, source_index: str) -> str:
        """Generate unique vector index name."""
        safe_model_name = model_name.replace("/", "_").replace("-", "_").lower()
        return f"vector_{safe_model_name}_{source_index}"
    
    def delete_all_vector_indices(self):
        """Delete all existing vector indices to start fresh."""
        print("🗑️  Deleting all existing vector indices...")
        
        try:
            # Get all indices
            indices = self.es.indices.get_alias("*")
            vector_indices = [name for name in indices.keys() if name.startswith("vector_")]
            
            if vector_indices:
                print(f"   Found {len(vector_indices)} vector indices to delete")
                self.es.indices.delete(index=vector_indices)
                print("   ✅ All vector indices deleted")
            else:
                print("   No vector indices found")
                
        except Exception as e:
            print(f"   ❌ Error deleting indices: {e}")
    
    def create_vector_index(self, index_name: str, dimensions: int):
        """Create a vector index with specific dimensions."""
        mapping = {
            "mappings": {
                "properties": {
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
                    
                    "model_vector": {
                        "type": "dense_vector",
                        "dims": dimensions,
                        "index": True,
                        "similarity": "cosine"
                    },
                    
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
        
        self.es.indices.create(index=index_name, body=mapping)
        print(f"   ✅ Created vector index: {index_name} ({dimensions}D)")
    
    def get_all_models_from_existing_indices(self) -> List[Dict[str, Any]]:
        """Get all models from existing indices."""
        all_models = []
        
        for source_index in config.SOURCE_INDICES:
            try:
                if not self.es.indices.exists(index=source_index):
                    continue
                
                search_body = {
                    "size": 1000,
                    "query": {"match_all": {}},
                    "_source": [
                        "name", "description", "mlTask", "keywords", "platform", 
                        "db_identifier", "sharedBy", "license"
                    ]
                }
                
                response = self.es.search(index=source_index, body=search_body)
                
                for hit in response["hits"]["hits"]:
                    model_data = hit["_source"]
                    model_data["source_index"] = source_index
                    all_models.append(model_data)
                
            except Exception as e:
                print(f"   ❌ Error reading {source_index}: {e}")
                continue
        
        return all_models
    
    def index_data_for_model(self, model_config: ModelConfig) -> bool:
        """Index data for a specific model."""
        print(f"📊 Indexing data for {model_config.name}...")
        
        # Initialize embedding service
        embedding_service = EmbeddingService(model_config.name)
        
        # Get all models
        all_models = self.get_all_models_from_existing_indices()
        if not all_models:
            print("❌ No models found")
            return False
        
        # Group models by source index
        models_by_index = {}
        for model in all_models:
            source_index = model.get("source_index", "hf_models")
            if source_index not in models_by_index:
                models_by_index[source_index] = []
            models_by_index[source_index].append(model)
        
        # Index for each source
        total_indexed = 0
        for source_index, models in models_by_index.items():
            vector_index_name = self.get_vector_index_name(model_config.name, source_index)
            
            # Create index if it doesn't exist
            if not self.es.indices.exists(index=vector_index_name):
                self.create_vector_index(vector_index_name, model_config.dimensions)
            
            # Index models
            indexed_count = self._bulk_index_models(models, vector_index_name, embedding_service, model_config.name)
            total_indexed += indexed_count
        
        print(f"✅ Indexed {total_indexed} models for {model_config.name}")
        return True
    
    def _bulk_index_models(self, models_data: List[Dict[str, Any]], 
                          vector_index_name: str, 
                          embedding_service: EmbeddingService,
                          model_name: str) -> int:
        """Bulk index models with vectors."""
        if not models_data:
            return 0
        
        success_count = 0
        batch_size = 50
        
        for i in range(0, len(models_data), batch_size):
            batch = models_data[i:i + batch_size]
            
            try:
                # Prepare batch
                batch_texts = []
                for model_data in batch:
                    searchable_text = self._prepare_searchable_text(model_data)
                    batch_texts.append(searchable_text)
                
                # Generate embeddings
                vectors = embedding_service.encode_batch(batch_texts)
                
                # Create documents
                batch_docs = []
                for model_data, vector in zip(batch, vectors):
                    doc_id = model_data.get("db_identifier", f"test_{time.time()}")
                    
                    doc = {
                        "_index": vector_index_name,
                        "_id": doc_id,
                        "_source": {
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
                            
                            "model_vector": vector,
                            "vector_created_at": int(time.time() * 1000),
                            "embedding_model": model_name,
                            "source_index": model_data.get("source_index", "")
                        }
                    }
                    batch_docs.append(doc)
                
                # Bulk index
                from elasticsearch.helpers import bulk
                success, failed = bulk(self.es, batch_docs)
                success_count += success
                
            except Exception as e:
                print(f"   ❌ Failed to process batch: {e}")
                continue
        
        return success_count
    
    def _prepare_searchable_text(self, model_data: Dict[str, Any]) -> str:
        """Prepare searchable text from model data."""
        text_parts = []
        
        for field in config.EMBEDDING_FIELDS:
            value = model_data.get(field, "")
            
            if isinstance(value, list):
                text_parts.extend([str(v) for v in value if v])
            elif value:
                text_parts.append(str(value))
        
        return config.FIELD_SEPARATOR.join(text_parts)
    
    def evaluate_search_results(self, results: List[Dict[str, Any]], 
                              expected_tasks: List[str]) -> Tuple[float, float]:
        """Evaluate search results based on ML task relevance."""
        if not results:
            print(f"   ⚠️  No results returned for query")
            return 0.0, 0.0
        
        task_matches = 0
        total_results = len(results)
        
        for result in results:
            # Get mlTask from model_data field (nested structure)
            model_data = result.get("model_data", {})
            model_tasks = model_data.get("mlTask", [])
            if not model_tasks:
                continue
            
            # Check if any model task matches expected tasks
            for task in model_tasks:
                if any(expected_task.lower() in task.lower() for expected_task in expected_tasks):
                    task_matches += 1
                    break
        
        # Calculate scores
        task_match_score = task_matches / total_results if total_results > 0 else 0.0
        relevance_score = min(task_match_score * 1.2, 1.0)
        
        return relevance_score, task_match_score
    
    def test_model_group(self, dimension: int, models: List[ModelConfig]):
        """Test all models in a dimension group."""
        print(f"\n🧪 Testing {dimension}D Models")
        print("=" * 60)
        
        for model_config in models:
            print(f"\n📊 Model: {model_config.name}")
            print(f"   Description: {model_config.description}")
            print("-" * 50)
            
            # Index data for this model
            if not self.index_data_for_model(model_config):
                print(f"❌ Failed to index data for {model_config.name}")
                continue
            
            # Wait for indexing
            time.sleep(2)
            
            # Initialize services
            embedding_service = EmbeddingService(model_config.name)
            vector_handler = VectorIndexHandler(embedding_service)
            hybrid_search = HybridSearch(embedding_service, vector_handler)
            
            # Override vector indices for this model
            vector_indices = {}
            for source_index in config.SOURCE_INDICES:
                vector_indices[source_index] = self.get_vector_index_name(model_config.name, source_index)
            vector_handler.vector_indices = vector_indices
            
            # Test each search method
            for search_method in self.search_methods:
                print(f"\n🔍 {search_method.upper()} Search:")
                
                method_scores = []
                
                # Test each query
                for test_case in self.test_queries:
                    query = test_case["query"]
                    expected_tasks = test_case["expected_tasks"]
                    
                    start_time = time.time()
                    
                    try:
                        if search_method == "text":
                            results = hybrid_search.text_search(query, "hf_models", top_k=5)
                        elif search_method == "vector":
                            results = hybrid_search.vector_search(query, "hf_models", top_k=5)
                        else:  # hybrid
                            results = hybrid_search.hybrid_search(query, "hf_models", top_k=5)
                        
                        execution_time = time.time() - start_time
                        
                        # Evaluate results
                        relevance_score, task_match_score = self.evaluate_search_results(
                            results, expected_tasks
                        )
                        
                        total_score = (relevance_score + task_match_score) / 2
                        method_scores.append(total_score)
                        
                        # Store result
                        evaluation_result = EvaluationResult(
                            model_name=model_config.name,
                            search_method=search_method,
                            query=query,
                            relevance_score=relevance_score,
                            task_match_score=task_match_score,
                            total_score=total_score,
                            execution_time=execution_time
                        )
                        
                        self.all_results.append(evaluation_result)
                        
                        print(f"   '{query}': {total_score:.3f} ({execution_time:.2f}s)")
                        
                    except Exception as e:
                        print(f"   '{query}': ERROR - {e}")
                        continue
                
                # Show average for this method
                if method_scores:
                    avg_score = sum(method_scores) / len(method_scores)
                    print(f"   Average: {avg_score:.3f}")
        
        # Clean up indices for this dimension group
        print(f"\n🗑️  Cleaning up {dimension}D indices...")
        self.delete_all_vector_indices()
    
    def run_evaluation(self):
        """Run the complete evaluation."""
        print("🚀 Dimension-Grouped Model Evaluation")
        print("=" * 70)
        print("This will test models grouped by embedding dimensions")
        print("to avoid dimension conflicts and enable proper vector search evaluation")
        print("=" * 70)
        
        # Test 384D models first
        if 384 in self.model_groups:
            self.test_model_group(384, self.model_groups[384])
        
        # Test 768D models second
        if 768 in self.model_groups:
            self.test_model_group(768, self.model_groups[768])
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        print("\n" + "=" * 70)
        print("📊 FINAL EVALUATION REPORT")
        print("=" * 70)
        
        # Group results by model
        model_results = {}
        for result in self.all_results:
            if result.model_name not in model_results:
                model_results[result.model_name] = []
            model_results[result.model_name].append(result)
        
        # Calculate metrics for each model
        print("\n🏆 MODEL PERFORMANCE RANKING:")
        print("-" * 50)
        
        model_scores = []
        for model_name, results in model_results.items():
            avg_score = sum(r.total_score for r in results) / len(results)
            avg_time = sum(r.execution_time for r in results) / len(results)
            total_tests = len(results)
            
            model_scores.append({
                'model': model_name,
                'avg_score': avg_score,
                'avg_time': avg_time,
                'total_tests': total_tests
            })
        
        # Sort by average score
        model_scores.sort(key=lambda x: x['avg_score'], reverse=True)
        
        for i, model_score in enumerate(model_scores, 1):
            print(f"{i}. {model_score['model']}")
            print(f"   Average Score: {model_score['avg_score']:.3f}")
            print(f"   Average Time: {model_score['avg_time']:.3f}s")
            print(f"   Tests: {model_score['total_tests']}")
            print()
        
        # Search method comparison
        print("\n🔍 SEARCH METHOD COMPARISON:")
        print("-" * 50)
        
        method_scores = {}
        for result in self.all_results:
            if result.search_method not in method_scores:
                method_scores[result.search_method] = []
            method_scores[result.search_method].append(result.total_score)
        
        for method, scores in method_scores.items():
            avg_score = sum(scores) / len(scores)
            print(f"{method.upper()}: {avg_score:.3f} (from {len(scores)} tests)")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save detailed results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dimension_grouped_results_{timestamp}.json"
        
        results_data = []
        for result in self.all_results:
            results_data.append({
                'model_name': result.model_name,
                'search_method': result.search_method,
                'query': result.query,
                'relevance_score': result.relevance_score,
                'task_match_score': result.task_match_score,
                'total_score': result.total_score,
                'execution_time': result.execution_time
            })
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")

def main():
    """Main function."""
    evaluator = DimensionGroupedEvaluator()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
