#!/usr/bin/env python3
"""
Native Elasticsearch Multi-Vector Similarity Search CLI

This CLI uses Elasticsearch's native vector similarity search capabilities with multiple vectors:
- name_vector: Vector for the model name only
- task_vector: Vector for mlTask + keywords combined  
- creator_vector: Vector for sharedBy field only
- Uses the vector_multi_hf_models index for optimal multi-vector search

Usage:
    python3 search_multi_cli_native.py "your search query"
    python3 search_multi_cli_native.py --interactive
    python3 search_multi_cli_native.py "query" --search-type hybrid --top-k 10

Example:
    Query: "BERT-based language models for biomedical text"
    Multi-vector search uses name, task, and creator vectors for comprehensive matching
"""

import sys
import argparse
import time
import re
from typing import List, Dict, Any, Optional, Set
from elasticsearch import Elasticsearch
from src.embedding_service import EmbeddingService
from src.config import config
from console_logger import start_console_logging, stop_console_logging, log_message

class GroundTruthParser:
    """
    Parser for ground truth data from Groundtruth.txt file.
    """
    
    def __init__(self, ground_truth_file: str = "Groundtruth.txt"):
        """Initialize the ground truth parser."""
        self.ground_truth_file = ground_truth_file
        self.ground_truth_data = {}
        self._load_ground_truth()
    
    def _load_ground_truth(self):
        """Load ground truth data from file."""
        try:
            with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the ground truth data
            sections = content.split('Query: ')
            
            for section in sections[1:]:  # Skip first empty section
                lines = section.strip().split('\n')
                if not lines:
                    continue
                
                query = lines[0].strip()
                results = []
                
                # Find results section
                in_results = False
                for line in lines[1:]:
                    if line.strip() == 'Results:':
                        in_results = True
                        continue
                    elif line.strip().startswith('---'):
                        break
                    elif in_results and line.strip():
                        # Parse result line: "org/model-name - Task"
                        result_line = line.strip()
                        if ' - ' in result_line:
                            model_name = result_line.split(' - ')[0].strip()
                            results.append(model_name)
                
                if results:
                    self.ground_truth_data[query] = results
                    
            print(f"✅ Loaded ground truth for {len(self.ground_truth_data)} queries")
            
        except FileNotFoundError:
            print(f"⚠️  Ground truth file {self.ground_truth_file} not found")
            self.ground_truth_data = {}
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            self.ground_truth_data = {}
    
    def get_ground_truth_models(self, query: str) -> Set[str]:
        """
        Get ground truth models for a query.
        
        Args:
            query: Search query
            
        Returns:
            Set of ground truth model names
        """
        # Try exact match first
        if query in self.ground_truth_data:
            return set(self.ground_truth_data[query])
        
        # Try case-insensitive match
        query_lower = query.lower()
        for gt_query, models in self.ground_truth_data.items():
            if gt_query.lower() == query_lower:
                return set(models)
        
        # Try partial match (query contains ground truth query or vice versa)
        for gt_query, models in self.ground_truth_data.items():
            if (query_lower in gt_query.lower() or 
                gt_query.lower() in query_lower):
                return set(models)
        
        return set()
    
    def calculate_recall(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate recall metrics for search results against ground truth.
        
        Args:
            query: Search query
            search_results: List of search results
            
        Returns:
            Dict with recall metrics
        """
        ground_truth_models = self.get_ground_truth_models(query)
        
        if not ground_truth_models:
            return {
                "has_ground_truth": False,
                "ground_truth_count": 0,
                "found_count": 0,
                "recall": 0.0,
                "found_models": [],
                "missing_models": []
            }
        
        # Extract model names from search results
        found_models = set()
        for result in search_results:
            model_data = result.get("model_data", {})
            model_name = model_data.get("name", "")
            if model_name:
                found_models.add(model_name)
        
        # Calculate recall
        found_ground_truth = ground_truth_models.intersection(found_models)
        recall = len(found_ground_truth) / len(ground_truth_models) if ground_truth_models else 0.0
        
        return {
            "has_ground_truth": True,
            "ground_truth_count": len(ground_truth_models),
            "found_count": len(found_ground_truth),
            "recall": recall,
            "found_models": list(found_ground_truth),
            "missing_models": list(ground_truth_models - found_ground_truth)
        }

class MultiVectorSearchHandler:
    """
    Handler for native Elasticsearch multi-vector similarity search.
    
    This class uses Elasticsearch's native dense_vector fields and KNN search
    with multiple vector fields for more precise and comprehensive search.
    """
    
    def __init__(self):
        """Initialize the multi-vector search handler."""
        self.es = None
        self.embedding_service = None
        
        # Connect to Elasticsearch
        self._connect_to_elasticsearch()
        
        # Initialize embedding service
        self._initialize_embedding_service()
    
    def _connect_to_elasticsearch(self):
        """Connect to Elasticsearch."""
        try:
            self.es = Elasticsearch(
                [config.get_elasticsearch_url()],
                basic_auth=(config.ELASTICSEARCH_USERNAME, config.ELASTICSEARCH_PASSWORD),
                request_timeout=30,
                max_retries=3,
                retry_on_timeout=True,
            )
            
            if not self.es.ping():
                raise Exception("Cannot connect to Elasticsearch")
                
            print("✅ Connected to Elasticsearch")
            
        except Exception as e:
            print(f"❌ Failed to connect to Elasticsearch: {e}")
            raise
    
    def _initialize_embedding_service(self):
        """Initialize the embedding service."""
        try:
            self.embedding_service = EmbeddingService()
            print("✅ Embedding service initialized")
        except Exception as e:
            print(f"⚠️  Failed to initialize embedding service: {e}")
            print("   Vector search will be disabled")
            self.embedding_service = None
    
    def find_multi_vector_index(self, source_index: str) -> Optional[str]:
        """
        Find the multi-vector index for a source index.
        
        Args:
            source_index: The source index name (e.g., "hf_models")
            
        Returns:
            str: The multi-vector index name if found, None otherwise
        """
        try:
            # Get all indices
            all_indices = self.es.cat.indices(format='json')
            
            # Look for multi-vector indices
            vector_index_patterns = [
                f"vector_multi_{source_index}",
            ]
            
            for pattern in vector_index_patterns:
                matching_indices = [idx['index'] for idx in all_indices if idx['index'] == pattern]
                if matching_indices:
                    return matching_indices[0]
            
            print(f"⚠️  No multi-vector index found for {source_index}")
            print(f"   Expected patterns: {vector_index_patterns}")
            print(f"   Available vector indices: {[idx['index'] for idx in all_indices if 'vector' in idx['index']]}")
            return None
            
        except Exception as e:
            print(f"❌ Error finding multi-vector index: {e}")
            return None
    
    def text_search(self, query: str, source_index: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform native text search using Elasticsearch's multi_match.
        
        Args:
            query: Search query string
            source_index: Source index name
            top_k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        try:
            search_body = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": config.EMBEDDING_FIELDS,
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "_source": [
                    "name", "description", "mlTask", "keywords", "platform", 
                    "db_identifier", "sharedBy", "license"
                ]
            }
            
            response = self.es.search(index=source_index, body=search_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "score": hit["_score"],
                    "model_data": hit["_source"],
                    "db_identifier": hit["_source"].get("db_identifier", ""),
                    "search_type": "text"
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Native text search failed: {e}")
            return []
    
    def multi_vector_search(self, query: str, source_index: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform native multi-vector search using Elasticsearch's KNN with multiple vector fields.
        
        Args:
            query: Search query string
            source_index: Source index name
            top_k: Number of results to return
            
        Returns:
            List of search results with similarity scores
        """
        if not self.embedding_service:
            print("❌ Embedding service not available for vector search")
            return []
        
        try:
            # Find the multi-vector index
            vector_index = self.find_multi_vector_index(source_index)
            if not vector_index:
                print(f"❌ No multi-vector index found for {source_index}")
                return []
            
            # Generate query vector
            query_vector = self.embedding_service.encode_text(query)
            
            # Build native KNN search query with multiple vector fields
            search_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [
                            # Name vector search
                            {
                                "knn": {
                                    "field": "name_vector",
                                    "query_vector": query_vector,
                                    "k": top_k,
                                    "num_candidates": config.VECTOR_SEARCH_CANDIDATES,
                                    "boost": 1.0
                                }
                            },
                            # Task vector search
                            {
                                "knn": {
                                    "field": "task_vector",
                                    "query_vector": query_vector,
                                    "k": top_k,
                                    "num_candidates": config.VECTOR_SEARCH_CANDIDATES,
                                    "boost": 0.8
                                }
                            },
                            # Creator vector search
                            {
                                "knn": {
                                    "field": "creator_vector",
                                    "query_vector": query_vector,
                                    "k": top_k,
                                    "num_candidates": config.VECTOR_SEARCH_CANDIDATES,
                                    "boost": 0.6
                                }
                            }
                        ]
                    }
                },
                "_source": [
                    "name", "description", "mlTask", "keywords", "platform", 
                    "db_identifier", "sharedBy", "license"
                ]
            }
            
            # Execute search
            response = self.es.search(index=vector_index, body=search_body)
            
            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "score": hit["_score"],
                    "model_data": hit["_source"],
                    "db_identifier": hit["_source"].get("db_identifier", ""),
                    "search_type": "multi_vector"
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Native multi-vector search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, source_index: str, top_k: int = 5, 
                     text_weight: float = 0.4, vector_weight: float = 0.6) -> List[Dict[str, Any]]:
        """
        Perform native hybrid search combining text and multi-vector search.
        
        Args:
            query: Search query string
            source_index: Source index name
            top_k: Number of results to return
            text_weight: Weight for text search (default: 0.3)
            vector_weight: Weight for vector search (default: 0.7)
            
        Returns:
            List of search results with combined scores
        """
        if not self.embedding_service:
            print("❌ Embedding service not available for hybrid search")
            return self.text_search(query, source_index, top_k)
        
        try:
            # Find the multi-vector index
            vector_index = self.find_multi_vector_index(source_index)
            if not vector_index:
                print(f"❌ No multi-vector index found for {source_index}, falling back to text search")
                return self.text_search(query, source_index, top_k)
            
            # Generate query vector
            query_vector = self.embedding_service.encode_text(query)
            
            # Build native hybrid search query
            search_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [
                            # Text search with weight
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": config.EMBEDDING_FIELDS,
                                    "type": "best_fields",
                                    "fuzziness": "AUTO",
                                    "boost": text_weight
                                }
                            },
                            # Multi-vector search with weight
                            {
                                "bool": {
                                    "should": [
                                        # Name vector search
                                        {
                                            "knn": {
                                                "field": "name_vector",
                                                "query_vector": query_vector,
                                                "k": top_k,
                                                "num_candidates": config.VECTOR_SEARCH_CANDIDATES,
                                                "boost": vector_weight * 1.0
                                            }
                                        },
                                        # Task vector search
                                        {
                                            "knn": {
                                                "field": "task_vector",
                                                "query_vector": query_vector,
                                                "k": top_k,
                                                "num_candidates": config.VECTOR_SEARCH_CANDIDATES,
                                                "boost": vector_weight * 0.8
                                            }
                                        },
                                        # Creator vector search
                                        {
                                            "knn": {
                                                "field": "creator_vector",
                                                "query_vector": query_vector,
                                                "k": top_k,
                                                "num_candidates": config.VECTOR_SEARCH_CANDIDATES,
                                                "boost": vector_weight * 0.6
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                },
                "_source": [
                    "name", "description", "mlTask", "keywords", "platform", 
                    "db_identifier", "sharedBy", "license"
                ]
            }
            
            # Execute search
            response = self.es.search(index=vector_index, body=search_body)
            
            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "score": hit["_score"],
                    "model_data": hit["_source"],
                    "db_identifier": hit["_source"].get("db_identifier", ""),
                    "search_type": "hybrid"
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Native hybrid search failed: {e}")
            # Fallback to text search
            return self.text_search(query, source_index, top_k)
    
    def search_all_methods(self, query: str, index_name: str, top_k: int = 5, 
                          ground_truth_parser: Optional[GroundTruthParser] = None) -> Dict[str, Any]:
        """
        Search using all three native methods and return results with ground truth comparison.
        
        Args:
            query: Search query string
            index_name: Index to search in
            top_k: Number of results to return
            ground_truth_parser: Optional ground truth parser for evaluation
            
        Returns:
            Dict containing results from all search methods with ground truth metrics
        """
        results = {}
        
        # Text Search
        try:
            start_time = time.time()
            text_results = self.text_search(query, index_name, top_k)
            text_time = time.time() - start_time
            
            # Calculate ground truth metrics if parser is available
            text_gt_metrics = {}
            if ground_truth_parser:
                text_gt_metrics = ground_truth_parser.calculate_recall(query, text_results)
            
            results["text_search"] = {
                "results": text_results,
                "time": text_time,
                "count": len(text_results),
                "ground_truth": text_gt_metrics
            }
        except Exception as e:
            print(f"❌ Text search failed: {e}")
            results["text_search"] = {"results": [], "time": 0, "count": 0, "ground_truth": {}}
        
        # Multi-Vector Search
        try:
            start_time = time.time()
            vector_results = self.multi_vector_search(query, index_name, top_k)
            vector_time = time.time() - start_time
            
            # Calculate ground truth metrics if parser is available
            vector_gt_metrics = {}
            if ground_truth_parser:
                vector_gt_metrics = ground_truth_parser.calculate_recall(query, vector_results)
            
            results["multi_vector_search"] = {
                "results": vector_results,
                "time": vector_time,
                "count": len(vector_results),
                "ground_truth": vector_gt_metrics
            }
            
            if not vector_results:
                print(f"⚠️  Multi-vector search returned no results. Index may not exist.")
                print(f"💡 Make sure the multi-vector index exists with name_vector, task_vector, creator_vector fields.")
                
        except Exception as e:
            print(f"❌ Multi-vector search failed: {e}")
            print(f"💡 Make sure the multi-vector index exists with name_vector, task_vector, creator_vector fields.")
            results["multi_vector_search"] = {"results": [], "time": 0, "count": 0, "ground_truth": {}}
        
        # Hybrid Search
        try:
            start_time = time.time()
            hybrid_results = self.hybrid_search(query, index_name, top_k)
            hybrid_time = time.time() - start_time
            
            # Calculate ground truth metrics if parser is available
            hybrid_gt_metrics = {}
            if ground_truth_parser:
                hybrid_gt_metrics = ground_truth_parser.calculate_recall(query, hybrid_results)
            
            results["hybrid_search"] = {
                "results": hybrid_results,
                "time": hybrid_time,
                "count": len(hybrid_results),
                "ground_truth": hybrid_gt_metrics
            }
        except Exception as e:
            print(f"❌ Hybrid search failed: {e}")
            results["hybrid_search"] = {"results": [], "time": 0, "count": 0, "ground_truth": {}}
        
        return results

def display_results_multi_native(results: Dict[str, Any], query: str):
    """
    Display search results in a formatted way for multi-vector native search with ground truth comparison.
    
    Args:
        results: Results dictionary from search_all_methods
        query: Original search query
    """
    print(f"\n🔍 Multi-Vector Native Elasticsearch Search Results for: '{query}'")
    print("=" * 70)
    
    # Text Search Results
    if "text_search" in results and results["text_search"]["results"]:
        print(f"\n📊 NATIVE TEXT SEARCH:")
        print("-" * 30)
        
        # Display ground truth metrics
        gt_metrics = results["text_search"].get("ground_truth", {})
        if gt_metrics.get("has_ground_truth", False):
            print(f"🎯 Ground Truth Match: {gt_metrics['found_count']} of {len(results['text_search']['results'])} returned models match ground truth")
        else:
            print("🎯 No ground truth available for this query")
        
        print()
        for i, result in enumerate(results["text_search"]["results"], 1):
            model_data = result["model_data"]
            score = result["score"]
            model_name = model_data.get('name', 'Unknown')
            
            # Mark if this model is in ground truth
            gt_marker = "🎯" if gt_metrics.get("found_models") and model_name in gt_metrics["found_models"] else "  "
            
            print(f"   {i}. {gt_marker} {model_name} ({model_data.get('platform', 'Unknown')})")
            print(f"      Score: {score:.4f}")
            print(f"      ML Tasks: {model_data.get('mlTask', 'N/A')}")
            # Construct HuggingFace URL for hf_models
            if model_name and "/" in model_name:  # HuggingFace models have format "org/model-name"
                source_url = f"https://huggingface.co/{model_name}"
                print(f"      Source URL: {source_url}")
            else:
                print(f"      Model ID: {model_data.get('db_identifier', '')}")
            print()
    
    # Multi-Vector Search Results
    if "multi_vector_search" in results and results["multi_vector_search"]["results"]:
        print(f"\n📊 NATIVE MULTI-VECTOR SEARCH:")
        print("-" * 30)
        
        # Display ground truth metrics
        gt_metrics = results["multi_vector_search"].get("ground_truth", {})
        if gt_metrics.get("has_ground_truth", False):
            print(f"🎯 Ground Truth Match: {gt_metrics['found_count']} of {len(results['multi_vector_search']['results'])} returned models match ground truth")
        else:
            print("🎯 No ground truth available for this query")
        
        print()
        for i, result in enumerate(results["multi_vector_search"]["results"], 1):
            model_data = result["model_data"]
            score = result["score"]
            model_name = model_data.get('name', 'Unknown')
            
            # Mark if this model is in ground truth
            gt_marker = "🎯" if gt_metrics.get("found_models") and model_name in gt_metrics["found_models"] else "  "
            
            print(f"   {i}. {gt_marker} {model_name} ({model_data.get('platform', 'Unknown')})")
            print(f"      Multi-Vector Score: {score:.4f}")
            print(f"      ML Tasks: {model_data.get('mlTask', 'N/A')}")
            # Construct HuggingFace URL for hf_models
            if model_name and "/" in model_name:  # HuggingFace models have format "org/model-name"
                source_url = f"https://huggingface.co/{model_name}"
                print(f"      Source URL: {source_url}")
            else:
                print(f"      Model ID: {model_data.get('db_identifier', '')}")
            print()
    
    # Hybrid Search Results
    if "hybrid_search" in results and results["hybrid_search"]["results"]:
        print(f"\n📊 NATIVE HYBRID SEARCH:")
        print("-" * 30)
        
        # Display ground truth metrics
        gt_metrics = results["hybrid_search"].get("ground_truth", {})
        if gt_metrics.get("has_ground_truth", False):
            print(f"🎯 Ground Truth Match: {gt_metrics['found_count']} of {len(results['hybrid_search']['results'])} returned models match ground truth")
        else:
            print("🎯 No ground truth available for this query")
        
        print()
        for i, result in enumerate(results["hybrid_search"]["results"], 1):
            model_data = result["model_data"]
            score = result["score"]
            model_name = model_data.get('name', 'Unknown')
            
            # Mark if this model is in ground truth
            gt_marker = "🎯" if gt_metrics.get("found_models") and model_name in gt_metrics["found_models"] else "  "
            
            print(f"   {i}. {gt_marker} {model_name} ({model_data.get('platform', 'Unknown')})")
            print(f"      Combined Score: {score:.4f}")
            print(f"      ML Tasks: {model_data.get('mlTask', 'N/A')}")
            # Construct HuggingFace URL for hf_models
            if model_name and "/" in model_name:  # HuggingFace models have format "org/model-name"
                source_url = f"https://huggingface.co/{model_name}"
                print(f"      Source URL: {source_url}")
            else:
                print(f"      Model ID: {model_data.get('db_identifier', '')}")
            print()
    
    # Performance Information
    if results:
        print(f"\n⏱️  Performance:")
        if "text_search" in results:
            print(f"   Native Text Search: {results['text_search']['time']:.3f}s")
        if "multi_vector_search" in results:
            print(f"   Native Multi-Vector Search: {results['multi_vector_search']['time']:.3f}s")
        if "hybrid_search" in results:
            print(f"   Native Hybrid Search: {results['hybrid_search']['time']:.3f}s")
        
        # Summary of ground truth performance
        print(f"\n🎯 Ground Truth Summary:")
        for search_type in ["text_search", "multi_vector_search", "hybrid_search"]:
            if search_type in results:
                gt_metrics = results[search_type].get("ground_truth", {})
                if gt_metrics.get("has_ground_truth", False):
                    returned_count = len(results[search_type]['results'])
                    found_count = gt_metrics['found_count']
                    print(f"   {search_type.replace('_', ' ').title()}: {found_count} of {returned_count} returned models match ground truth")
                else:
                    print(f"   {search_type.replace('_', ' ').title()}: No ground truth available")

def interactive_mode():
    """
    Run the CLI in interactive mode with native multi-vector Elasticsearch search.
    """
    # Start logging for interactive mode
    start_console_logging("logs", "search_multi_cli_native_interactive")
    
    print("🚀 Native Multi-Vector Elasticsearch Search CLI - Interactive Mode")
    print("=" * 70)
    print("Available commands:")
    print("  - Type your search query to search")
    print("  - Use 'index:query' format (e.g., 'hf_models:BERT model')")
    print("  - Use 'type:text/vector/hybrid:query' format for specific search type")
    print("  - Type 'help' for more information")
    print("  - Type 'quit' to exit")
    print()
    
    # Initialize the multi-vector search handler and ground truth parser
    try:
        handler = MultiVectorSearchHandler()
        ground_truth_parser = GroundTruthParser()
    except Exception as e:
        print(f"❌ Failed to initialize multi-vector search handler: {e}")
        return
    
    while True:
        try:
            user_input = input("🔍 Multi-Vector Search> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                stop_console_logging()
                break
            
            if user_input.lower() == 'help':
                print("\n📚 Help:")
                print("  Basic search: 'BERT model'")
                print("  Index-specific: 'hf_models:BERT model'")
                print("  Type-specific: 'type:hybrid:BERT model'")
                print("  Combined: 'hf_models:type:vector:BERT model'")
                print("  Available search types: text, vector, hybrid")
                print("  Available indices: hf_models, openml_models, ai4life_models")
                print()
                continue
            
            if not user_input:
                continue
            
            # Parse the input
            index_name = "hf_models"  # Default
            search_type = "all"  # Default to all methods
            query = user_input
            
            # Check for index specification
            if ':' in user_input:
                parts = user_input.split(':', 1)
                if parts[0] in ['hf_models', 'openml_models', 'ai4life_models']:
                    index_name = parts[0]
                    query = parts[1]
            
            # Check for search type specification
            if 'type:' in query:
                type_part, query = query.split(':', 1)
                search_type_str = type_part.replace('type', '').strip()
                if search_type_str in ['text', 'vector', 'hybrid']:
                    search_type = search_type_str
            
            # Perform search
            if search_type == "all":
                results = handler.search_all_methods(query, index_name, top_k=5, ground_truth_parser=ground_truth_parser)
                display_results_multi_native(results, query)
            elif search_type == "text":
                results = handler.text_search(query, index_name, top_k=5)
                # Calculate ground truth metrics for single search type
                gt_metrics = ground_truth_parser.calculate_recall(query, results)
                display_results_multi_native({"text_search": {"results": results, "ground_truth": gt_metrics}}, query)
            elif search_type == "vector":
                results = handler.multi_vector_search(query, index_name, top_k=5)
                # Calculate ground truth metrics for single search type
                gt_metrics = ground_truth_parser.calculate_recall(query, results)
                display_results_multi_native({"multi_vector_search": {"results": results, "ground_truth": gt_metrics}}, query)
            elif search_type == "hybrid":
                results = handler.hybrid_search(query, index_name, top_k=5)
                # Calculate ground truth metrics for single search type
                gt_metrics = ground_truth_parser.calculate_recall(query, results)
                display_results_multi_native({"hybrid_search": {"results": results, "ground_truth": gt_metrics}}, query)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            stop_console_logging()
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Native Multi-Vector Elasticsearch Search CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 search_multi_cli_native.py "BERT model"
  python3 search_multi_cli_native.py "image classification" --search-type hybrid
  python3 search_multi_cli_native.py "huggingface" --search-type vector --top-k 5
  python3 search_multi_cli_native.py --interactive
        """
    )
    
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--index", "-i", 
        choices=["hf_models", "openml_models", "ai4life_models"],
        default="hf_models",
        help="Index to search in (default: hf_models)"
    )
    parser.add_argument(
        "--search-type", "-st",
        choices=["text", "vector", "hybrid", "all"],
        default="all",
        help="Search type (default: all)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )
    parser.add_argument(
        "--interactive", "-int",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # Single query mode
    if not args.query:
        parser.print_help()
        return
    
    # Start logging for single query mode
    start_console_logging("logs", "search_multi_cli_native_single")
    
    try:
        # Initialize the multi-vector search handler and ground truth parser
        handler = MultiVectorSearchHandler()
        ground_truth_parser = GroundTruthParser()
        
        # Perform search based on type
        if args.search_type == "all":
            results = handler.search_all_methods(args.query, args.index, args.top_k, ground_truth_parser=ground_truth_parser)
            display_results_multi_native(results, args.query)
        elif args.search_type == "text":
            results = handler.text_search(args.query, args.index, args.top_k)
            # Calculate ground truth metrics for single search type
            gt_metrics = ground_truth_parser.calculate_recall(args.query, results)
            display_results_multi_native({"text_search": {"results": results, "ground_truth": gt_metrics}}, args.query)
        elif args.search_type == "vector":
            results = handler.multi_vector_search(args.query, args.index, args.top_k)
            # Calculate ground truth metrics for single search type
            gt_metrics = ground_truth_parser.calculate_recall(args.query, results)
            display_results_multi_native({"multi_vector_search": {"results": results, "ground_truth": gt_metrics}}, args.query)
        elif args.search_type == "hybrid":
            results = handler.hybrid_search(args.query, args.index, args.top_k)
            # Calculate ground truth metrics for single search type
            gt_metrics = ground_truth_parser.calculate_recall(args.query, results)
            display_results_multi_native({"hybrid_search": {"results": results, "ground_truth": gt_metrics}}, args.query)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        # Always stop logging
        stop_console_logging()

if __name__ == "__main__":
    main()
