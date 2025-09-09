import time
from typing import List, Dict, Any, Optional, Tuple
from elasticsearch import Elasticsearch
from .embedding_service import EmbeddingService
from .vector_index_handler import VectorIndexHandler
from .config import config

class HybridSearch:
    """
    Combines text search and vector search for optimal results.
    """
    
    def __init__(self, embedding_service: EmbeddingService, vector_handler: VectorIndexHandler):
        self.embedding_service = embedding_service
        self.vector_handler = vector_handler
        self.es = Elasticsearch([config.get_elasticsearch_url()])
        
    def text_search(self, query: str, index_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform traditional text search using Elasticsearch.
        """
        try:
            search_body = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            f"name^{config.TEXT_SEARCH_BOOSTS['name']}", 
                            f"description^{config.TEXT_SEARCH_BOOSTS['description']}", 
                            f"mlTask^{config.TEXT_SEARCH_BOOSTS['mlTask']}", 
                            f"keywords^{config.TEXT_SEARCH_BOOSTS['keywords']}"
                        ],
                        "fuzziness": "AUTO",
                        "type": "best_fields"
                    }
                },
                "_source": ["name", "description", "mlTask", "keywords", "platform", "db_identifier", "sharedBy", "license"]
            }
            
            response = self.es.search(index=index_name, body=search_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append({
                    "score": hit["_score"],
                    "search_type": "text",
                    "model_data": {
                        "db_identifier": source.get("db_identifier"),
                        "name": source.get("name"),
                        "description": source.get("description"),
                        "mlTask": source.get("mlTask"),
                        "keywords": source.get("keywords"),
                        "platform": source.get("platform"),
                        "sharedBy": source.get("sharedBy"),
                        "license": source.get("license"),
                    }
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Text search failed: {e}")
            return []
    
    def vector_search(self, query: str, index_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        """
        try:
            results = self.vector_handler.vector_search(query, index_name, top_k)
            
            # Add search type to results
            for result in results:
                result["search_type"] = "vector"
            
            return results
            
        except Exception as e:
            print(f"❌ Vector search failed: {e}")
            return []
    
    def hybrid_search(self, 
                     query: str, 
                     index_name: str, 
                     top_k: int = 10,
                     text_weight: float = 0.3,
                     vector_weight: float = 0.7,
                     normalize_scores: bool = True) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining text and vector search.
        
        Args:
            query: Search query
            index_name: Index to search in
            top_k: Number of results to return
            text_weight: Weight for text search results (0.0 to 1.0)
            vector_weight: Weight for vector search results (0.0 to 1.0)
            normalize_scores: Whether to normalize scores between 0 and 1
        """
        
        start_time = time.time()
        
        # Perform both searches
        text_results = self.text_search(query, index_name, top_k * 2)  # Get more for better combination
        vector_results = self.vector_search(query, index_name, top_k * 2)
        
        # Combine results
        combined_results = self._combine_search_results(
            text_results, vector_results, 
            text_weight, vector_weight, normalize_scores
        )
        
        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        final_results = combined_results[:top_k]
        
        end_time = time.time()
        
        return final_results
    
    def _combine_search_results(self, 
                               text_results: List[Dict[str, Any]], 
                               vector_results: List[Dict[str, Any]],
                               text_weight: float, 
                               vector_weight: float,
                               normalize_scores: bool = True) -> List[Dict[str, Any]]:
        """
        Combine text and vector search results with weighted scoring.
        """
        # Create a dictionary to store combined results
        combined_dict = {}
        
        # Process text search results
        for result in text_results:
            model_id = result["model_data"]["db_identifier"]
            if model_id not in combined_dict:
                combined_dict[model_id] = {
                    "model_data": result["model_data"],
                    "text_score": 0.0,
                    "vector_score": 0.0,
                    "text_result": None,
                    "vector_result": None
                }
            
            combined_dict[model_id]["text_score"] = result["score"]
            combined_dict[model_id]["text_result"] = result
        
        # Process vector search results
        for result in vector_results:
            model_id = result["model_data"]["db_identifier"]
            if model_id not in combined_dict:
                combined_dict[model_id] = {
                    "model_data": result["model_data"],
                    "text_score": 0.0,
                    "vector_score": 0.0,
                    "text_result": None,
                    "vector_result": None
                }
            
            combined_dict[model_id]["vector_score"] = result["score"]
            combined_dict[model_id]["vector_result"] = result
        
        # Normalize scores if requested
        if normalize_scores:
            text_scores = [r["text_score"] for r in combined_dict.values() if r["text_score"] > 0]
            vector_scores = [r["vector_score"] for r in combined_dict.values() if r["vector_score"] > 0]
            
            if text_scores:
                max_text_score = max(text_scores)
                min_text_score = min(text_scores)
                text_range = max_text_score - min_text_score if max_text_score != min_text_score else 1
                
                for result in combined_dict.values():
                    if result["text_score"] > 0:
                        result["text_score"] = (result["text_score"] - min_text_score) / text_range
            
            if vector_scores:
                max_vector_score = max(vector_scores)
                min_vector_score = min(vector_scores)
                vector_range = max_vector_score - min_vector_score if max_vector_score != min_vector_score else 1
                
                for result in combined_dict.values():
                    if result["vector_score"] > 0:
                        result["vector_score"] = (result["vector_score"] - min_vector_score) / vector_range
        
        # Calculate combined scores and create final results
        combined_results = []
        for model_id, data in combined_dict.items():
            combined_score = (text_weight * data["text_score"] + 
                            vector_weight * data["vector_score"])
            
            combined_results.append({
                "model_data": data["model_data"],
                "combined_score": combined_score,
                "text_score": data["text_score"],
                "vector_score": data["vector_score"],
                "search_type": "hybrid",
                "text_result": data["text_result"],
                "vector_result": data["vector_result"]
            })
        
        return combined_results
    
    def compare_search_methods(self, query: str, index_name: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compare all three search methods side by side.
        """
        
        results = {}
        
        # Text search
        start_time = time.time()
        results["text"] = self.text_search(query, index_name, top_k)
        text_time = time.time() - start_time
        
        # Vector search
        start_time = time.time()
        results["vector"] = self.vector_search(query, index_name, top_k)
        vector_time = time.time() - start_time
        
        # Hybrid search
        start_time = time.time()
        results["hybrid"] = self.hybrid_search(query, index_name, top_k)
        hybrid_time = time.time() - start_time
        
        # Add timing information
        results["timing"] = {
            "text_search": text_time,
            "vector_search": vector_time,
            "hybrid_search": hybrid_time
        }
        
        return results
    
    def get_search_statistics(self, query: str, index_name: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Get detailed statistics about search performance.
        """
        results = self.compare_search_methods(query, index_name, top_k)
        
        stats = {
            "query": query,
            "index": index_name,
            "top_k": top_k,
            "timing": results["timing"],
            "result_counts": {
                "text": len(results["text"]),
                "vector": len(results["vector"]),
                "hybrid": len(results["hybrid"])
            },
            "score_ranges": {
                "text": self._get_score_range(results["text"]),
                "vector": self._get_score_range(results["vector"]),
                "hybrid": self._get_score_range(results["hybrid"])
            }
        }
        
        return stats
    
    def _get_score_range(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Get score range for a set of results.
        """
        if not results:
            return {"min": 0.0, "max": 0.0, "avg": 0.0}
        
        scores = [r["score"] for r in results]
        return {
            "min": min(scores),
            "max": max(scores),
            "avg": sum(scores) / len(scores)
        }

def main():
    """
    Test the hybrid search functionality.
    """
    print("🧪 Testing Hybrid Search...")
    
    try:
        # Initialize services
        embedding_service = EmbeddingService()
        vector_handler = VectorIndexHandler(embedding_service)
        hybrid_search = HybridSearch(embedding_service, vector_handler)
        
        # Test queries
        test_queries = [
            "language understanding model",
            "image classification model",
            "transformer model",
            "BERT model"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            print("=" * 60)
            
            # Compare all methods
            results = hybrid_search.compare_search_methods(query, "hf_models", top_k=3)
            
            # Display results
            for method, method_results in results.items():
                if method == "timing":
                    continue
                    
                print(f"\n📊 {method.upper()} SEARCH RESULTS:")
                print("-" * 40)
                
                if method_results:
                    for i, result in enumerate(method_results):
                        name = result["model_data"]["name"]
                        score = result["score"] if method != "hybrid" else result["combined_score"]
                        platform = result["model_data"]["platform"]
                        print(f"   {i+1}. {name} ({platform}) - Score: {score:.4f}")
                else:
                    print("   No results found")
            
            # Show timing
            print(f"\n⏱️  TIMING:")
            print(f"   Text search: {results['timing']['text_search']:.3f}s")
            print(f"   Vector search: {results['timing']['vector_search']:.3f}s")
            print(f"   Hybrid search: {results['timing']['hybrid_search']:.3f}s")
        
        # Test hybrid search with different weights
        print(f"\n🎯 Testing different weight combinations:")
        print("=" * 60)
        
        query = "language understanding model"
        weight_combinations = [
            (0.1, 0.9),  # Mostly vector
            (0.5, 0.5),  # Balanced
            (0.9, 0.1),  # Mostly text
        ]
        
        for text_weight, vector_weight in weight_combinations:
            print(f"\n📊 Weights - Text: {text_weight}, Vector: {vector_weight}")
            results = hybrid_search.hybrid_search(query, "hf_models", top_k=3, 
                                                text_weight=text_weight, vector_weight=vector_weight)
            
            for i, result in enumerate(results):
                name = result["model_data"]["name"]
                combined_score = result["combined_score"]
                text_score = result["text_score"]
                vector_score = result["vector_score"]
                print(f"   {i+1}. {name} - Combined: {combined_score:.4f} (Text: {text_score:.4f}, Vector: {vector_score:.4f})")
                
    except Exception as e:
        print(f"❌ Hybrid search test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
