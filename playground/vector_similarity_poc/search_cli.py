import sys
import argparse
from typing import List, Dict, Any
from src.embedding_service import EmbeddingService
from src.vector_index_handler import VectorIndexHandler
from src.hybrid_search import HybridSearch
from src.config import config

def search_all_methods(query: str, index_name: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search using all three methods and return results.
    """
    try:
        # Initialize services
        embedding_service = EmbeddingService()
        vector_handler = VectorIndexHandler(embedding_service)
        hybrid_search = HybridSearch(embedding_service, vector_handler)
        
        # Perform all searches
        results = hybrid_search.compare_search_methods(query, index_name, top_k)
        
        return results
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return {}

def format_results(results: Dict[str, List[Dict[str, Any]]], query: str) -> None:
    """
    Format and display search results.
    """
    print(f"\n🔍 Search Results for: '{query}'")
    print("=" * 60)
    
    # Display results for each method
    for method in ["text", "vector", "hybrid"]:
        method_results = results.get(method, [])
        
        print(f"\n📊 {method.upper()} SEARCH:")
        print("-" * 30)
        
        if method_results:
            for i, result in enumerate(method_results, 1):
                name = result["model_data"]["name"]
                platform = result["model_data"].get("platform", "Unknown")
                
                if method == "hybrid":
                    score = result["combined_score"]
                    text_score = result["text_score"]
                    vector_score = result["vector_score"]
                    print(f"   {i}. {name} ({platform})")
                    print(f"      Combined Score: {score:.4f}")
                    print(f"      Text Score: {text_score:.4f}, Vector Score: {vector_score:.4f}")
                else:
                    score = result["score"]
                    print(f"   {i}. {name} ({platform})")
                    print(f"      Score: {score:.4f}")
        else:
            print("   No results found")
    
    # Show timing information
    timing = results.get("timing", {})
    if timing:
        print(f"\n⏱️  Performance:")
        print(f"   Text Search: {timing.get('text_search', 0):.3f}s")
        print(f"   Vector Search: {timing.get('vector_search', 0):.3f}s")
        print(f"   Hybrid Search: {timing.get('hybrid_search', 0):.3f}s")

def interactive_mode():
    """
    Run in interactive mode for multiple queries.
    """
    print("🚀 Vector Similarity Search CLI - Interactive Mode")
    print("=" * 50)
    print("Type your search queries and press Enter.")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print("Type 'help' for available commands.")
    print()
    
    # Initialize services once
    try:
        embedding_service = EmbeddingService()
        vector_handler = VectorIndexHandler(embedding_service)
        hybrid_search = HybridSearch(embedding_service, vector_handler)
        print("✅ Services initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        return
    
    # Get available indices
    available_indices = config.SOURCE_INDICES
    print(f"📁 Available indices: {', '.join(available_indices)}")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("🔍 Enter search query: ").strip()
            
            if not user_input:
                continue
                
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("\n📖 Available Commands:")
                print("   • Enter any search query to search")
                print("   • 'quit', 'exit', or 'q' to stop")
                print("   • 'help' to show this message")
                print(f"   • Available indices: {', '.join(available_indices)}")
                print()
                continue
            
            # Parse query for index specification
            if ':' in user_input:
                parts = user_input.split(':', 1)
                index_name = parts[0].strip()
                query = parts[1].strip()
                
                if index_name not in available_indices:
                    print(f"❌ Invalid index: {index_name}")
                    print(f"   Available indices: {', '.join(available_indices)}")
                    continue
            else:
                # Default to first available index
                index_name = available_indices[0]
                query = user_input
            
            # Perform search
            results = hybrid_search.compare_search_methods(query, index_name, top_k=5)
            
            if results:
                format_results(results, query)
            else:
                print("❌ No results found or search failed")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()  # Add spacing between queries

def main():
    """
    Main function for the CLI.
    """
    
    parser = argparse.ArgumentParser(
        description="Vector Similarity Search CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 search_cli.py "language understanding model"
  python3 search_cli.py "BERT model" --index hf_models
  python3 search_cli.py --interactive
  python3 search_cli.py "image classification" --top-k 10
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query (required unless using --interactive)"
    )
    
    parser.add_argument(
        "--index", "-i",
        default=config.SOURCE_INDICES[0],
        choices=config.SOURCE_INDICES,
        help=f"Index to search in (default: {config.SOURCE_INDICES[0]})"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    
    parser.add_argument(
        "--interactive", "-int",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Handle interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # Handle single query mode
    if not args.query:
        print("❌ Error: Query is required unless using --interactive")
        print("Use --help for usage information")
        sys.exit(1)
    
    # Perform search
    print(f"🔍 Searching for: '{args.query}'")
    print(f"📁 Index: {args.index}")
    print(f"📊 Top-K: {args.top_k}")
    
    results = search_all_methods(args.query, args.index, args.top_k)
    
    if results:
        format_results(results, args.query)
    else:
        print("❌ No results found or search failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
