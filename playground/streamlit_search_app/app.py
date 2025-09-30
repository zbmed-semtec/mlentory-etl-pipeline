#!/usr/bin/env python3
"""
Streamlit Search Application for Model Discovery

This application provides a web interface for searching ML models using:
- Multiple embedding models (sentence-transformers/all-mpnet-base-v2, intfloat/e5-base-v2, BAAI/bge-base-en-v1.5)
- Configurable field selection and weights
- Three search types: Vector, Text, and Hybrid search
- Interactive results display with tabs

Usage:
    streamlit run app.py
"""

import streamlit as st
import sys
import os
import time
from typing import List, Dict, Any, Optional
import pandas as pd

from config import config
from embedding_service import EmbeddingService
from elasticsearch import Elasticsearch

# Page configuration
st.set_page_config(
    page_title="ML Model Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitSearchHandler:
    """
    Handler for Streamlit-based model search functionality.
    """
    
    def __init__(self):
        """Initialize the search handler."""
        self.es = None
        self.embedding_services = {}
        self.initialize_services()
    
    def initialize_services(self):
        """Initialize Elasticsearch and embedding services."""
        try:
            # Connect to Elasticsearch
            self.es = Elasticsearch(
                [config.get_elasticsearch_url()],
                basic_auth=(config.ELASTICSEARCH_USERNAME, config.ELASTICSEARCH_PASSWORD),
                request_timeout=30,
                max_retries=3,
                retry_on_timeout=True,
            )
            
            if not self.es.ping():
                st.error("❌ Cannot connect to Elasticsearch")
                return False
                
            # Initialize embedding services for all models
            model_configs = config.get_all_multi_models()
            for model_key, model_config in model_configs.items():
                try:
                    self.embedding_services[model_key] = EmbeddingService(
                        model_name=model_config['model_name']
                    )
                except Exception as e:
                    st.warning(f"⚠️ Failed to load {model_key}: {e}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to initialize services: {e}")
            return False
    
    def find_vector_index(self, model_key: str, source_index: str = "hf_models") -> Optional[str]:
        """Find the vector index for a specific model using the same logic as create_multi_model_vector_indices.py."""
        try:
            # Get the model configuration
            model_config = config.get_multi_model_config(model_key)
            if not model_config:
                st.error(f"❌ Unknown model key: {model_key}")
                return None
            
            # Create the vector index name using the same format as create_multi_model_vector_indices.py
            # Format: vector_{model_suffix}_{source_index}
            index_suffix = model_config.get('index_suffix', model_key)
            index_name = f"vector_{index_suffix}_{source_index}"
            
            # Check if the specific model's vector index exists
            if self.es.indices.exists(index=index_name):
                return index_name
            
            # Fallback: check if source index has dense_vector fields
            if self._check_for_dense_vector_fields(source_index):
                return source_index
            
            st.warning(f"⚠️ Vector index {index_name} not found for {model_key}")
            return None
            
        except Exception as e:
            st.error(f"❌ Error finding vector index: {e}")
            return None
    
    def _check_for_dense_vector_fields(self, index_name: str) -> bool:
        """Check if an index has dense_vector fields."""
        try:
            mapping = self.es.indices.get_mapping(index=index_name)
            properties = mapping[index_name]['mappings']['properties']
            
            for field_name, field_config in properties.items():
                if field_config.get('type') == 'dense_vector':
                    return True
            return False
        except:
            return False
    
    def text_search(self, query: str, source_index: str, selected_fields: List[str], 
                   field_weights: Dict[str, float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform text search with custom field weights."""
        try:
            # Build weighted fields for search
            weighted_fields = []
            for field in selected_fields:
                weight = field_weights.get(field, 1.0)
                weighted_fields.append(f"{field}^{weight}")
            
            search_body = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": weighted_fields,
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
            st.error(f"❌ Text search failed: {e}")
            return []
    
    def vector_search(self, query: str, model_key: str, source_index: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform vector search using specified embedding model."""
        if model_key not in self.embedding_services:
            st.error(f"❌ Embedding model {model_key} not available")
            return []
        
        try:
            # Find the vector index
            vector_index = self.find_vector_index(model_key, source_index)
            if not vector_index:
                st.error(f"❌ No vector index found for {model_key}")
                return []
            
            # Generate query vector
            embedding_service = self.embedding_services[model_key]
            query_vector = embedding_service.encode_text(query)
            
            # Build KNN search query
            search_body = {
                "size": top_k,
                "knn": {
                    "field": "model_vector",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": config.VECTOR_SEARCH_CANDIDATES
                },
                "_source": [
                    "name", "description", "mlTask", "keywords", "platform", 
                    "db_identifier", "sharedBy", "license"
                ]
            }
            
            response = self.es.search(index=vector_index, body=search_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "score": hit["_score"],
                    "model_data": hit["_source"],
                    "db_identifier": hit["_source"].get("db_identifier", ""),
                    "search_type": "vector"
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            st.error(f"❌ Vector search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, model_key: str, source_index: str, 
                     selected_fields: List[str], field_weights: Dict[str, float],
                     text_weight: float, vector_weight: float, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search combining text and vector search."""
        if model_key not in self.embedding_services:
            st.error(f"❌ Embedding model {model_key} not available")
            return self.text_search(query, source_index, selected_fields, field_weights, top_k)
        
        try:
            # Find the vector index
            vector_index = self.find_vector_index(model_key, source_index)
            if not vector_index:
                st.warning(f"⚠️ No vector index found for {model_key}, falling back to text search")
                return self.text_search(query, source_index, selected_fields, field_weights, top_k)
            
            # Generate query vector
            embedding_service = self.embedding_services[model_key]
            query_vector = embedding_service.encode_text(query)
            
            # Build weighted fields for text search
            weighted_fields = []
            for field in selected_fields:
                weight = field_weights.get(field, 1.0)
                weighted_fields.append(f"{field}^{weight}")
            
            # Build hybrid search query
            search_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [
                            # Text search with field weights
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": weighted_fields,
                                    "type": "best_fields",
                                    "fuzziness": "AUTO",
                                    "boost": text_weight
                                }
                            },
                            # Vector search with weight
                            {
                                "knn": {
                                    "field": "model_vector",
                                    "query_vector": query_vector,
                                    "k": top_k,
                                    "num_candidates": config.VECTOR_SEARCH_CANDIDATES,
                                    "boost": vector_weight
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
            
            response = self.es.search(index=vector_index, body=search_body)
            
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
            st.error(f"❌ Hybrid search failed: {e}")
            return self.text_search(query, source_index, selected_fields, field_weights, top_k)

    def hybrid_search_rrf(self, query: str, model_key: str, source_index: str, 
                         selected_fields: List[str], field_weights: Dict[str, float],
                         text_weight: float, vector_weight: float, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using Reciprocal Rank Fusion (RRF).
        RRF combines rankings without requiring score normalization.
        """
        if model_key not in self.embedding_services:
            st.warning(f"⚠️ Embedding model {model_key} not available, falling back to text search")
            return self.text_search(query, source_index, selected_fields, field_weights, top_k)
        try:
            # Retrieve more candidates to improve fusion overlap
            candidates_multiplier = 3
            num_candidates = top_k * candidates_multiplier

            # Independent runs
            text_results = self.text_search(
                query, source_index, selected_fields, field_weights, num_candidates
            )
            vector_results = self.vector_search(
                query, model_key, source_index, num_candidates
            )

            # Standard RRF parameter
            k = 60

            rrf_scores: Dict[str, Dict[str, Any]] = {}

            # Process text ranking
            for rank, result in enumerate(text_results, start=1):
                model_id = result.get('db_identifier') or result['model_data'].get('name') or str(rank)
                contribution = float(text_weight) / float(k + rank)
                if model_id not in rrf_scores:
                    rrf_scores[model_id] = {
                        'model_data': result['model_data'],
                        'db_identifier': model_id,
                        'rrf_score': 0.0,
                        'text_rank': None,
                        'vector_rank': None,
                        'text_score': None,
                        'vector_score': None,
                        'search_type': 'hybrid'
                    }
                rrf_scores[model_id]['rrf_score'] += contribution
                rrf_scores[model_id]['text_rank'] = rank
                rrf_scores[model_id]['text_score'] = result.get('score')

            # Process vector ranking
            for rank, result in enumerate(vector_results, start=1):
                model_id = result.get('db_identifier') or result['model_data'].get('name') or str(rank)
                contribution = float(vector_weight) / float(k + rank)
                if model_id not in rrf_scores:
                    rrf_scores[model_id] = {
                        'model_data': result['model_data'],
                        'db_identifier': model_id,
                        'rrf_score': 0.0,
                        'text_rank': None,
                        'vector_rank': None,
                        'text_score': None,
                        'vector_score': None,
                        'search_type': 'hybrid'
                    }
                rrf_scores[model_id]['rrf_score'] += contribution
                rrf_scores[model_id]['vector_rank'] = rank
                rrf_scores[model_id]['vector_score'] = result.get('score')

            # Sort by fused score, break ties by best rank
            def sort_key(item: Dict[str, Any]):
                best_rank = min([r for r in [item.get('text_rank'), item.get('vector_rank')] if r is not None] or [999999])
                return (item['rrf_score'], -best_rank)

            fused = sorted(rrf_scores.values(), key=sort_key, reverse=True)

            final_results: List[Dict[str, Any]] = []
            for entry in fused[:top_k]:
                final_results.append({
                    'score': entry['rrf_score'],
                    'model_data': entry['model_data'],
                    'db_identifier': entry['db_identifier'],
                    'search_type': 'hybrid',
                    'ranking_details': {
                        'text_rank': entry.get('text_rank'),
                        'vector_rank': entry.get('vector_rank'),
                        'text_score': entry.get('text_score'),
                        'vector_score': entry.get('vector_score')
                    }
                })
            return final_results
        except Exception as e:
            st.error(f"❌ Hybrid RRF search failed: {e}")
            return self.text_search(query, source_index, selected_fields, field_weights, top_k)

def display_search_results(results: List[Dict[str, Any]], search_type: str, query: str):
    """Display search results in a formatted way."""
    if not results:
        st.warning(f"No results found for {search_type} search")
        return
    
    st.subheader(f"🔍 {search_type.title()} Search Results")
    st.caption(f"Query: '{query}' | Found {len(results)} results")
    
    # Create a DataFrame for better display
    data = []
    for i, result in enumerate(results, 1):
        model_data = result["model_data"]
        if search_type.lower() in ("text", "vector"):
            # Minimal columns for Text and Vector tables
            data.append({
                "Rank": i,
                "Model Name": model_data.get('name', 'Unknown'),
                "Score": f"{result['score']:.4f}",
                "ML Tasks": ", ".join(model_data.get('mlTask', [])) if isinstance(model_data.get('mlTask'), list) else model_data.get('mlTask', 'N/A'),
            })
        else:
            # Full columns for Hybrid
            data.append({
                "Rank": i,
                "Model Name": model_data.get('name', 'Unknown'),
                "Score": f"{result['score']:.4f}",
                "ML Tasks": ", ".join(model_data.get('mlTask', [])) if isinstance(model_data.get('mlTask'), list) else model_data.get('mlTask', 'N/A'),
                "Keywords": ", ".join(model_data.get('keywords', [])) if isinstance(model_data.get('keywords'), list) else model_data.get('keywords', 'N/A'),
                "Shared By": model_data.get('sharedBy', 'N/A'),
                "Description": model_data.get('description', 'N/A')[:100] + "..." if len(model_data.get('description', '')) > 100 else model_data.get('description', 'N/A')
            })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    
    # Show detailed results in expandable sections
    for i, result in enumerate(results, 1):
        model_data = result["model_data"]
        model_name = model_data.get('name', 'Unknown')
        
        with st.expander(f"{i}. {model_name} (Score: {result['score']:.4f})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Information:**")
                st.write(f"- **Name:** {model_name}")
                st.write(f"- **ML Tasks:** {', '.join(model_data.get('mlTask', [])) if isinstance(model_data.get('mlTask'), list) else model_data.get('mlTask', 'N/A')}")
                st.write(f"- **Keywords:** {', '.join(model_data.get('keywords', [])) if isinstance(model_data.get('keywords'), list) else model_data.get('keywords', 'N/A')}")
                st.write(f"- **Shared By:** {model_data.get('sharedBy', 'N/A')}")
            
            with col2:
                st.write("**Description:**")
                st.write(model_data.get('description', 'No description available'))
                
                # Add HuggingFace URL if applicable
                if model_name and "/" in model_name:
                    hf_url = f"https://huggingface.co/{model_name}"
                    st.write(f"**Source:** [View on HuggingFace]({hf_url})")

def display_search_results_with_details(results: List[Dict[str, Any]], search_type: str, query: str):
    """Display results with additional ranking information (for RRF)."""
    if not results:
        st.warning(f"No results found for {search_type} search")
        return
    st.subheader(f"🔍 {search_type.title()} Search Results")
    st.caption(f"Query: '{query}' | Found {len(results)} results")
    data = []
    for i, result in enumerate(results, 1):
        model_data = result["model_data"]
        row = {
            "Rank": i,
            "Model Name": model_data.get('name', 'Unknown'),
            "RRF Score": f"{result['score']:.6f}",
            "ML Tasks": ", ".join(model_data.get('mlTask', [])) if isinstance(model_data.get('mlTask'), list) else model_data.get('mlTask', 'N/A'),
        }
        if 'ranking_details' in result:
            details = result['ranking_details']
            row["Text Rank"] = details['text_rank'] if details['text_rank'] else '-'
            row["Vector Rank"] = details['vector_rank'] if details['vector_rank'] else '-'
        data.append(row)
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    for i, result in enumerate(results, 1):
        model_data = result["model_data"]
        model_name = model_data.get('name', 'Unknown')
        with st.expander(f"{i}. {model_name} (RRF Score: {result['score']:.6f})"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Model Information:**")
                st.write(f"- **Name:** {model_name}")
                st.write(f"- **ML Tasks:** {', '.join(model_data.get('mlTask', [])) if isinstance(model_data.get('mlTask'), list) else model_data.get('mlTask', 'N/A')}")
                st.write(f"- **Keywords:** {', '.join(model_data.get('keywords', [])) if isinstance(model_data.get('keywords'), list) else model_data.get('keywords', 'N/A')}")
                st.write(f"- **Shared By:** {model_data.get('sharedBy', 'N/A')}")
                if 'ranking_details' in result:
                    st.write("**Ranking Details:**")
                    details = result['ranking_details']
                    st.write(f"- Text Rank: {details['text_rank'] if details['text_rank'] else 'Not in text results'}")
                    st.write(f"- Vector Rank: {details['vector_rank'] if details['vector_rank'] else 'Not in vector results'}")
                    if details['text_score'] is not None:
                        st.write(f"- Text Score: {details['text_score']:.4f}")
                    if details['vector_score'] is not None:
                        st.write(f"- Vector Score: {details['vector_score']:.4f}")
            with col2:
                st.write("**Description:**")
                st.write(model_data.get('description', 'No description available'))
                if model_name and "/" in model_name:
                    hf_url = f"https://huggingface.co/{model_name}"
                    st.write(f"**Source:** [View on HuggingFace]({hf_url})")

def main():
    """Main Streamlit application."""    
    # Initialize or refresh search handler (handle hot-reload stale objects)
    need_init = (
        'search_handler' not in st.session_state or
        not hasattr(st.session_state.get('search_handler'), 'hybrid_search_rrf')
    )
    if need_init:
        with st.spinner("Initializing search services..."):
            st.session_state.search_handler = StreamlitSearchHandler()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Search Configuration")
        
        # Embedding model selection
        st.subheader("🤖 Embedding Model")
        model_configs = config.get_all_multi_models()
        model_options = {key: f"{config['model_name']} ({config['description']})" 
                        for key, config in model_configs.items()}
        
        selected_model = st.selectbox(
            "Choose embedding model:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )
        
        # Field selection
        st.subheader("📝 Search Fields")
        available_fields = config.EMBEDDING_FIELDS
        selected_fields = []
        
        # Create checkboxes for each field
        for field in available_fields:
            if st.checkbox(
                f"Search in '{field}'",
                value=True,  # All fields selected by default
                key=f"field_{field}"
            ):
                selected_fields.append(field)
        
        # Field weights
        st.subheader("⚖️ Field Weights")
        field_weights = {}
        for field in selected_fields:
            weight = st.slider(
                f"Weight for '{field}':",
                min_value=0.1,
                max_value=5.0,
                value=config.get_field_weight(field),
                step=0.1
            )
            field_weights[field] = weight
        
        # Hybrid method and weights
        st.subheader("🔄 Hybrid Search")
        hybrid_method = st.radio(
            "Hybrid method:",
            options=["Weighted (ES)", "RRF"],
            index=0,
            horizontal=True,
            key="hybrid_method_choice"
        )
        st.caption("Weighted: ES sums boosted scores. RRF: rank fusion (scale-invariant).")

        st.markdown("**Weights**")
        
        # Text weight slider
        text_weight = st.slider(
            "Text Search Weight:",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1,
            key="text_weight_slider"
        )
        
        # Calculate vector weight based on text weight
        vector_weight = 1.0 - text_weight
        
        # Display weights in a nice format
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Text Weight", f"{text_weight:.1f}")
        with col2:
            st.metric("Vector Weight", f"{vector_weight:.1f}")
        
        # Search parameters
        st.subheader("🔧 Search Parameters")
        top_k = st.slider(
            "Number of results:",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )
        
        # Determine source index based on selected embedding model
        # This follows the same logic as create_multi_model_vector_indices.py
        model_config = config.get_multi_model_config(selected_model)
        source_index = "hf_models"  # Default source index
        
        # The vector index name will be determined by the model
        # Format: vector_{model_suffix}_{source_index}
        # e.g., vector_mpnet_hf_models, vector_e5_hf_models, vector_bge_hf_models
    
    # Search input and button side by side
    col_query, col_btn = st.columns([4, 1])
    with col_query:
        query = st.text_input(
            "Query:",
            placeholder="Enter your search query",
            label_visibility="collapsed"
        )
    with col_btn:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Perform all searches and display results
    if query and search_button:
        # Create tabs for results
        tab1, tab2, tab3 = st.tabs(["🔤 Text Search", "🧠 Vector Search", "🔄 Hybrid Search"])
        
        with tab1:
            with st.spinner("Performing text search..."):
                start_time = time.time()
                text_results = st.session_state.search_handler.text_search(
                    query, source_index, selected_fields, field_weights, top_k
                )
                text_time = time.time() - start_time
                
                display_search_results(text_results, "Text", query)
                st.caption(f"Search completed in {text_time:.3f} seconds")
        
        with tab2:
            with st.spinner(f"Performing vector search with {selected_model}..."):
                start_time = time.time()
                vector_results = st.session_state.search_handler.vector_search(
                    query, selected_model, source_index, top_k
                )
                vector_time = time.time() - start_time
                
                display_search_results(vector_results, "Vector", query)
                st.caption(f"Search completed in {vector_time:.3f} seconds")
        
        with tab3:
            with st.spinner(f"Performing hybrid search with {selected_model}..."):
                start_time = time.time()
                if hybrid_method == "RRF":
                    hybrid_results = st.session_state.search_handler.hybrid_search_rrf(
                        query, selected_model, source_index, selected_fields,
                        field_weights, text_weight, vector_weight, top_k
                    )
                else:
                    hybrid_results = st.session_state.search_handler.hybrid_search(
                        query, selected_model, source_index, selected_fields, 
                        field_weights, text_weight, vector_weight, top_k
                    )
                hybrid_time = time.time() - start_time
                if hybrid_method == "RRF":
                    display_search_results_with_details(hybrid_results, "Hybrid (RRF)", query)
                else:
                    display_search_results(hybrid_results, "Hybrid (Weighted)", query)
                st.caption(f"Search completed in {hybrid_time:.3f} seconds")

if __name__ == "__main__":
    main()
