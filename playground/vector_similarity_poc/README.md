# Vector Similarity POC

A Proof of Concept for implementing vector similarity search in the MLentory project, enabling semantic search capabilities to find ML models based on meaning rather than using only text matches.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Elasticsearch running on localhost:9200

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Verify Elasticsearch
curl http://localhost:9200
```

### Usage
```bash
# 1. Create vector indices
python scripts/index_creation/create_single_vector_index.py

# 2. Run searches
python scripts/search/search_cli_native.py "language understanding"
python scripts/search/search_cli_native.py --interactive

# 3. Multi-vector search
python scripts/search/search_multi_cli_native.py "transformer model"

# 4. N-gram enhanced search
python scripts/search/search_multi_ngram_cli_native.py "computer vision"
```

## 🔍 Search Methods

- **Text Search**: Fast keyword matching with fuzzy search
- **Vector Search**: Semantic similarity using embeddings  
- **Hybrid Search**: Combines both approaches (recommended)

## ⚙️ Configuration

Configuration is in `src/config.py`. Key settings:
- Embedding model: `sentence-transformers/all-mpnet-base-v2`
- Elasticsearch: `localhost:9200`
- Search size: 10 results default

## 🧪 Available Scripts

### Search Scripts
- **`search_cli_native.py`** - Basic native Elasticsearch search
- **`search_multi_cli_native.py`** - Multi-vector search implementation  
- **`search_multi_ngram_cli_native.py`** - N-gram enhanced search

### Index Creation Scripts
- **`create_single_vector_index.py`** - Single vector per model
- **`create_multi_vector_index.py`** - Multiple vectors per model (recommended)
- **`create_multi_ngrams_vector_index.py`** - N-gram enhanced vectors

## 🆘 Troubleshooting

- **Elasticsearch not running**: `curl http://localhost:9200`
- **Import errors**: `pip install -r requirements.txt`
- **Index not found**: Run index creation scripts first
- **Search not working**: Check if vector indices exist in Elasticsearch


