# Vector Similarity POC - Complete Guide

## 🎯 What is This?

This is a **Proof of Concept (POC)** for implementing **vector similarity search** in the MLentory project. It adds semantic search capabilities to find ML models based on meaning, not just exact text matches.

### **The Problem We're Solving**
- **Current search**: "Show me models with 'BERT' in the name" → Only finds models with exact word "BERT"
- **Vector search**: "Show me models for language understanding" → Finds BERT, GPT, RoBERTa, etc. even if they don't contain those exact words

## 🚀 Quick Start Guide

### **Step 1: Install Dependencies**
```bash
cd /home/ubuntu/suhasini/mlentory-etl-pipeline/playground/vector_similarity_poc
conda create --name vs_poc
conda activate vs_poc
pip install -r requirements.txt
```

### **Step 2: Make Sure Elasticsearch is Running**
```bash
# Check if Elasticsearch is running
curl http://localhost:9200
# Should return: {"name":"elastic","cluster_name":"es-docker-cluster",...}
```

### **Step 3: Test the System**
```bash
# Test with real data (all search methods)
python3 test_real_data.py
```

### **Step 4: Test Command Line Search**
```bash
# Quick search test
python3 search_cli.py "language understanding model"

# Interactive mode for multiple queries
python3 search_cli.py --interactive

# More results
python3 search_cli.py "transformer model" --top-k 10
```

## Model Evaluation

### **Dimension-Grouped Model Evaluation**

For comprehensive testing of different embedding models, use the dimension-grouped evaluation script:

```bash
# Run complete evaluation of multiple embedding models
python3 dimension_grouped_evaluation.py
```

**What it does:**
- Tests multiple embedding models grouped by their vector dimensions (384D, 768D)
- Compares text search, vector search, and hybrid search methods
- Evaluates models on semantic queries like "language understanding model"
- Generates detailed performance reports and rankings
- Saves results to timestamped JSON files

**Models tested:**
- **384D Models**: `all-MiniLM-L6-v2`, `all-MiniLM-L12-v2` (fast, good quality)
- **768D Models**: `all-mpnet-base-v2`, `paraphrase-multilingual-mpnet-base-v2` (best quality)

