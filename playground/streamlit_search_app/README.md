# ML Model Search Streamlit Application

A web-based interface for searching and discovering ML models using advanced vector similarity and text search capabilities.

## Features

### 🤖 Multiple Embedding Models
- **sentence-transformers/all-mpnet-base-v2**: High quality general purpose embeddings
- **intfloat/e5-base-v2**: Optimized for retrieval tasks  
- **BAAI/bge-base-en-v1.5**: Balanced general embeddings

### 📝 Configurable Field Search
- **name**: Model name (weight: 3.0)
- **description**: Model description (weight: 2.5)
- **mlTask**: ML tasks (weight: 2.0)
- **keywords**: Keywords/tags (weight: 1.5)
- **sharedBy**: Creator information (weight: 0.5)

### 🔍 Three Search Types
1. **Text Search**: Traditional Elasticsearch text matching with field weights
2. **Vector Search**: Semantic similarity using embedding models
3. **Hybrid Search**: Combines text and vector search with adjustable weights

### ⚙️ Interactive Controls
- Embedding model selection
- Field selection with checkboxes
- Editable field weights (sliders)
- Hybrid search weight adjustment
- Number of results control

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Elasticsearch is running:**
   - The app connects to Elasticsearch using the same configuration as the vector_similarity_poc
   - Make sure your Elasticsearch instance is running and accessible

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Configure Search Settings** (Sidebar):
   - Select embedding model
   - Choose search fields
   - Adjust field weights
   - Set hybrid search weights
   - Choose number of results

2. **Enter Search Query**:
   - Type your search query in the text input
   - Use natural language descriptions

3. **Choose Search Type**:
   - Click "Text Search" for traditional text matching
   - Click "Vector Search" for semantic similarity
   - Click "Hybrid Search" for combined approach

4. **View Results**:
   - Results are displayed in tabs
   - Each result shows model information, scores, and descriptions
   - Expandable sections for detailed information

## Configuration

The app uses the same configuration as the vector_similarity_poc:
- Elasticsearch connection settings
- Embedding model configurations
- Field weights and search parameters

