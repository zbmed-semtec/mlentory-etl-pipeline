# MLentory ETL Pipeline Code

This folder contains the core implementation of the MLentory ETL (Extract, Transform, Load) pipeline, designed to collect and process machine learning model metadata from various sources.

<img src="../docs/Readme_images/MLentory Backend TDD Diagrams-Main_component_interaction_Diagram_v3.png"/>
<p style="text-align: center; font-size: 0.8em; color: #cccccc">MLentory Pipeline Architecture</p>

## Project Structure

```cmd
code/
├── extractors/   
├── transform/   
└── load/        
```

### 1. Extractors

Platform-specific modules that extract ML model metadata from different sources:
- HuggingFace Hub extractor
- Future extractors for other platforms

For detailed information, see the [extractors documentation](extractors/README.md)

### 2. Transform

Transforms extracted data into a standardized schema:
- Configurable transformation rules
- Field processing and validation
- Schema mapping

For detailed information, see the [transform documentation](transform/README.md) .

### 3. Load
Handles storage and versioning of processed data:
- PostgreSQL for relational data
- Virtuoso for RDF triples
- Elasticsearch for search capabilities

For detailed information, see the [load documentation](load/README.md) .

## Run the project

If you want to run the full extraction, transformation and loading process you can follow the instructions in the [deployment documentation](deployment/README.md).

If you want to run any of the specific components you need to have the prerequisites installed from the [deployment documentation](deployment/README.md#prerequisites), if you already have them installed you can follow the instuctions from any of the components folders.




