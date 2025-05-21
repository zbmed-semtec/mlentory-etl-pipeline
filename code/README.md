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

### Full ETL Pipeline

To run the full extraction, transformation and loading process, use the automated setup script:

```bash
cd deployment
sudo ./start_mlentory_etl.sh
```

This script will:
- Detect if GPU is available and select the appropriate profile
- Create required directories with proper permissions
- Start all containers with the appropriate profile

For more options and details, see the [deployment documentation](../deployment/README.md).

### Database Management

The MLentory system includes tools for database management and visualization:

#### pgAdmin for PostgreSQL Visualization

Access the PostgreSQL databases through a web interface:
- URL: http://localhost:5050
- Credentials: admin@admin.com / admin


### Individual Components

If you want to run any of the specific components, you need to have the prerequisites installed from the [deployment documentation](../deployment/README.md#prerequisites). If you already have them installed, you can follow the instructions from any of the component folders.




