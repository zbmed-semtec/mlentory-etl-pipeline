# MLentory Load

A Python package for loading and managing ML model metadata across multiple database systems.

## Installation

If you want to install the load package in your local machine you can run the following command:
```bash
pip install -e .
```

If you want to use a docker container to install the load package you can create a new Dockerfile:

```
FROM python:3.10

COPY ./load/ /app
WORKDIR /app

RUN pip install -e .

# Let the container run indefinitely, this is useful to keep the container running after the installation is finished.

CMD ["tail", "-f", "/dev/null"]
```

Then you can build the docker image and install the package:
```bash
docker build -t mlentory_load .
docker run -it mlentory_load
docker exec -it mlentory_load /bin/bash
```

## Overview

MLentory Load manages the storage and synchronization of ML model metadata across three different database systems:
- PostgreSQL (Relational Database)
- Virtuoso (RDF Triple Store)
- Elasticsearch (Search Index)

## Features

- Multi-database synchronization
- Version control for model metadata
- Graph-based data management
- Full-text search capabilities
- Support for temporal queries
- Batch processing of model updates

## Components

### Database Handlers

1. **SQLHandler**: PostgreSQL database operations
   - CRUD operations for model metadata
   - Version tracking
   - Extraction information storage

2. **RDFHandler**: Virtuoso RDF store management
   - Graph operations
   - SPARQL query support
   - Docker container management

3. **IndexHandler**: Elasticsearch operations
   - Full-text search
   - Model metadata indexing
   - Batch document processing

### Core Components

1. **GraphHandler**: Manages graph operations
   - Triplet processing
   - Version control
   - Multi-database synchronization

2. **LoadProcessor**: Main processing pipeline
   - Coordinates database operations
   - Manages data flow
   - Handles batch processing

## Prerequisites

You need to have the require databases running and the required credentials to connect to them in order to run the load package.

If you want to setup your own databases you can follow the instructions in the [deployment documentation](deployment/README.md). In the docker-compose.yml file you can find examples of the required databases and their credentials, and how to set them up.

## Usage

```python
from mlentory_load.dbHandler import SQLHandler, RDFHandler, IndexHandler
from mlentory_load.core import LoadProcessor, GraphHandler

# Initialize handlers
sql_handler = SQLHandler(
    host="localhost",
    user="user",
    password="pass",
    database="mlentory"
)

rdf_handler = RDFHandler(
    container_name="virtuoso",
    _user="dba",
    _password="dba",
    kg_files_directory="./kg_files",
    sparql_endpoint="http://localhost:8890/sparql"
)

index_handler = IndexHandler(
    es_host="localhost",
    es_port=9200
)

# Initialize graph handler
graph_handler = GraphHandler(
    SQLHandler=sql_handler,
    RDFHandler=rdf_handler,
    IndexHandler=index_handler,
    kg_files_directory="./kg_files"
)

# Initialize load processor
processor = LoadProcessor(
    SQLHandler=sql_handler,
    RDFHandler=rdf_handler,
    IndexHandler=index_handler,
    GraphHandler=graph_handler,
    kg_files_directory="./kg_files"
)

# Load data
processor.load_df(your_dataframe, output_ttl_file_path="./output")
```

## Requirements

- Python >= 3.8.10
- PostgreSQL
- Virtuoso Triple Store
- Elasticsearch 7.x
- Docker
- Additional dependencies listed in setup.py

## Database Setup

### PostgreSQL
- Requires a PostgreSQL instance
- Database schema will be automatically created

### Virtuoso
- Requires a running Virtuoso instance in Docker
- Default port: 8890

### Elasticsearch
- Requires Elasticsearch 7.x
- Default port: 9200
- Basic authentication enabled

## Output

The system generates several types of output:
1. SQL tables with version control
2. RDF graphs in TTL format
3. Elasticsearch indices
4. Temporal metadata tracking
