# Load Component

This package provides the loading functionality for the FAIR4ML system. It handles:

- File monitoring and processing
- Database interactions (SQL, RDF, Elasticsearch)
- Graph handling and updates
- Data loading and transformation

## Installation

```bash
pip install mlentory-loader
```

## Usage

Basic usage:

```python
from mlentory_load.core import LoadProcessor, FileProcessor
from mlentory_load.core.db_handler import SQLHandler, RDFHandler, IndexHandler
from mlentory_load.core.graph_handler import GraphHandler

sql_handler = SQLHandler(host="localhost", user="user", password="pass", database="db")


```
