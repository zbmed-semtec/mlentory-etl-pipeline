# MLEntory ETL Pipeline Tests

This folder contains all the test files for the MLentory ETL pipeline. The tests are organized to validate each component of the pipeline's functionality.

## Test Structure

The tests are organized according to the main components of the ETL pipeline:

### Extract
- `tests/unit/hf/extractors/test_HFExtractor.py` - Tests for the HuggingFace model extraction
- `tests/unit/hf/extractors/test_ModelCardQAParser.py` - Tests for parsing model card information

### Transform
- `tests/unit/hf/transform/test_FieldProcessorHF.py` - Tests for field processing and transformations

### Load
- `tests/unit/hf/load/test_Elasticsearch.py` - Tests for Elasticsearch integration
- `tests/unit/hf/load/test_GraphHandler.py` - Tests for graph data handling
- `tests/unit/hf/load/test_IndexHandler.py` - Tests for index management
- `tests/unit/hf/load/test_SQLHandler.py` - Tests for SQL database operations

## Running the Tests

### Using Docker Compose (Recommended)

1. Build the test containers:
```bash
docker-compose --profile test build
```

2. Start the test environment:
```bash
docker-compose --profile test up
```

3. Access the test container:
```bash
docker ps  # Find the test container ID
docker exec -it <test_container_name> /bin/bash
```

4. Run the tests:
```bash
pytest  # Run all tests
pytest tests/unit/hf/extractors/  # Run specific test directory
pytest tests/unit/hf/extractors/test_HFExtractor.py  # Run specific test file
```

### Using Shell Script

If you have WSL2 or are on a Unix-based system:

1. Navigate to the tests directory
2. Run:
```bash
bash validate_tests.sh
```

## Test Dependencies

The test environment requires several services:
- PostgreSQL for SQL database testing
- Elasticsearch for search functionality
- Virtuoso for graph database operations

These services are automatically configured when using Docker Compose.

For more detailed information about the ETL pipeline components and their interactions, refer to the main documentation.



