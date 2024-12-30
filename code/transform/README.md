# Transform Package

A Python package for transforming ML model metadata from different sources into the M4ML schema.

## Overview

The transform package is part of an ETL pipeline for ML model metadata. It processes extracted metadata from HuggingFace and transforms it into a standardized M4ML schema format. The package implements a queue-based system that monitors a directory for new files and processes them automatically.

## Features

- Parallel processing of multiple files
- Configurable batch processing
- Standardized transformation to M4ML schema
- Comprehensive logging system

## Installation

The package requires Python 3.8.10 or later. To install the package:

```bash
pip install -e .
```

## Usage

### Basic Usage

```python
from transform import QueueObserver, FilesProcessor, FieldProcessorHF

# Initialize the field processor
field_processor = FieldProcessorHF(path_to_config_data="path/to/config")

# Initialize the files processor
files_processor = FilesProcessor(
    num_workers=4,
    next_batch_proc_time=30,
    processed_files_log_path="./processing_logs/Processed_files.txt",
    load_queue_path="./load_queue",
    field_processor_HF=field_processor
)

# Initialize and start the queue observer
observer = QueueObserver(watch_dir="./transform_queue", files_processor=files_processor)
observer.start()
```

### Command Line Usage

The package includes a command-line interface:

```bash
python -m transform.main --folder /path/to/watch/directory
```

## Directory Structure

- `/transform_queue`: Directory monitored for new files to process
- `/config_data`: Contains configuration files including M4ML schema
- `/load_queue`: Output directory for transformed files
- `/processing_logs`: Contains processing logs and record of processed files
- `/execution_logs`: Contains execution logs with detailed information about the transform process

## Configuration

The package requires:
- M4ML schema configuration file in TSV format
- Input directory for monitoring (`transform_queue`)
- Output directory for transformed files (`load_queue`)
- Directory for logs (`processing_logs` and `execution_logs`)

## Docker Support

The package can be run in a Docker container. A Dockerfile is provided in the repository.

To build and run with Docker:

```bash
docker build -t transform .
docker run -v /path/to/transform_queue:/transform_queue \
          -v /path/to/config_data:/config_data \
          -v /path/to/load_queue:/load_queue \
          transform
```

## Logging

The package implements two types of logging:
1. Execution logs: General runtime information and errors
2. Processing logs: Detailed information about processed files

Logs are stored in their respective directories with timestamps for easy tracking.

## Dependencies

Main dependencies include:
- pandas
- watchdog
- rdflib
- tqdm

For a complete list of dependencies, see `setup.py`.
