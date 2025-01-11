# MLentory Transform

A Python package for transforming ML model metadata from different sources into standardized schemas.

## Installation

If you want to install the load package in your local machine you can run the following command:

```bash
pip install -e .
```

If you want to use a docker container to install the load package you can create a new Dockerfile:

```
FROM python:3.10

COPY ./transform/ /app
WORKDIR /app

RUN pip install -e .

# Let the container run indefinitely, this is useful to keep the container running after the installation is finished.

CMD ["tail", "-f", "/dev/null"]
```

Then you can build the docker image and install the package:

```bash
docker build -t mlentory_transform .
docker run -it mlentory_transform
docker exec -it mlentory_transform /bin/bash
```

## Overview

The MLentory Transform package is designed to transform extracted ML model metadata into standardized formats. It currently supports transforming HuggingFace model metadata using configurable transformation rules and schemas.

## Features

- Transform HuggingFace model metadata into standardized schemas
- Configurable transformation rules
- Support for custom field processing
- JSON output format
- Progress tracking with tqdm

## Usage

### Basic Usage

```python
from mlentory_transform.hf_transform import TransformHF
import pandas as pd

# Load your schema and transformations
new_schema = pd.read_csv("path/to/schema.tsv", sep="\t")
transformations = pd.read_csv("path/to/transformations.tsv", sep="\t")

# Initialize transformer
transformer = TransformHF(
    new_schema=new_schema,
    transformations=transformations
)

# Transform extracted data
transformed_df = transformer.transform(
    extracted_df=your_extracted_data,
    save_output_in_json=True,
    output_dir="./outputs"
)
```

## Configuration Files

The package requires two main configuration files:

1. **Schema Definition (TSV)**
   - Defines the target schema structure
   - Example:
   ```tsv
   column_name    data_type    description
   model_name     string       Name of the model
   created_date   datetime     Creation date of the model
   ```

2. **Transformations (TSV)**
   - Defines how source fields map to target schema
   - Example:
   ```tsv
   source_column    target_column    transformation_function    parameters
   q_id_0          model_name       find_value_in_HF          {"property_name": "q_id_0"}
   q_id_2          created_date     find_value_in_HF          {"property_name": "q_id_2"}
   ```

## Available Transformation Functions

- `find_value_in_HF`: Extract values from HuggingFace properties
- `build_HF_link`: Construct HuggingFace model links
- `process_trainedOn`: Process training dataset information
- `process_softwareRequirements`: Process software requirements
- `process_not_extracted`: Handle unextracted fields

## Requirements

- Python >= 3.8.10
- pandas
- tqdm
- transformers
- watchdog
- datasets
- huggingface-hub
- Other dependencies listed in setup.py

## Package Structure

```
mlentory_transform/
├── hf_transform/
│   ├── __init__.py
│   ├── TransformHF.py
│   └── FieldProcessorHF.py
```

## Output

The transformed data can be saved in JSON format with timestamps. Example output path:
```
./outputs/2024-03-21_14-30-00_transformation_results.json
```
