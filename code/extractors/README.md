# MLentory Extract

A package for extracting model information from different ML platforms.

## Installation

If you want to install the load package in your local machine you can run the following command:
```bash
pip install -e .
```

If you want to use a docker container to install the load package you can create a new Dockerfile:

```
FROM python:3.10

COPY ./extractors/ /app
WORKDIR /app

RUN pip install -e .

# Let the container run indefinitely, this is useful to keep the container running after the installation is finished.

CMD ["tail", "-f", "/dev/null"]
```

Then you can build the docker image and install the package:
```bash
docker build -t mlentory_extract .
docker run -it mlentory_extract
docker exec -it mlentory_extract /bin/bash
```

## Usage

1. **Basic Usage**:

```python
from mlentory_extract.hf_extract import HFExtractor

# Initialize extractor
extractor = HFExtractor(
    qa_model="Intel/dynamic_tinybert",
    questions=questions,
    tags_language=tags_language,
    tags_libraries=tags_libraries,
    tags_other=tags_other,
    tags_task=tags_task
)

# Download and process models
df = extractor.download_models(
    num_models=10,
    output_dir="./outputs",
    save_raw_data=True,
    save_result_in_json=True
)
```

2. **Using with Configuration Files**:

```python
import pandas as pd
from mlentory_extract.hf_extract import HFExtractor

def load_tsv_file_to_list(path: str) -> list[str]:
    return [val[0] for val in pd.read_csv(path, sep="\t").values.tolist()]

# Load configuration data
config_path = "./config_data"
questions = load_tsv_file_to_list(f"{config_path}/questions.tsv")
tags_language = load_tsv_file_to_list(f"{config_path}/tags_language.tsv")
tags_libraries = load_tsv_file_to_list(f"{config_path}/tags_libraries.tsv")
tags_other = load_tsv_file_to_list(f"{config_path}/tags_other.tsv")
tags_task = load_tsv_file_to_list(f"{config_path}/tags_task.tsv")

# Initialize and use extractor
extractor = HFExtractor(
    qa_model="Intel/dynamic_tinybert",
    questions=questions,
    tags_language=tags_language,
    tags_libraries=tags_libraries,
    tags_other=tags_other,
    tags_task=tags_task
)

# Process models with additional options
df = extractor.download_models(
    num_models=10,
    output_dir="./outputs",
    save_raw_data=True,        # Save original dataset
    save_result_in_json=True,  # Save results in JSON format
    from_date="2024-01-01"    # Filter models by date
)
```

## Configuration Files

The package uses TSV (Tab-Separated Values) files for configuration. Each file should contain one item per line. Here are examples of the expected format:

1. **questions.tsv** - Questions for information extraction:
```tsv
What is the name of the model?
What user shared the model?
What tasks can the model solve?
What datasets was the model trained on?
```

2. **tags_language.tsv** - Supported language tags:
```tsv
en
zh
de
es
fr
```

3. **tags_libraries.tsv** - Supported ML libraries:
```tsv
PyTorch
TensorFlow
JAX
Transformers
Diffusers
```

4. **tags_task.tsv** - Supported ML tasks:
```tsv
Image Classification
Object Detection
Text Classification
Question Answering
Translation
```

5. **tags_other.tsv** - Other relevant tags:
```tsv
Inference Endpoints
AutoTrain Compatible
Has a Space
4-bit precision
8-bit precision
```

Configuration files should be placed in a directory structure like:
```
config_data/
├── questions.tsv
├── tags_language.tsv
├── tags_libraries.tsv
├── tags_other.tsv
└── tags_task.tsv
```

## Package Structure

```
mlentory_extract/
├── core/
│   ├── __init__.py
│   └── ModelCardQAParser.py
└── hf_extract/
    ├── __init__.py
    └── HFExtractor.py
```

## Requirements

- Python >= 3.8.10
- pandas
- transformers
- datasets
- torch
- huggingface-hub
- tqdm


