# Extractors

A package for extracting model information from different platforms.

## Installation

```bash
pip install -e .
```

## Usage

1. **Basic Usage**:

```python
from extractors.hf_extractor import HFExtractor

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
    save_original=True
)

2. **Using with Configuration Files**:
```python
import pandas as pd
from extractors.hf_extractor import HFExtractor

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
```


