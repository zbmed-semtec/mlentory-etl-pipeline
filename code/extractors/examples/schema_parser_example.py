#!/usr/bin/env python3
"""
Example script demonstrating the use of ModelCardToSchemaParser.

This script shows how to use the ModelCardToSchemaParser to extract information
from HuggingFace model cards and map it to FAIR4ML schema properties.
"""

import os
import pandas as pd
from datetime import datetime

from mlentory_extract.core.ModelCardToSchemaParser import ModelCardToSchemaParser
from mlentory_extract.hf_extract.HFDatasetManager import HFDatasetManager


def main():
    """
    Main function to demonstrate the ModelCardToSchemaParser.
    """
    # Define paths
    schema_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "FAIR4ML_schema.tsv")
    language_tags_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "tags_language.tsv")
    libraries_tags_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "tags_libraries.tsv")
    task_tags_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "tags_task.tsv")
    other_tags_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "tags_other.tsv")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    # Define tag lists
    tags_language = [val[0] for val in pd.read_csv(language_tags_path, sep="\t").values.tolist()]
    tags_libraries = [val[0] for val in pd.read_csv(libraries_tags_path, sep="\t").values.tolist()]
    tags_task = [val[0] for val in pd.read_csv(task_tags_path, sep="\t").values.tolist()]
    tags_other = [val[0] for val in pd.read_csv(other_tags_path, sep="\t").values.tolist()]

    # Initialize the parser with schema file path
    parser = ModelCardToSchemaParser(
        matching_model="sentence-transformers/all-MiniLM-L6-v2",
        tags_language=tags_language,
        tags_libraries=tags_libraries,
        tags_task=tags_task,
        tags_other=tags_other,
        schema_file=schema_file
    )

    # Initialize the dataset manager
    dataset_manager = HFDatasetManager()

    # Get model metadata (limit to 5 models for this example)
    print("Downloading model metadata...")
    hf_df = dataset_manager.get_model_metadata_dataset(limit=10)
    hf_df.to_csv(os.path.join(output_dir,"hf_df.csv"), index=False)
    
    # Process the DataFrame
    print("Processing model metadata...")
    processed_df = parser.process_dataframe(hf_df)
    
    # Print detailed information about the processed DataFrame
    # parser.print_detailed_dataframe(processed_df)
    
    # Save the results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f"{timestamp}_FAIR4ML_Schema_Models.json")
    
    processed_df.to_json(path_or_buf=output_path, orient="records", indent=4)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main() 