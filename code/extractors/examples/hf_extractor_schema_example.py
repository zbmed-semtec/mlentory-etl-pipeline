#!/usr/bin/env python3
"""
Example script demonstrating the use of HFExtractor with FAIR4ML schema.

This script shows how to use the HFExtractor class to download and process
HuggingFace model cards according to the FAIR4ML schema.
"""

import os
from mlentory_extract.hf_extract.HFExtractor import HFExtractor


def main():
    """
    Main function to demonstrate the HFExtractor with FAIR4ML schema.
    """
    # Define paths
    schema_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "configuration",
        "hf",
        "transform",
        "FAIR4ML_schema.tsv"
    )
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    
    # Define tag lists
    tags_language = ["en", "english", "fr", "french", "de", "german", "es", "spanish"]
    tags_libraries = ["pytorch", "tensorflow", "jax", "keras", "transformers", "huggingface"]
    tags_task = [
        "text classification", "token classification", "question answering",
        "summarization", "translation", "text generation", "fill mask",
        "sentence similarity", "feature extraction", "text2text generation",
        "image classification", "object detection", "image segmentation",
        "audio classification", "automatic speech recognition"
    ]
    tags_other = ["multilingual", "multimodal", "conversational", "few-shot", "zero-shot"]

    # Define questions (needed for HFExtractor initialization but not used for schema parsing)
    questions = [
        "What is the model ID?",
        "Who is the author of the model?",
        "When was the model created?",
        "What task does the model perform?",
        "What dataset was the model trained on?",
    ]

    # Initialize the extractor
    extractor = HFExtractor(
        qa_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        questions=questions,
        tags_language=tags_language,
        tags_libraries=tags_libraries,
        tags_other=tags_other,
        tags_task=tags_task,
    )

    # Download and process models using FAIR4ML schema
    print("Downloading and processing models using FAIR4ML schema...")
    processed_df = extractor.download_models_schema(
        num_models=5,  # Process 5 models for this example
        update_recent=True,
        output_dir=output_dir,
        save_result_in_json=True,
        threads=4,
        matching_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        schema_file=schema_file  # Pass the schema file path
    )

    # Print information about the processed DataFrame
    print(f"\nProcessed {len(processed_df)} models")
    print(f"Columns: {processed_df.columns.tolist()}")
    print(f"Shape: {processed_df.shape}")


if __name__ == "__main__":
    main() 