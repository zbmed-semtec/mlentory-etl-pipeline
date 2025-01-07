import pandas as pd
from extractors.hf_extractor import HFExtractor
from typing import List


def load_tsv_file_to_list(path: str) -> List[str]:
    return [val[0] for val in pd.read_csv(path, sep="\t").values.tolist()]


def main():
    # Load configuration data
    config_path = "../configuration"  # Path to configuration folder

    questions = load_tsv_file_to_list(f"{config_path}/extractors/hf/questions.tsv")
    tags_language = load_tsv_file_to_list(
        f"{config_path}/extractors/hf/tags_language.tsv"
    )
    tags_libraries = load_tsv_file_to_list(
        f"{config_path}/extractors/hf/tags_libraries.tsv"
    )
    tags_other = load_tsv_file_to_list(f"{config_path}/extractors/hf/tags_other.tsv")
    tags_task = load_tsv_file_to_list(f"{config_path}/extractors/hf/tags_task.tsv")

    # Initialize extractor with configuration
    extractor = HFExtractor(
        qa_model="Intel/dynamic_tinybert",
        questions=questions,
        tags_language=tags_language,
        tags_libraries=tags_libraries,
        tags_other=tags_other,
        tags_task=tags_task,
    )

    # Download and process models
    df = extractor.download_models(
        num_models=5,  # Start with a small number for testing
        # output_dir="/transform_queue",  # Mount point in container
        save_original=True,
        save_result_in_json=False,
    )

    print(f"Processed {len(df)} models")
    print("\nSample results:")
    print(df.head())


if __name__ == "__main__":
    main()
