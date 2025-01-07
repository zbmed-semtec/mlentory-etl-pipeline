import pandas as pd
import logging
import datetime
from typing import List
from tqdm import tqdm
import os

from extractors.hf_extractor import HFExtractor
from transform.core.FilesProcessor import FilesProcessor
from transform.core.FieldProcessorHF import FieldProcessorHF


def load_tsv_file_to_list(path: str) -> List[str]:
    return [val[0] for val in pd.read_csv(path, sep="\t").values.tolist()]


def main():
    # Setting up logging system
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    base_log_path = "./hf_etl/execution_logs"
    os.makedirs(base_log_path, exist_ok=True)
    logging_filename = f"{base_log_path}/transform_{timestamp}.log"

    logging.basicConfig(
        filename=logging_filename,
        filemode="w",
        format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Load configuration data
    config_path = "./configuration/hf"  # Path to configuration folder

    questions = load_tsv_file_to_list(f"{config_path}/extract/questions.tsv")
    tags_language = load_tsv_file_to_list(f"{config_path}/extract/tags_language.tsv")
    tags_libraries = load_tsv_file_to_list(f"{config_path}/extract/tags_libraries.tsv")
    tags_other = load_tsv_file_to_list(f"{config_path}/extract/tags_other.tsv")
    tags_task = load_tsv_file_to_list(f"{config_path}/extract/tags_task.tsv")

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
        save_original=False,
        save_result_in_json=False,
    )

    # print(f"Processed {len(df)} models")
    # print("\nSample results:")
    # print(df.head())

    # Initializing the updater
    new_schema = pd.read_csv(f"{config_path}/transform/M4ML_schema.tsv", sep="\t")
    transformations = pd.read_csv(
        f"{config_path}/transform/column_transformations.csv",
        lineterminator="\n",
        sep=",",
    )
    fields_processor_HF = FieldProcessorHF(new_schema, transformations)
    processed_models = []

    for row_num, row in tqdm(
        df.iterrows(), total=len(df), desc="Transforming progress"
    ):
        model_data = fields_processor_HF.process_row(row)
        processed_models.append(model_data)

    m4ml_models_df = pd.DataFrame(list(processed_models))

    print("Transformed models:")
    print(m4ml_models_df.head())

    m4ml_models_df.to_json(
        f"./hf_etl/M4ML_schema_transformed.json", orient="records", indent=4
    )


if __name__ == "__main__":
    main()
