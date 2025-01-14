import pandas as pd
import logging
import datetime
from typing import List
from tqdm import tqdm
import os
import argparse
import time

from mlentory_extract.hf_extract import HFExtractor
from mlentory_transform.hf_transform.FieldProcessorHF import FieldProcessorHF
from mlentory_load.core import LoadProcessor, GraphHandler
from mlentory_load.dbHandler import RDFHandler, SQLHandler, IndexHandler
from mlentory_transform.hf_transform.TransformHF import TransformHF


def load_tsv_file_to_list(path: str) -> List[str]:
    return [val[0] for val in pd.read_csv(path, sep="\t").values.tolist()]


def setup_logging() -> logging.Logger:
    """
    Sets up the logging system.
    """
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


def initialize_extractor(config_path: str) -> HFExtractor:
    """
    Initializes the extractor with the configuration data.

    Args:
        config_path (str): The path to the configuration data.

    Returns:
        HFExtractor: The extractor instance.
    """
    questions = load_tsv_file_to_list(f"{config_path}/extract/questions.tsv")
    tags_language = load_tsv_file_to_list(f"{config_path}/extract/tags_language.tsv")
    tags_libraries = load_tsv_file_to_list(f"{config_path}/extract/tags_libraries.tsv")
    tags_other = load_tsv_file_to_list(f"{config_path}/extract/tags_other.tsv")
    tags_task = load_tsv_file_to_list(f"{config_path}/extract/tags_task.tsv")
    return HFExtractor(
        qa_model="Intel/dynamic_tinybert",
        questions=questions,
        tags_language=tags_language,
        tags_libraries=tags_libraries,
        tags_other=tags_other,
        tags_task=tags_task,
    )


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="HuggingFace ETL Process")
    parser.add_argument(
        "--save-extraction",
        action="store_true",
        default=False,
        help="Save the results of the extraction phase",
    )
    parser.add_argument(
        "--save-transformation",
        action="store_true",
        default=False,
        help="Save the results of the transformation phase",
    )
    parser.add_argument(
        "--save-load-data",
        action="store_true",
        default=False,
        help="Save the data that will be loaded into the database",
    )
    parser.add_argument(
        "--from-date",
        type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d"),
        default=datetime.datetime(2000, 1, 1),
        help="Download models from this date (format: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--num-models", type=int, default=10, help="Number of models to download"
    )
    parser.add_argument(
        "--output-dir",
        default="./hf_etl/outputs/files",
        help="Directory to save intermediate results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    # Setup configuration data
    config_path = "./configuration/hf"  # Path to configuration folder

    # Extract
    start_time = time.time()
    extractor = initialize_extractor(config_path)
    end_time = time.time()
    print(f"Initialization time: {end_time - start_time} seconds")

    start_time = time.time()
    extracted_df = extractor.download_models(
        num_models=args.num_models,
        from_date=args.from_date,
        save_result_in_json=args.save_extraction,
        save_raw_data=False,
    )
    end_time = time.time()
    print(f"Extraction time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
