import pandas as pd
import logging
import datetime
from typing import List
from tqdm import tqdm
import os
import argparse
import time
from mlentory_extract.ai4life_extract import AI4LifeExtractor


def setup_logging() -> logging.Logger:
    """
    Sets up the logging system.
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    base_log_path = "./ai4life_etl/execution_logs"
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
    
def initialize_extractor(config_path:str) -> AI4LifeExtractor:
    """
    Initializes the extractor
    Args:
        config_path (str): The path to the
    Returns:
        AI4LifeExtractor: The extractor instance.
    """
    schema_file = f"{config_path}/extract/model_mapping.tsv"
    return AI4LifeExtractor(
        schema_file = schema_file
    )
    
def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments.
    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="AI4Life ETL Process")
    parser.add_argument(
        "--save-extraction",
        action="store_true",
        default=True,
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
        "--num-models", type=int, default=10000, help="Number of models to download"
    )
    parser.add_argument(
        "--output-dir",
        default="./ai4life_etl/outputs/files",
        help="Directory to save intermediate results",
    )
    return parser.parse_args()
def main():
    args = parse_args()
    setup_logging()
    # Setup configuration data
    config_path = "./configuration/ai4life"  # Path to configuration folder
    # Extract
    start_time = time.time()
    extractor = initialize_extractor(config_path)
    end_time = time.time()
    print(f"Initialization time: {end_time - start_time} seconds")
    start_time = time.time()
    extracted_df = extractor.download_modelfiles_with_additional_entities(
        num_models=args.num_models,
        output_dir=args.output_dir,
        additional_entities = ["dataset", "application"]
    )
    end_time = time.time()
    print(f"Extraction time: {end_time - start_time} seconds")
    
if __name__ == "__main__":
    main()