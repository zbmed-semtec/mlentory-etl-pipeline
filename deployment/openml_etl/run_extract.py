import os
import time
import logging
import argparse
import datetime
import pandas as pd
from tqdm import tqdm
from typing import List

from mlentory_extract.openml_extract import OpenMLExtractor

def setup_logging() -> logging.Logger:
    """
    Sets up the logging system.
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    base_log_path = "./openml_etl/execution_logs"
    os.makedirs(base_log_path, exist_ok=True)
    logging_filename = f"{base_log_path}/extraction_{timestamp}.log"

    logging.basicConfig(
        filename=logging_filename,
        filemode="w",
        format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    return logger

def initialize_extractor(config_path: str) -> OpenMLExtractor:
    """
    Initializes the extractor with the configuration data.

    Args:
        config_path (str): The path to the configuration data.

    Returns:
        OpenMLExtractor: The extractor instance.
    """
    schema_file = f"{config_path}/extract/metadata_schema.json"

    return OpenMLExtractor(
        schema_file = schema_file
    )

def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="OpenML ETL Process")

    parser.add_argument(
        "--save-extraction",
        action="store_true",
        default=True,
        help="Save the results of the extraction phase",
    )
    
    parser.add_argument(
        "--num-runs", type=int, default=20, help="Number of runs to extract metadata from"
    )

    parser.add_argument(
        "--output-dir",
        default="./openml_etl/outputs",
        help="Directory to save results",
    )
    return parser.parse_args()
    
    
def main():
    args = parse_args()
    logger = setup_logging()

    # Setup configuration data
    config_path = "./configuration/openml"  # Path to configuration folder

    # Extract
    start_time = time.time()
    extractor = initialize_extractor(config_path)
    end_time = time.time()
    logger.info(f"Initialization time: {end_time - start_time} seconds")
    
    start_time = time.time()
    extracted_df = extractor.get_multiple_runs_metadata(
        num_runs=args.num_runs, 
        output_dir=args.output_dir,
        save_result_in_json=args.save_extraction
    )
    end_time = time.time()

    # log.info DataFrame details
    logger.info("\n--- DataFrame Summary ---\n")
    logger.info(f"Shape: {extracted_df.shape}")  # (rows, columns)
    logger.info(f"Number of Rows: {extracted_df.shape[0]}")
    logger.info(f"Number of Columns: {extracted_df.shape[1]}")

    logger.info("\n--- Column Info ---\n")
    logger.info(extracted_df.info(),"\n")  # Shows data types and missing values

    logger.info("--- First Few Rows ---\n")
    logger.info(extracted_df.head(),"\n")  # Displays first 5 rows

    logger.info(f"Extraction Time: {end_time - start_time:.2f} seconds")
    
    logger.info(f"Extraction time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()