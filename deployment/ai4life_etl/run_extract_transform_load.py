import pandas as pd
import logging
import datetime
from typing import List
from tqdm import tqdm
import os
import argparse
import time
from datetime import datetime
from mlentory_extract.ai4life_extract import AI4LifeExtractor
from mlentory_transform.core import (
    MlentoryTransform,
    KnowledgeGraphHandler,
    MlentoryTransformWithGraphBuilder,
)

def setup_logging() -> logging.Logger:
    """
    Sets up the logging system.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    base_log_path = "./hf_etl/outputs/execution_logs"
    os.makedirs(base_log_path, exist_ok=True)
    logging_filename = f"{base_log_path}/transform_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        handlers=[
            logging.FileHandler(logging_filename, mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger
    
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

def initialize_transform(config_path: str) -> MlentoryTransform:
    """
    Initializes the transformer with the configuration data.

    Args:
        config_path (str): The path to the configuration data.

    Returns:
        MlentoryTransform: The transformer instance.
    """
    new_schema = pd.read_csv(
        f"{config_path}/transform/FAIR4ML_schema.csv", sep=",", lineterminator="\n"
    )

    transformer = MlentoryTransformWithGraphBuilder(base_namespace="https://w3id.org/mlentory/mlentory_graph/", FAIR4ML_schema_data=new_schema)

    return transformer

    
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
    logger = setup_logging()

    # Setup configuration data
    config_path = "./configuration/ai4life"  # Path to configuration folder
    # Extract
    start_time = time.time()
    extractor = initialize_extractor(config_path)
    end_time = time.time()
    logger.info(f"Initialization time: {end_time - start_time} seconds")
    start_time = time.time()
    extracted_entities = extractor.download_modelfiles_with_additional_entities(
        num_models=args.num_models,
        output_dir=args.output_dir,
        additional_entities = ["dataset", "application"]
    )
    end_time = time.time()
    logger.info(f"Extraction time: {end_time - start_time} seconds")

    # Initialize transformer (outside the if/else)
    logger.info("Initializing transformer...")
    start_time = time.time()
    transformer = initialize_transform(config_path)
    end_time = time.time()
    logger.info(f"Transformer initialization took {end_time - start_time:.2f} seconds")

    logger.info("Starting transformation process...")
    start_time = time.time()
    kg_integrated, kg_metadata_integrated = transformer.transform_AI4Life_models_with_related_entities(
        extracted_entities=extracted_entities,
        save_output=True,
        kg_output_dir=args.output_dir+"/kg",
    )
    end_time = time.time()
    logger.info(f"Transformation process took {end_time - start_time:.2f} seconds")
    
if __name__ == "__main__":
    main()