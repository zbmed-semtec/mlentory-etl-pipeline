import os
import time
import logging
import argparse
import datetime
import pandas as pd
from typing import List

import numpy as np
np.float_ = np.float64
import pandas as pd
from rdflib import Graph

from mlentory_extract.openml_extract import OpenMLExtractor
from mlentory_transform.core.MlentoryTransform import (
    MlentoryTransform,
    KnowledgeGraphHandler,
)

def load_tsv_file_to_list(path: str) -> List[str]:
    return [val[0] for val in pd.read_csv(path, sep="\t").values.tolist()]

def setup_logging() -> logging.Logger:
    """
    Sets up the logging system.
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    base_log_path = "./openml_etl/execution_logs"
    os.makedirs(base_log_path, exist_ok=True, mode=0o777)
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

def initialize_transform_hf(config_path: str) -> MlentoryTransform:
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

    kg_handler = KnowledgeGraphHandler(
        FAIR4ML_schema_data=new_schema, 
        base_namespace="http://mlentory.zbmed.de/mlentory_graph/"
    )

    transformer = MlentoryTransform(kg_handler, None)

    return transformer

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
        "--num-instances", type=int, default=20, help="Number of instances to extract metadata from"
    )

    parser.add_argument(
        "--offset", type=int, default=0, help="Number of instances to skip before extracting"
    )

    parser.add_argument(
        "--threads", type=int, default=4, help="Number of threads for parallel processing"
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, mode=0o777) 

    # Setup configuration data
    config_path = "./configuration/openml"  # Path to configuration folder

    # Extract
    extractor = initialize_extractor(config_path)

    extracted_entities = extractor.extract_run_info_with_additional_entities(
        num_instances=args.num_instances, 
        offset=args.offset,
        threads=args.threads, 
        output_dir=args.output_dir,
        save_result_in_json=args.save_extraction,
        additional_entities=["dataset"]
        )
    
    # Transform
    transformer = initialize_transform_hf(config_path)

    print(type(transformer))

    models_kg, models_extraction_metadata = transformer.transform_OpenML_runs(
        extracted_df=extracted_entities["run"],
        save_output_in_json=True,
        output_dir=args.output_dir+"/runs",
    )


if __name__ == "__main__":
    main()