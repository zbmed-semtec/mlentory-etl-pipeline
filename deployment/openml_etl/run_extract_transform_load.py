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
import sys
import os

from mlentory_extract.openml_extract import OpenMLExtractor
from mlentory_transform.core.MlentoryTransform import (
    MlentoryTransform,
    KnowledgeGraphHandler,
)
from mlentory_load.core import LoadProcessor, GraphHandlerForKG
from mlentory_load.dbHandler import SQLHandler, RDFHandler, IndexHandler

# Load environment variables with defaults
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "history_DB")

VIRTUOSO_HOST = os.getenv("VIRTUOSO_HOST", "virtuoso_db")
VIRTUOSO_USER = os.getenv("VIRTUOSO_USER", "dba")
VIRTUOSO_PASSWORD = os.getenv("VIRTUOSO_PASSWORD", "my_strong_password")
VIRTUOSO_SPARQL_ENDPOINT = os.getenv("VIRTUOSO_SPARQL_ENDPOINT", f"http://{VIRTUOSO_HOST}:8890/sparql") # Default uses VIRTUOSO_HOST

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "elastic_db")
ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200")) # Env vars are strings


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
        level=logging.INFO,  # Set level for the root logger
    )
    logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO) # No longer needed as root logger level is set

    # Create a StreamHandler to output logs to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s - %(levelname)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    console_handler.setFormatter(formatter)
    logging.getLogger("").addHandler(console_handler) # Add handler to the root logger

    logger.info(f"Logging to file: {os.path.abspath(logging_filename)}")

    return logger

def initialize_extractor(config_path: str, logger: logging.Logger) -> OpenMLExtractor:
    """
    Initializes the extractor with the configuration data.

    Args:
        config_path (str): The path to the configuration data.
        logger (logging.Logger): Logger instance for logging events.

    Returns:
        OpenMLExtractor: The extractor instance.
    """
    logger.info("Initializing extractor")
    schema_file = f"{config_path}/extract/metadata_schema.json"
    logger.info(f"Using schema file: {schema_file}")
    try:
        extractor = OpenMLExtractor(schema_file=schema_file, logger=logger)
        logger.info("Extractor initialized successfully")
        return extractor
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {str(e)}")
        raise

def initialize_transformer(config_path: str, logger: logging.Logger) -> MlentoryTransform:
    """
    Initializes the transformer with the configuration data.

    Args:
        config_path (str): The path to the configuration data.
        logger (logging.Logger): Logger instance for logging events.

    Returns:
        MlentoryTransform: The transformer instance.
    """
    logger.info("Initializing transformer")
    try:
        new_schema = pd.read_csv(
            f"{config_path}/transform/FAIR4ML_schema.csv", sep=",", lineterminator="\n"
        )
        logger.info("FAIR4ML schema loaded successfully")
        
        kg_handler = KnowledgeGraphHandler(
            FAIR4ML_schema_data=new_schema, 
            base_namespace="https://w3id.org/mlentory/mlentory_graph/"
        )
        logger.info("KnowledgeGraphHandler initialized")
        
        transformer = MlentoryTransform(kg_handler, None)
        logger.info("Transformer initialized successfully")
        return transformer
    except Exception as e:
        logger.error(f"Failed to initialize transformer: {str(e)}")
        raise

def initialize_load_processor(kg_files_directory: str, logger: logging.Logger) -> LoadProcessor:
    """
    Initializes the load processor with the configuration data.

    Args:
        kg_files_directory (str): The path to the kg files directory.
        logger (logging.Logger): Logger instance for logging events.

    Returns:
        LoadProcessor: The load processor instance.
    """
    logger.info("Initializing load processor components")

    try:
        logger.info("Connecting to PostgreSQL")
        sqlHandler = SQLHandler(
            host=POSTGRES_HOST,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DB,
        )
        sqlHandler.connect()
        logger.info("Successfully connected to PostgreSQL")

        logger.info("Connecting to Virtuoso")
        rdfHandler = RDFHandler(
            container_name=VIRTUOSO_HOST,
            kg_files_directory=kg_files_directory,
            _user=VIRTUOSO_USER,
            _password=VIRTUOSO_PASSWORD,
            sparql_endpoint=VIRTUOSO_SPARQL_ENDPOINT,
        )
        logger.info("Successfully connected to Virtuoso")

        logger.info("Connecting to Elasticsearch")
        elasticsearchHandler = IndexHandler(
            es_host=ELASTICSEARCH_HOST,
            es_port=ELASTICSEARCH_PORT,
        )
        elasticsearchHandler.initialize_OpenML_index(index_name="openml_models")
        logger.info("Successfully connected to Elasticsearch and initialized index")

        logger.info("Initializing GraphHandlerForKG")
        graphHandler = GraphHandlerForKG(
            SQLHandler=sqlHandler,
            RDFHandler=rdfHandler,
            IndexHandler=elasticsearchHandler,
            kg_files_directory=kg_files_directory,
            graph_identifier="https://w3id.org/mlentory/mlentory_graph/",
            deprecated_graph_identifier="https://w3id.org/mlentory/deprecated_mlentory_graph/",
        )

        logger.info("LoadProcessor initialized successfully")
        return LoadProcessor(
            SQLHandler=sqlHandler,
            RDFHandler=rdfHandler,
            IndexHandler=elasticsearchHandler,
            GraphHandler=graphHandler,
            kg_files_directory=kg_files_directory,
        )

    except Exception as e:
        logger.error(f"Failed to initialize load processor: {str(e)}")
        raise


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
    logger = setup_logging()
    logger.info("Starting ETL process")

    args = parse_args()
    logger.info(f"Parsed arguments: {vars(args)}")

    config_path = "configuration/openml"
    kg_files_directory = "./../kg_files" 
    
    os.makedirs(kg_files_directory, exist_ok=True, mode=0o777)
    os.makedirs(args.output_dir, exist_ok=True, mode=0o777)

    # Extract
    logger.info("Starting extraction phase")
    extractor = initialize_extractor(config_path, logger)

    logger.info("Extracting run info with additional entities")
    start_time = time.time()
    extracted_entities = extractor.extract_run_info_with_additional_entities(
        num_instances=args.num_instances, 
        offset=args.offset,
        threads=args.threads, 
        output_dir=args.output_dir,
        save_result_in_json=args.save_extraction,
        additional_entities=["dataset"],
        )
    end_time = time.time()
    logger.info("Extraction completed successfully")
    logger.info(f"TIME TAKEN FOR EXTRACTION: {end_time - start_time} seconds")
    
    # Transform
    logger.info("Starting transformation phase")
    transformer = initialize_transformer(config_path, logger)

    logger.info("Transforming OpenML runs")
    start_time = time.time()
    runs_kg, runs_extraction_metadata = transformer.transform_OpenML_runs(
        extracted_df=extracted_entities["run"],
        save_output_in_json=True,
        output_dir=args.output_dir+"/runs",
    )
    end_time = time.time()
    logger.info("OpenML runs transformation completed")
    logger.info(f"TIME TAKEN FOR TRANSFORMATION OF RUNS:  {end_time - start_time} seconds")

    logger.info("Transforming OpenML datasets")
    start_time = time.time()
    datasets_kg, datasets_extraction_metadata = transformer.transform_OpenML_datasets(
        extracted_df=extracted_entities["dataset"],
        save_output_in_json=True,
        output_dir=args.output_dir+"/datasets",
    )
    end_time = time.time()
    logger.info("OpenML datasets transformation completed")
    logger.info(f"TIME TAKEN FOR TRANSFORMATION OF DATASETS:  {end_time - start_time} seconds")

    logger.info("Unifying knowledge graphs")
    start_time = time.time()
    kg_integrated = transformer.unify_graphs(
            [runs_kg, datasets_kg],
            save_output_in_json=True,
            output_dir=args.output_dir+"/kg",
        )
    end_time = time.time()
    logger.info("Knowledge graphs unified successfully")
    logger.info(f"TIME TAKEN FOR UNIFYING KNOWLEDGE GRAPHS:  {end_time - start_time} seconds")
    
    logger.info("Unifying extraction metadata")
    start_time = time.time()
    extraction_metadata_integrated = transformer.unify_graphs(
            [runs_extraction_metadata,
            datasets_extraction_metadata],
            save_output_in_json=True,
            output_dir=args.output_dir+"/extraction_metadata",
        )  
    end_time = time.time()
    logger.info("Extraction metadata unified successfully")
    logger.info(f"TIME TAKEN FOR UNIFYING METADATA:  {end_time - start_time} seconds")

    # Load

    logger.info("Starting transformation phase")
    loader = initialize_load_processor(kg_files_directory, logger)

    logger.info("Loading the knowledge graph to database")
    start_time = time.time()
    loader.update_dbs_with_kg(kg_integrated, extraction_metadata_integrated)
    end_time = time.time()
    logger.info("Loading data to db completed successfully")
    logger.info(f"TIME TAKEN FOR LOADING METADATA:  {end_time - start_time} seconds")

if __name__ == "__main__":
    main()