import os
import shutil
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
# from openml_etl_component import OpenMLETLComponent

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

# REMOTE_API_BASE_URL = os.getenv("REMOTE_API_BASE_URL", "http://10.0.7.249:8000")
REMOTE_API_BASE_URL = os.getenv("REMOTE_API_BASE_URL", "http://backend:8000")


def load_tsv_file_to_list(path: str) -> List[str]:
    """
    Loads a TSV file and returns the first column as a list.

    Args:
        path (str): Path to the TSV file.

    Returns:
        List[str]: List of values from the first column.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        pd.errors.EmptyDataError: If the file is empty.
    """
    return [val[0] for val in pd.read_csv(path, sep="\t").values.tolist()]

def setup_logging() -> logging.Logger:
    """
    Sets up the logging system with both file and console output.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        >>> logger = setup_logging()
        >>> logger.info("Pipeline started")
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

    Raises:
        Exception: If extractor initialization fails.

    Example:
        >>> extractor = initialize_extractor("./config", logger)
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

    Raises:
        Exception: If transformer initialization fails.

    Example:
        >>> transformer = initialize_transformer("./config", logger)
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

    Raises:
        Exception: If load processor initialization fails.

    Example:
        >>> loader = initialize_load_processor("./kg_files", logger)
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
        # Initialize all indices to prevent conflicts between different data sources
        elasticsearchHandler.initialize_HF_index(index_name="hf_models")
        elasticsearchHandler.initialize_OpenML_index(index_name="openml_models")
        elasticsearchHandler.initialize_AI4Life_index(index_name="ai4life_models")
        logger.info("Successfully connected to Elasticsearch and initialized all indices")

        logger.info("Initializing GraphHandlerForKG")
        graphHandler = GraphHandlerForKG(
            SQLHandler=sqlHandler,
            RDFHandler=rdfHandler,
            IndexHandler=elasticsearchHandler,
            kg_files_directory=kg_files_directory,
            graph_identifier="https://w3id.org/mlentory/mlentory_graph",
            deprecated_graph_identifier="https://w3id.org/mlentory/deprecated_mlentory_graph",
        )

        logger.info("LoadProcessor initialized successfully")
        return LoadProcessor(
            SQLHandler=sqlHandler,
            RDFHandler=rdfHandler,
            IndexHandler=elasticsearchHandler,
            GraphHandler=graphHandler,
            kg_files_directory=kg_files_directory,
            remote_api_base_url=REMOTE_API_BASE_URL,
        )

    except Exception as e:
        logger.error(f"Failed to initialize load processor: {str(e)}")
        raise


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.

    Usage modes:
        1. Full ETL: Extract, transform, and load new data
        2. Dummy data: Load pre-existing dummy data for testing
        3. File loading: Load existing KG and metadata files directly

    Example:
        >>> args = parse_args()
        >>> print(args.num_instances)
    """
    parser = argparse.ArgumentParser(
        description="OpenML ETL Process",
        epilog="""
Usage examples:
  # Full ETL with 50 instances
  %(prog)s --num-instances 50 --remote-db True
  
  # Load from existing files
  %(prog)s --kg-file-path ./kg.nt --metadata-file-path ./metadata.nt --remote-db True
  
  # Use dummy data for testing
  %(prog)s --use-dummy-data True --chunk-size 500
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--save-extraction",
        action="store_true",
        default=False,
        help="Save the results of the extraction phase",
    )
    
    parser.add_argument(
        "--save-transformation", "-st",
        action="store_true",
        default=True,
        help="Save the results of the transformation phase",
    )
    
    parser.add_argument(
        "--num-instances","-nm", type=int, default=20, help="Number of instances to extract metadata from"
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
    
    parser.add_argument(
        "--use-dummy-data",
        "-ud",
        default=False,
        help="Use dummy data for testing purposes.",
    )
    
    parser.add_argument(
        "--remote-db",
        "-rd",
        default=False,
        help="Use remote databases for loading data.",
    )
    
    parser.add_argument(
        "--chunking",
        default=False,
        help="Whether or not to chunk the data for the uploading step"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of chunks when uploading to remote database (default: 1000, use smaller values for slower connections)"
    )
    
    parser.add_argument(
        "--upload-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for HTTP uploads to remote database (default: 600 seconds/10 minutes)"
    )
    
    parser.add_argument(
        "--kg-file-path",
        type=str,
        help="Path to an existing KG file .nt to load directly (skips extraction and transformation). Must be used with --metadata-file-path."
    )
    
    parser.add_argument(
        "--metadata-file-path", 
        type=str,
        help="Path to an existing metadata file .nt to load directly (skips extraction and transformation). Must be used with --kg-file-path."
    )
    
    return parser.parse_args()

def intialize_folder_structure(output_dir: str, clean_folders: bool = False) -> None:
    """
    Initializes the folder structure for the ETL pipeline.

    Args:
        output_dir (str): Base output directory.
        clean_folders (bool): Whether to clean existing folders.

    Example:
        >>> intialize_folder_structure("./output", clean_folders=True)
    """
    if clean_folders:
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+"/runs", exist_ok=True)
    os.makedirs(output_dir+"/datasets", exist_ok=True)
    os.makedirs(output_dir+"/kg", exist_ok=True)
    os.makedirs(output_dir+"/extraction_metadata", exist_ok=True)
    os.makedirs(output_dir+"/chunks", exist_ok=True)
    
def main():
    """
    Main function that orchestrates the OpenML ETL process.

    The function supports multiple modes:
    - Full ETL: Extract, transform, and load data
    - File loading: Load from existing KG files
    - Dummy data: Use pre-existing test data

    Raises:
        SystemExit: If invalid argument combinations are provided.

    Example:
        >>> main()
    """
    args = parse_args()
    logger = setup_logging()
    logger.info("Starting ETL process")

    # Validate argument combinations
    file_loading_mode = args.kg_file_path or args.metadata_file_path
    if file_loading_mode and args.use_dummy_data:
        logger.error("Cannot use --kg-file-path/--metadata-file-path with --use-dummy-data. Choose one mode.")
        return

    logger.info(f"Parsed arguments: {vars(args)}")

    config_path = "configuration/openml"
    kg_files_directory = "./../kg_files" 
    
    os.makedirs(kg_files_directory, exist_ok=True, mode=0o777)
    intialize_folder_structure(args.output_dir, clean_folders=False)

    kg_integrated = Graph()
    extraction_metadata_integrated = Graph()

    # Check if loading from existing files
    if args.kg_file_path or args.metadata_file_path:
        if args.kg_file_path and args.metadata_file_path:
            logger.info(f"Loading KG from file: {args.kg_file_path}")
            logger.info(f"Loading metadata from file: {args.metadata_file_path}")
            
            # Validate file existence
            if not os.path.exists(args.kg_file_path):
                logger.error(f"KG file not found: {args.kg_file_path}")
                return
            if not os.path.exists(args.metadata_file_path):
                logger.error(f"Metadata file not found: {args.metadata_file_path}")
                return
            
            start_time = time.time()
            
            # Determine format based on file extension
            # kg_format = "turtle" if args.kg_file_path.endswith(('.ttl', '.turtle')) else "nt"
            # metadata_format = "turtle" if args.metadata_file_path.endswith(('.ttl', '.turtle')) else "nt"
            
            kg_integrated.parse(args.kg_file_path, format="nt")
            extraction_metadata_integrated.parse(args.metadata_file_path, format="nt")
            
            end_time = time.time()
            logger.info(f"Loading files took {end_time - start_time:.2f} seconds")
            logger.info(f"KG loaded with {len(kg_integrated)} triples")
            logger.info(f"Metadata loaded with {len(extraction_metadata_integrated)} triples")
            
        else:
            logger.error("Both --kg-file-path and --metadata-file-path must be provided together")
            return

    elif args.use_dummy_data is False:
        # Full ETL process
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
            save_output_in_json=args.save_transformation,
            output_dir=args.output_dir+"/runs",
        )
        end_time = time.time()
        logger.info("OpenML runs transformation completed")
        logger.info(f"TIME TAKEN FOR TRANSFORMATION OF RUNS:  {end_time - start_time} seconds")

        logger.info("Transforming OpenML datasets")
        start_time = time.time()
        datasets_kg, datasets_extraction_metadata = transformer.transform_OpenML_datasets(
            extracted_df=extracted_entities["dataset"],
            save_output_in_json=args.save_transformation,
            output_dir=args.output_dir+"/datasets",
        )
        end_time = time.time()
        logger.info("OpenML datasets transformation completed")
        logger.info(f"TIME TAKEN FOR TRANSFORMATION OF DATASETS:  {end_time - start_time} seconds")

        logger.info("Unifying knowledge graphs")
        start_time = time.time()
        kg_integrated = transformer.unify_graphs(
                [runs_kg, datasets_kg],
                save_output_in_json=args.save_transformation,
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
                save_output_in_json=args.save_transformation,
                output_dir=args.output_dir+"/extraction_metadata",
            )  
        end_time = time.time()
        logger.info("Extraction metadata unified successfully")
        logger.info(f"TIME TAKEN FOR UNIFYING METADATA:  {end_time - start_time} seconds")

    else:
        # Load dummy data
        logger.info("Loading dummy KG file...")
        start_time = time.time()
        kg_integrated.parse(
            args.output_dir + "/../files/kg/example_openml_kg.nt",
            format="nt",
        )
        end_time = time.time()
        logger.info(f"Loading dummy KG file took {end_time - start_time:.2f} seconds")

        logger.info("Loading dummy extraction metadata file...")
        start_time = time.time()
        extraction_metadata_integrated.parse(
            args.output_dir + "/../files/extraction_metadata/example_openml_metadata.nt",
            format="nt",
        )
        end_time = time.time()
        logger.info(f"Loading dummy extraction metadata file took {end_time - start_time:.2f} seconds")

    # Load
    logger.info("Starting loading phase")
    loader = initialize_load_processor(kg_files_directory, logger)
    
    # Set upload timeout if using remote database
    if args.remote_db:
        loader.set_upload_timeout(args.upload_timeout)
        logger.info(f"Set upload timeout to {args.upload_timeout} seconds for remote database uploads")
    
    logger.info("Cleaning databases...")
    start_time = time.time()
    # loader.clean_DBs()
    # loader = initialize_load_processor(args.output_dir+"/kg", logger)
    end_time = time.time()
    logger.info(f"Database cleaning took {end_time - start_time:.2f} seconds")

    logger.info("Loading the knowledge graph to database")
    start_time = time.time()
    if args.chunking is True:
        logger.info(f"Using chunk size: {args.chunk_size}")
        loader.update_dbs_with_kg(kg_integrated,
                              extraction_metadata_integrated,
                              extraction_name="openml_extraction",
                              remote_db=args.remote_db,
                              kg_chunks_size=args.chunk_size,
                              save_load_output=True,
                              load_output_dir=args.output_dir+"/chunks")
    else:
        loader.update_dbs_with_kg(kg_integrated,
                              extraction_metadata_integrated,
                              extraction_name="openml_extraction",
                              remote_db=args.remote_db,
                              kg_chunks_size=0,
                              save_load_output=True,
                              load_output_dir=args.output_dir+"/chunks")
    end_time = time.time()
    logger.info("Loading data to db completed successfully")
    logger.info(f"TIME TAKEN FOR LOADING METADATA:  {end_time - start_time} seconds")
    
    """Main entry point for OpenML ETL process."""
    # etl = OpenMLETLComponent()
    # etl.run()

if __name__ == "__main__":
    main()