import shutil
import pandas as pd
from rdflib import Graph
import logging
import datetime
from typing import List
from tqdm import tqdm
import os
import argparse
import time
from datetime import datetime
from mlentory_extract.ai4life_extract import AI4LifeExtractor
from mlentory_load.core import LoadProcessor, GraphHandlerForKG
from mlentory_load.dbHandler import RDFHandler, SQLHandler, IndexHandler
from mlentory_transform.core import (
    MlentoryTransform,
    MlentoryTransformWithGraphBuilder,
)
# from ai4life_etl_component import AI4LifeETLComponent

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

def initialize_transform(config_path: str) -> MlentoryTransformWithGraphBuilder:
    """
    Initializes the transformer with the configuration data.

    Args:
        config_path (str): The path to the configuration data.

    Returns:
        MlentoryTransformWithGraphBuilder: The transformer instance.
    """
    new_schema = pd.read_csv(
        f"{config_path}/transform/FAIR4ML_schema.csv", sep=",", lineterminator="\n"
    )

    transformer = MlentoryTransformWithGraphBuilder(base_namespace="https://w3id.org/mlentory/mlentory_graph/", FAIR4ML_schema_data=new_schema)

    return transformer

def intialize_folder_structure(output_dir: str, clean_folders: bool = False) -> None:
    """
    Initializes the folder structure.
    """
    if clean_folders:
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+"/models", exist_ok=True)
    os.makedirs(output_dir+"/datasets", exist_ok=True)
    os.makedirs(output_dir+"/articles", exist_ok=True)
    os.makedirs(output_dir+"/kg", exist_ok=True)
    os.makedirs(output_dir+"/extraction_metadata", exist_ok=True)
    os.makedirs(output_dir+"/chunks", exist_ok=True)
    
def initialize_load_processor(
    kg_files_directory: str, logger: logging.Logger
) -> LoadProcessor:
    """
    Initializes the load processor with the configuration data.

    Args:
        kg_files_directory (str): The path to the kg files directory.
        logger (logging.Logger): The logger instance.

    Returns:
        LoadProcessor: The load processor instance.
    """
    sqlHandler = SQLHandler(
        host=POSTGRES_HOST,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=POSTGRES_DB,
    )
    sqlHandler.connect()

    rdfHandler = RDFHandler(
        container_name=VIRTUOSO_HOST,  # Assuming container name is the same as the host
        kg_files_directory=kg_files_directory,
        _user=VIRTUOSO_USER,
        _password=VIRTUOSO_PASSWORD,
        sparql_endpoint=VIRTUOSO_SPARQL_ENDPOINT,
    )

    elasticsearchHandler = IndexHandler(
        es_host=ELASTICSEARCH_HOST,
        es_port=ELASTICSEARCH_PORT,
    )

    # Initialize all indices to prevent conflicts between different data sources
    elasticsearchHandler.initialize_HF_index(index_name="hf_models")
    elasticsearchHandler.initialize_OpenML_index(index_name="openml_models")
    elasticsearchHandler.initialize_AI4Life_index(index_name="ai4life_models")

    # Initializing the graph creator
    graphHandler = GraphHandlerForKG(
        SQLHandler=sqlHandler,
        RDFHandler=rdfHandler,
        IndexHandler=elasticsearchHandler,
        kg_files_directory=kg_files_directory,
        graph_identifier="https://w3id.org/mlentory/mlentory_graph",
        deprecated_graph_identifier="https://w3id.org/mlentory/deprecated_mlentory_graph",
        logger=logger,
    )

    # Initializing the load processor
    return LoadProcessor(
        SQLHandler=sqlHandler,
        RDFHandler=rdfHandler,
        IndexHandler=elasticsearchHandler,
        GraphHandler=graphHandler,
        kg_files_directory=kg_files_directory,
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
        "--num-models", type=int, default=1000, help="Number of models to download"
    )
    parser.add_argument(
        "--output-dir",
        default="./ai4life_etl/outputs/files",
        help="Directory to save intermediate results",
    )
    parser.add_argument(
        "--load-extraction-and-transform-data",
        "-led",
        # action="store_true",
        default=False,
        help="Load the extraction data into the database",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()
    intialize_folder_structure(args.output_dir,clean_folders=False)

    # Setup configuration data
    config_path = "./configuration/ai4life"  # Path to configuration folder
    kg_integrated, kg_metadata_integrated = Graph(), Graph()

    if args.load_extraction_and_transform_data:
        kg_integrated.parse(f"./{args.output_dir}/kg/2025-08-15_12-40-32_unified_kg.nt", format="nt")
        kg_metadata_integrated.parse(f"./{args.output_dir}/extraction_metadata/2025-08-15_12-40-33_unified_kg.nt", format="nt")
    else:
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
            kg_output_dir=args.output_dir,
        )
        end_time = time.time()
        logger.info(f"Transformation process took {end_time - start_time:.2f} seconds")

    # Initialize loader
    logger.info("Initializing loader...")
    start_time = time.time()
    loader = initialize_load_processor(args.output_dir+"/kg", logger)
    end_time = time.time()
    logger.info(f"Loader initialization took {end_time - start_time:.2f} seconds")

    logger.info("Cleaning databases...")
    start_time = time.time()
    # loader.clean_DBs()
    end_time = time.time()
    logger.info(f"Database cleaning took {end_time - start_time:.2f} seconds")

    # Load data
    logger.info("Starting database update with KG...")
    start_time = time.time()
    loader.update_dbs_with_kg(kg=kg_integrated, extraction_metadata=kg_metadata_integrated, remote_db=False, save_load_output=True, extraction_name="ai4life_models",kg_chunks_size=0,
                              load_output_dir=args.output_dir+"/chunks")
    end_time = time.time()
    logger.info(f"Database update with KG took {end_time - start_time:.2f} seconds")
    
    # """Main entry point for AI4Life ETL process."""
    # etl = AI4LifeETLComponent()
    # etl.run()

    
if __name__ == "__main__":
    main()