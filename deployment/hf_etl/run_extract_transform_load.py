import shutil
import numpy as np
import time
np.float_ = np.float64
import pandas as pd
from rdflib import Graph
import logging
from datetime import datetime
from typing import List
from tqdm import tqdm
import os
import argparse
import re

from mlentory_extract.hf_extract import HFDatasetManager
from mlentory_extract.hf_extract import HFExtractor
from mlentory_extract.core import ModelCardToSchemaParser
from mlentory_load.core import LoadProcessor, GraphHandlerForKG
from mlentory_load.dbHandler import RDFHandler, SQLHandler, IndexHandler
from mlentory_transform.core import (
    MlentoryTransform,
    KnowledgeGraphHandler,
    MlentoryTransformWithGraphBuilder,
)
# from hf_etl_component import HuggingFaceETLComponent


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

def load_tsv_file_to_df(path: str) -> pd.DataFrame:
    """Loads data from a TSV file into a Pandas DataFrame."""
    return pd.read_csv(path, sep="\t")

def load_models_from_file(file_path: str) -> List[str]:
    """
    Loads model IDs from a text file, extracting 'owner/repo_name' format.

    Args:
        file_path (str): The path to the text file containing model IDs or URLs.

    Returns:
        List[str]: A list of model IDs in 'owner/repo_name' format.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If a line in the file does not contain a valid model ID format.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model list file not found: {file_path}")

    model_ids = []
    
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): # Skip empty lines and comments
                continue
            
            id_match = line.split("/")[-1]

            if id_match:
                model_ids.append(id_match)
            else:
                logging.warning(f"Skipping invalid line in model list file: {line}")
                # Or raise ValueError("Invalid model format found in file: {line}")

    logging.info(f"Loaded {len(model_ids)} model IDs from {file_path}")
    return model_ids

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


def initialize_extractor(config_path: str) -> HFExtractor:
    """
    Initializes the extractor with the configuration data.

    Args:
        config_path (str): The path to the configuration data.

    Returns:
        HFExtractor: The extractor instance.
    """
    tags_language = load_tsv_file_to_list(f"{config_path}/extract/tags_language.tsv")
    
    tags_libraries = load_tsv_file_to_df(f"{config_path}/extract/tags_libraries.tsv")
    tags_other = load_tsv_file_to_df(f"{config_path}/extract/tags_other.tsv")
    tags_task = load_tsv_file_to_df(f"{config_path}/extract/tags_task.tsv")
    
    dataset_manager = HFDatasetManager(api_token=os.getenv("HF_TOKEN"))
    
    parser = ModelCardToSchemaParser(
        # qa_model="sentence-transformers/all-MiniLM-L6-v2",
        # qa_model="BAAI/bge-m3",
        qa_model_name="Qwen/Qwen3-1.7B",
        matching_model_name="Alibaba-NLP/gte-base-en-v1.5",
        # matching_model_name="intfloat/multilingual-e5-large-instruct",
        schema_file=f"{config_path}/transform/FAIR4ML_schema.tsv",
        tags_language=tags_language,
        tags_libraries=tags_libraries,
        tags_other=tags_other,
        tags_task=tags_task,
    )
    
    return HFExtractor(
        dataset_manager=dataset_manager,
        parser=parser
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

    transformer = MlentoryTransformWithGraphBuilder(base_namespace="https://w3id.org/mlentory/mlentory_graph/", FAIR4ML_schema_data=new_schema)

    return transformer


def initialize_load_processor(
    kg_files_directory: str, logger: logging.Logger, remote_db: bool = False
) -> LoadProcessor:
    """
    Initializes the load processor with the configuration data.

    Args:
        kg_files_directory (str): The path to the kg files directory.
        logger (logging.Logger): The logger instance.

    Returns:
        LoadProcessor: The load processor instance.
    """
    # Get database configuration from environment variables
    postgres_host = os.getenv("POSTGRES_HOST", "postgres_db")
    postgres_user = os.getenv("POSTGRES_USER", "user") 
    postgres_password = os.getenv("POSTGRES_PASSWORD", "password")
    postgres_db = os.getenv("POSTGRES_DB", "history_DB")
    
    virtuoso_host = os.getenv("VIRTUOSO_HOST", "virtuoso_db")
    virtuoso_http_port = os.getenv("VIRTUOSO_HTTP_PORT", "8890")
    virtuoso_password = os.getenv("VIRTUOSO_DBA_PASSWORD", "my_strong_password")
    
    elasticsearch_host = os.getenv("ELASTICSEARCH_HOST", "elastic_db")
    elasticsearch_port = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
    
    # remote_api_base_url = os.getenv("REMOTE_API_BASE_URL", "http://10.0.7.249:8000")
    remote_api_base_url = os.getenv("REMOTE_API_BASE_URL", "http://backend:8000")
    
    print(f"postgres_host: {postgres_host}")
    print(f"postgres_user: {postgres_user}")
    print(f"postgres_password: {postgres_password}")
    print(f"postgres_db: {postgres_db}")
    print(f"virtuoso_host: {virtuoso_host}")
    print(f"virtuoso_http_port: {virtuoso_http_port}")
    print(f"virtuoso_password: {virtuoso_password}")
    print(f"elasticsearch_host: {elasticsearch_host}")
    print(f"elasticsearch_port: {elasticsearch_port}")
    print(f"remote_api_base_url: {remote_api_base_url}")
    
    
    sqlHandler = None
    rdfHandler = None
    elasticsearchHandler = None
    
    if not remote_db:
        sqlHandler = SQLHandler(
            host=postgres_host,
            user=postgres_user,
            password=postgres_password,
            database=postgres_db,
        )
        sqlHandler.connect()

        rdfHandler = RDFHandler(
            container_name=virtuoso_host,
            kg_files_directory=kg_files_directory,
            _user="dba",
            _password=virtuoso_password,
            sparql_endpoint=f"http://{virtuoso_host}:{virtuoso_http_port}/sparql",
        )

        elasticsearchHandler = IndexHandler(
            es_host=elasticsearch_host,
            es_port=elasticsearch_port,
        )

         # elasticsearchHandler.clean_indices()

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
        remote_api_base_url=remote_api_base_url,
    )

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

def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    
    Usage modes:
        1. Full ETL: Extract, transform, and load new data
        2. Dummy data: Load pre-existing dummy data for testing
        3. File loading: Load existing KG and metadata files directly
    """
    parser = argparse.ArgumentParser(
        description="HuggingFace ETL Process",
        epilog="""
Usage examples:
  # Full ETL with 50 models
  %(prog)s --num-models 50 --remote-db True

  # Full ETL with pagination (skip first 100 models)
  %(prog)s --num-models 50 --offset 100 --remote-db True

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
        default=False,
        help="Save the results of the transformation phase",
    )
    parser.add_argument(
        "--from-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=datetime(2000, 1, 1),
        help="Download models from this date (format: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--num-models", "-nm", type=int, default=20, help="Number of models to download"
    )
    parser.add_argument(
        "--offset", "-o", type=int, default=0, help="Offset for pagination when downloading models (only used when not updating recent models)"
    )
    parser.add_argument(
        "--num-datasets", "-nd", type=int, default=20, help="Number of datasets to download"
    )
    parser.add_argument(
        "--output-dir",
        default="./hf_etl/outputs/files",
        help="Directory to save intermediate results",
    )
    parser.add_argument(
        "--model-list-file",
        "-mlf",
        type=str,
        # default="./hf_etl/inputs/models_to_download.txt",
        help="Path to a text file containing a list of Hugging Face model IDs (one per line). If provided, --num-models and --from-date are ignored.",
    )
    parser.add_argument(
        "--use-dummy-data",
        "-ud",
        # action="store_true",
        default=False,
        help="Use dummy data for testing purposes.",
    )
    parser.add_argument(
        "--unstructured-text-strategy",
        "-uts",
        type=str,
        default="None",
        help="Strategy to use for unstructured text extraction.",
    )
    
    parser.add_argument(
        "--remote-db",
        "-rd",
        # action="store_true",
        default=False,
        help="Use remote databases for loading data.",
    )
    
    parser.add_argument(
        "--chunking",
        default=False,
        help="Wheter or not to chunk the data for the uploading step"
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
        default=800,
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


def main():
    args = parse_args()
    logger = setup_logging()

    # Validate argument combinations
    file_loading_mode = args.kg_file_path or args.metadata_file_path
    if file_loading_mode and args.use_dummy_data:
        logger.error("Cannot use --kg-file-path/--metadata-file-path with --use-dummy-data. Choose one mode.")
        return
    if file_loading_mode and args.model_list_file:
        logger.warning("--model-list-file is ignored when loading from existing KG files")

    # Setup configuration data
    config_path = "./configuration/hf"  # Path to configuration folder
    kg_files_directory = "./../kg_files"  # Path to kg files directory
    intialize_folder_structure(args.output_dir,clean_folders=False)
    
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

        # Initialize extractor
        logger.info("Initializing extractor...")
        start_time = time.time()
        extractor = initialize_extractor(config_path)
        end_time = time.time()
        logger.info(f"Extractor initialization took {end_time - start_time:.2f} seconds")
        
        extracted_entities = {}
        models_df = pd.DataFrame()

        if args.model_list_file:
            logger.info(f"Processing models from file: {args.model_list_file}")
            try:
                model_ids_from_file = load_models_from_file(args.model_list_file)
            except FileNotFoundError as e:
                logging.error(e)
                return # Exit if file not found

            if not model_ids_from_file:
                logging.warning("Model list file is empty or contains no valid IDs. Skipping extraction.")
                extracted_entities = {"models": pd.DataFrame()} # Ensure models key exists
            else:
                # Call the new method in HFExtractor
                entities_to_download_config = ["datasets", "articles", "keywords", "base_models", "licenses"]
                
                logger.info("Starting model extraction from file...")
                start_time = time.time()
                extracted_entities = extractor.download_specific_models_with_related_entities(
                    model_ids=model_ids_from_file,
                    output_dir=args.output_dir, # Pass the base output_dir
                    save_result_in_json=args.save_extraction,
                    threads=4, # TODO: Consider making threads an arg
                    related_entities_to_download=entities_to_download_config,
                    unstructured_text_strategy=args.unstructured_text_strategy,
                )
                end_time = time.time()
                logger.info(f"Model extraction from file took {end_time - start_time:.2f} seconds")
                
                if "models" not in extracted_entities or extracted_entities["models"].empty:
                    logging.warning("No models were extracted using the model list file.")

        else:
            # Existing logic: download models and related entities based on num_models/date
            logging.info(f"Downloading {args.num_models} models, last modified after {args.from_date.strftime('%Y-%m-%d')}")
            entities_to_download_config = ["datasets", "articles", "keywords", "base_models", "licenses"]
            
            logger.info("Starting model extraction with default parameters...")
            start_time = time.time()
            extracted_entities = extractor.download_models_with_related_entities(
                num_models=args.num_models,
                from_date=args.from_date, # Use the parsed date
                output_dir=args.output_dir, # Use the base output dir
                save_initial_data=False, # Controlled by save_extraction?
                save_result_in_json=args.save_extraction, # Reuse flag
                update_recent=False, # Default behavior
                related_entities_to_download=entities_to_download_config,
                unstructured_text_strategy=args.unstructured_text_strategy,
                threads=4, # Reuse threads
                depth=2, # Default behavior
                offset=args.offset, # Use the offset argument
            )
            end_time = time.time()
            logger.info(f"Model extraction with default parameters took {end_time - start_time:.2f} seconds")
            
            # 'models' key in extracted_entities already contains the combined models here
            if "models" not in extracted_entities or extracted_entities["models"].empty:
                logging.warning("No models were extracted using the default method. Check parameters or HF connection.")
                # Decide if to exit or continue without models
                # return

        # Initialize transformer (outside the if/else)
        logger.info("Initializing transformer...")
        start_time = time.time()
        transformer = initialize_transform_hf(config_path)
        end_time = time.time()
        logger.info(f"Transformer initialization took {end_time - start_time:.2f} seconds")

        logger.info("Starting transformation process...")
        start_time = time.time()
        kg_integrated, extraction_metadata_integrated = transformer.transform_HF_models_with_related_entities(
            extracted_entities=extracted_entities,
            save_output=True,
            kg_output_dir=args.output_dir+"/kg",
            extraction_metadata_output_dir=args.output_dir+"/extraction_metadata",
        )
        end_time = time.time()
        logger.info(f"Transformation process took {end_time - start_time:.2f} seconds")
    else:
        # load kg with rdflib
        logger.info("Loading dummy KG TTL file...")
        start_time = time.time()
        # kg_integrated.parse(
        #     args.output_dir
        #     + "/../../copy_examples/files/kg/example_HF_models_kg.nt",
        #     format="nt",
        # )
        kg_integrated.parse(
            args.output_dir
            + "/../files/kg/2025-08-12_05-01-12_unified_kg.nt",
            format="turtle",
        )
        end_time = time.time()
        logger.info(
            f"Loading dummy KG TTL file took {end_time - start_time:.2f} seconds"
        )

        logger.info("Loading dummy extraction metadata TTL file...")
        start_time = time.time()
        # extraction_metadata_integrated.parse(
        #     args.output_dir
        #     + "/../../copy_examples/files/extraction_metadata/example_HF_models_extraction_metadata_kg.nt",
        #     format="nt",
        # )
        extraction_metadata_integrated.parse(
            args.output_dir
            + "/../files/extraction_metadata/2025-08-12_08-57-58_unified_kg.nt",
            format="turtle",
        )
        end_time = time.time()
        logger.info(
            f"Loading dummy extraction metadata TTL file took {end_time - start_time:.2f} seconds"
        )

    # Initialize loader
    logger.info("Initializing loader...")
    start_time = time.time()
    loader = initialize_load_processor(kg_files_directory, logger, remote_db=args.remote_db)
    end_time = time.time()
    logger.info(f"Loader initialization took {end_time - start_time:.2f} seconds")
    
    # Set upload timeout if using remote database
    if args.remote_db:
        loader.set_upload_timeout(args.upload_timeout)
        logger.info(f"Set upload timeout to {args.upload_timeout} seconds for remote database uploads")

    logger.info("Cleaning databases...")
    start_time = time.time()
    # loader.clean_DBs()
    # time.sleep(50)
    # loader = initialize_load_processor(kg_files_directory, logger)
    end_time = time.time()
    logger.info(f"Database cleaning took {end_time - start_time:.2f} seconds")

    # Load data
    logger.info("Starting database update with KG...")
    start_time = time.time()
    if args.chunking is True:
        logger.info(f"Using chunk size: {args.chunk_size}")
        loader.update_dbs_with_kg(kg_integrated,
                              extraction_metadata_integrated,
                              extraction_name="hf_extraction",
                              remote_db=args.remote_db,
                              kg_chunks_size=args.chunk_size,
                              save_load_output=True,
                              load_output_dir=args.output_dir+"/chunks")
    else:
        loader.update_dbs_with_kg(kg_integrated,
                              extraction_metadata_integrated,
                              extraction_name="hf_extraction",
                              remote_db=args.remote_db,
                              kg_chunks_size=0,
                              save_load_output=True,
                              load_output_dir=args.output_dir+"/chunks")
    end_time = time.time()
    logger.info(f"Database update with KG took {end_time - start_time:.2f} seconds")

    # """Main entry point for HuggingFace ETL process."""
    # etl = HuggingFaceETLComponent()
    # etl.run()


if __name__ == "__main__":
    main()
