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

# Load environment variables with defaults
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres_db")
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
    return pd.read_csv(path, sep="\t")


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
    tags_language = load_tsv_file_to_list(f"{config_path}/extract/tags_language.tsv")
    
    tags_libraries = load_tsv_file_to_df(f"{config_path}/extract/tags_libraries.tsv")
    tags_other = load_tsv_file_to_df(f"{config_path}/extract/tags_other.tsv")
    tags_task = load_tsv_file_to_df(f"{config_path}/extract/tags_task.tsv")
    
    dataset_manager = HFDatasetManager(api_token=os.getenv("HF_TOKEN"))
    
    parser = ModelCardToSchemaParser(
        # qa_model="sentence-transformers/all-MiniLM-L6-v2",
        # qa_model="BAAI/bge-m3",
        qa_model_name="Qwen/Qwen2.5-3B",
        matching_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
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

    transformer = MlentoryTransformWithGraphBuilder(base_namespace="http://mlentory.de/mlentory_graph/", FAIR4ML_schema_data=new_schema)

    return transformer


def initialize_load_processor(kg_files_directory: str) -> LoadProcessor:
    """
    Initializes the load processor with the configuration data.

    Args:
        kg_files_directory (str): The path to the kg files directory.

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

    elasticsearchHandler.initialize_HF_index(index_name="hf_models")

    # Initializing the graph creator
    graphHandler = GraphHandlerForKG(
        SQLHandler=sqlHandler,
        RDFHandler=rdfHandler,
        IndexHandler=elasticsearchHandler,
        kg_files_directory=kg_files_directory,
        graph_identifier="http://mlentory.de/mlentory_graph",
        deprecated_graph_identifier="http://mlentory.de/deprecated_mlentory_graph",
    )

    # Initializing the load processor
    return LoadProcessor(
        SQLHandler=sqlHandler,
        RDFHandler=rdfHandler,
        IndexHandler=elasticsearchHandler,
        GraphHandler=graphHandler,
        kg_files_directory=kg_files_directory,
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
        "--num-datasets", "-nd", type=int, default=20, help="Number of datasets to download"
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
    kg_files_directory = "./../kg_files"  # Path to kg files directory
    intialize_folder_structure(args.output_dir,clean_folders=False)
    
    use_dummy_data = False
    kg_integrated = Graph()  
    extraction_metadata_integrated = Graph()
    
    if not use_dummy_data:

        # Initialize extractor
        extractor = initialize_extractor(config_path)

        extracted_entities = extractor.download_models_with_related_entities(
            num_models=args.num_models,
            from_date=datetime(2023, 1, 1),
            output_dir=args.output_dir+"/models",
            save_result_in_json=True,
            save_initial_data=False,
            update_recent=False,
            related_entities_to_download=["datasets", "articles", "base_models", "keywords"],
            threads=4,
            depth=2,
        )
        
        # Initialize transformer
        transformer = initialize_transform_hf(config_path)

        kg_integrated, extraction_metadata_integrated = transformer.transform_HF_models_with_related_entities(
            extracted_entities=extracted_entities,
            save_output=True,
            kg_output_dir=args.output_dir+"/kg",
            extraction_metadata_output_dir=args.output_dir+"/extraction_metadata",
        )
    else:
        # load kg with rdflib   
        kg_integrated.parse(args.output_dir + "/kg/2025-02-24_05-23-35_unified_kg.ttl", format="turtle")
        extraction_metadata_integrated.parse(args.output_dir + "/extraction_metadata/2025-02-24_05-24-15_unified_kg.ttl", format="turtle")

    # Initialize loader
    loader = initialize_load_processor(kg_files_directory)

    # loader.clean_DBs()

    # Load data
    loader.update_dbs_with_kg(kg_integrated, extraction_metadata_integrated)


if __name__ == "__main__":
    main()
