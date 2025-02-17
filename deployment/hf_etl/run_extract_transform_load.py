import numpy as np

np.float_ = np.float64
import pandas as pd
import logging
from datetime import datetime
from typing import List
from tqdm import tqdm
import os
import argparse

from mlentory_extract.hf_extract import HFDatasetManager
from mlentory_extract.hf_extract import HFExtractor
from mlentory_transform.hf_transform.FieldProcessorHF import FieldProcessorHF
from mlentory_load.core import LoadProcessor, GraphHandler
from mlentory_load.dbHandler import RDFHandler, SQLHandler, IndexHandler
from mlentory_transform.hf_transform.TransformHF import TransformHF
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
    questions = load_tsv_file_to_list(f"{config_path}/extract/questions.tsv")
    tags_language = load_tsv_file_to_list(f"{config_path}/extract/tags_language.tsv")
    tags_libraries = load_tsv_file_to_list(f"{config_path}/extract/tags_libraries.tsv")
    tags_other = load_tsv_file_to_list(f"{config_path}/extract/tags_other.tsv")
    tags_task = load_tsv_file_to_list(f"{config_path}/extract/tags_task.tsv")

    dataset_manager = HFDatasetManager(api_token=os.getenv("HF_TOKEN"))

    return HFExtractor(
        qa_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dataset_manager=dataset_manager,
        questions=questions,
        tags_language=tags_language,
        tags_libraries=tags_libraries,
        tags_other=tags_other,
        tags_task=tags_task,
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
        f"{config_path}/transform/M4ML_schema.csv", sep=",", lineterminator="\n"
    )
    transformations = pd.read_csv(
        f"{config_path}/transform/column_transformations.csv",
        lineterminator="\n",
        sep=",",
    )

    transform_hf = TransformHF(new_schema, transformations)

    kg_handler = KnowledgeGraphHandler(
        M4ML_schema=new_schema, base_namespace="http://mlentory.com/mlentory_graph/"
    )

    transformer = MlentoryTransform(kg_handler, transform_hf)

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
        host="postgres",
        user="user",
        password="password",
        database="history_DB",
    )
    sqlHandler.connect()

    rdfHandler = RDFHandler(
        container_name="virtuoso",
        kg_files_directory=kg_files_directory,
        _user="dba",
        _password="my_strong_password",
        sparql_endpoint="http://virtuoso:8890/sparql",
    )

    elasticsearchHandler = IndexHandler(
        es_host="elastic",
        es_port=9200,
    )

    elasticsearchHandler.initialize_HF_index(index_name="hf_models")

    # Initializing the graph creator
    graphHandler = GraphHandler(
        SQLHandler=sqlHandler,
        RDFHandler=rdfHandler,
        IndexHandler=elasticsearchHandler,
        kg_files_directory=kg_files_directory,
        graph_identifier="http://mlentory.com/mlentory_graph",
        deprecated_graph_identifier="http://mlentory.com/deprecated_mlentory_graph",
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
        "--from-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=datetime(2000, 1, 1),
        help="Download models from this date (format: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--num-models", type=int, default=10, help="Number of models to download"
    )
    parser.add_argument(
        "--num-datasets", type=int, default=10, help="Number of datasets to download"
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

    # Initialize extractor
    extractor = initialize_extractor(config_path)

    # Extract data (limited sample)
    extracted_models_df = extractor.download_models(
        num_models=args.num_models,
        from_date=datetime(2023, 1, 1),
        output_dir=args.output_dir,
        save_result_in_json=False,
        save_raw_data=False,
        update_recent=False,
        threads=4,
    )

    extracted_datasets_df = extractor.download_datasets(
        num_datasets=args.num_datasets,
        output_dir=args.output_dir,
        save_result_in_json=False,
        update_recent=False,
        threads=4,
    )

    # Initialize transformer
    transformer = initialize_transform_hf(config_path)

    models_kg, models_extraction_metadata = transformer.transform_HF_models(
        extracted_df=extracted_models_df,
        save_output_in_json=False,
        output_dir=args.output_dir,
    )

    datasets_kg, datasets_extraction_metadata = transformer.transform_HF_datasets(
        extracted_df=extracted_datasets_df,
        save_output_in_json=False,
        output_dir=args.output_dir,
    )

    kg_integrated = transformer.unify_graphs(
        [models_kg, datasets_kg],
        save_output_in_json=True,
        output_dir=args.output_dir + "/kg",
    )

    extraction_metadata_integrated = transformer.unify_graphs(
        [models_extraction_metadata, datasets_extraction_metadata],
        save_output_in_json=True,
        output_dir=args.output_dir + "/metadata",
    )

    # Initialize loader
    loader = initialize_load_processor(kg_files_directory)

    # loader.clean_DBs()

    # Load data
    loader.update_dbs_with_kg(kg_integrated, extraction_metadata_integrated)


if __name__ == "__main__":
    main()
