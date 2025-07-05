import pytest
import pandas as pd
import os
from datetime import datetime
from typing import List, Tuple

from mlentory_extract.hf_extract import HFExtractor, HFDatasetManager
from mlentory_transform.core.MlentoryTransform import (
    MlentoryTransform,
    KnowledgeGraphHandler,
)
from mlentory_transform.hf_transform.TransformHF import TransformHF
from mlentory_load.core.LoadProcessor import LoadProcessor
from mlentory_load.core.GraphHandler import GraphHandler
from mlentory_load.dbHandler import SQLHandler, RDFHandler, IndexHandler


def load_tsv_file_to_list(path: str) -> List[str]:
    return [val[0] for val in pd.read_csv(path, sep="\t").values.tolist()]


def setup_dirs(output_dir: str):
    kg_output_dir = output_dir + "/kg"
    metadata_output_dir = output_dir + "/metadata"
    kg_files_directory = output_dir + "/kgs_to_load"

    if not os.path.exists(kg_output_dir):
        os.makedirs(kg_output_dir)
    if not os.path.exists(metadata_output_dir):
        os.makedirs(metadata_output_dir)
    if not os.path.exists(kg_files_directory):
        os.makedirs(kg_files_directory)

    for file in os.listdir(kg_output_dir):
        if file.endswith(".json"):
            os.remove(os.path.join(kg_output_dir, file))
    for file in os.listdir(metadata_output_dir):
        if file.endswith(".json"):
            os.remove(os.path.join(metadata_output_dir, file))
    for file in os.listdir(kg_files_directory):
        if file.endswith(".ttl"):
            os.remove(os.path.join(kg_files_directory, file))


@pytest.fixture
def initialize_extractor() -> HFExtractor:
    """
    Initializes the extractor with the configuration data.

    Returns:
        HFExtractor: The extractor instance.
    """
    config_path = "config/hf/extract"
    questions = load_tsv_file_to_list(f"{config_path}/questions.tsv")
    tags_language = load_tsv_file_to_list(f"{config_path}/tags_language.tsv")
    tags_libraries = load_tsv_file_to_list(f"{config_path}/tags_libraries.tsv")
    tags_other = load_tsv_file_to_list(f"{config_path}/tags_other.tsv")
    tags_task = load_tsv_file_to_list(f"{config_path}/tags_task.tsv")

    dataset_manager = HFDatasetManager()

    return HFExtractor(
        # qa_model="jxm/cde-small-v2",
        qa_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dataset_manager=dataset_manager,
        questions=questions,
        tags_language=tags_language,
        tags_libraries=tags_libraries,
        tags_other=tags_other,
        tags_task=tags_task,
    )


@pytest.fixture
def initialize_transform() -> TransformHF:
    """
    Initializes the transformer with the configuration data.

    Args:
        config_path (str): The path to the configuration data.

    Returns:
        TransformHF: The transformer instance.
    """
    config_path = "config/hf/transform"
    new_schema = pd.read_csv(
        f"{config_path}/FAIR4ML_schema.csv", sep=",", lineterminator="\n"
    )
    transformations = pd.read_csv(
        f"{config_path}/column_transformations.csv",
        lineterminator="\n",
        sep=",",
    )

    kg_handler = KnowledgeGraphHandler(
        FAIR4ML_schema=new_schema, base_namespace="https://mlentory.com/mlentory_graph/"
    )
    transform_hf = TransformHF(new_schema, transformations)
    transformer = MlentoryTransform(kg_handler, transform_hf)

    return transformer


@pytest.fixture
def initialize_loader() -> LoadProcessor:
    kg_files_directory = "integration/pipeline/outputs/kgs_to_load"
    sqlHandler = SQLHandler(
        host="postgres",
        user="test_user",
        password="test_password",
        database="test_DB",
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
        graph_identifier="https://mlentory.com/mlentory_graph",
        deprecated_graph_identifier="https://mlentory.com/deprecated_mlentory_graph",
    )

    return LoadProcessor(
        kg_files_directory=kg_files_directory,
        GraphHandler=graphHandler,
        SQLHandler=sqlHandler,
        RDFHandler=rdfHandler,
        IndexHandler=elasticsearchHandler,
    )


class TestHFETLIntegration:
    """Integration tests for HuggingFace ETL pipeline."""

    def test_extract_transform_pipeline(
        self, initialize_extractor, initialize_transform, initialize_loader, monkeypatch
    ):
        """
        Test the complete extract-transform pipeline.

        Args:
            config_data: Tuple containing config path, schema, and transformations
            output_dir: Path to output directory
            monkeypatch: pytest fixture for mocking
        """

        output_dir = "integration/pipeline/outputs"
        setup_dirs(output_dir)

        # Initialize extractor
        extractor = initialize_extractor

        # Extract data (limited sample)
        extracted_models_df = extractor.download_models(
            num_models=1,
            from_date=datetime(2023, 1, 1),
            output_dir=output_dir,
            save_result_in_json=False,
            save_raw_data=False,
            update_recent=True,
            threads=4,
        )

        extracted_datasets_df = extractor.download_datasets(
            num_datasets=1,
            output_dir=output_dir,
            save_result_in_json=False,
            update_recent=True,
            threads=4,
        )

        # Initialize transformer
        transformer = initialize_transform

        models_kg, models_metadata = transformer.transform_HF_models(
            extracted_df=extracted_models_df,
            save_output_in_json=False,
            output_dir=output_dir,
        )

        datasets_kg, datasets_metadata = transformer.transform_HF_datasets(
            extracted_df=extracted_datasets_df,
            save_output_in_json=False,
            output_dir=output_dir,
        )

        kg_integrated = transformer.unify_graphs(
            [models_kg, datasets_kg],
            save_output_in_json=False,
            output_dir=output_dir + "/kg",
        )

        metadata_integrated = transformer.unify_graphs(
            [models_metadata, datasets_metadata],
            save_output_in_json=True,
            output_dir=output_dir + "/metadata",
        )

        # Initialize loader
        loader = initialize_loader

        # Load data
        loader.update_dbs_with_kg(metadata_integrated)

        # Verify output files
        files = os.listdir(output_dir)
        print(files)
        assert len(files) >= 2
