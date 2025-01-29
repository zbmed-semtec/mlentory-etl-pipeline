import pytest
import pandas as pd
import os
from datetime import datetime
from typing import List, Tuple

from mlentory_extract.hf_extract import HFExtractor, HFDatasetManager
from mlentory_transform.core.MlentoryTransform import MlentoryTransform, KnowledgeGraphHandler
from mlentory_transform.hf_transform.TransformHF import TransformHF


def load_tsv_file_to_list(path: str) -> List[str]:
    return [val[0] for val in pd.read_csv(path, sep="\t").values.tolist()]


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
    new_schema = pd.read_csv(f"{config_path}/M4ML_schema.csv", sep=",",lineterminator="\n")
    transformations = pd.read_csv(
        f"{config_path}/column_transformations.csv",
        lineterminator="\n",
        sep=",",
    )
    
    kg_handler = KnowledgeGraphHandler(M4ML_schema=new_schema, base_namespace="http://test_example.org/")
    transform_hf = TransformHF(new_schema, transformations)
    transformer = MlentoryTransform(kg_handler, transform_hf)
    
    return transformer


class TestHFETLIntegration:
    """Integration tests for HuggingFace ETL pipeline."""

    def test_extract_transform_pipeline(
        self,
        initialize_extractor,
        initialize_transform,
        monkeypatch
    ):
        """
        Test the complete extract-transform pipeline.

        Args:
            config_data: Tuple containing config path, schema, and transformations
            output_dir: Path to output directory
            monkeypatch: pytest fixture for mocking
        """

        # Initialize extractor
        extractor = initialize_extractor
        
        output_dir = "integration/pipeline/outputs"
        # Clean the output directory
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

        # Extract data (limited sample)
        extracted_df = extractor.download_models(
            num_models=2,
            from_date=datetime(2023, 1, 1),
            output_dir=output_dir,
            save_result_in_json=True,
            save_raw_data=False,
            update_recent=True
        )

        # Initialize transformer
        transformer = initialize_transform
        
        transformer.transform_HF_models(
            extracted_df=extracted_df,
            save_output_in_json=True,
            output_dir=output_dir
        )

        # Verify output files
        files = os.listdir(output_dir)
        print(files)
        assert any(file.endswith("_transformation_results.csv") for file in files)
        assert any(file.endswith(".json") for file in files)