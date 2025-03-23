import time
from datasets import load_dataset
from datetime import datetime
import pandas as pd
import os
from typing import List
import pytest
from unittest.mock import Mock, MagicMock

from mlentory_extract.hf_extract import HFExtractor
from mlentory_extract.core import ModelCardQAParser


class TestHFExtractor:
    """
    A class to test the HFExtractor class
    """

    @pytest.fixture
    def mock_dataset(self) -> pd.DataFrame:
        """
        Create a mock dataset that mimics the HuggingFace dataset structure

        Returns:
            pd.DataFrame: A mock dataset object
        """
        mock_df = pd.DataFrame(
            {
                "modelId": ["model1"],
                "author": ["author1"],
                "createdAt": ["2024-01-01"],
                "last_modified": ["2024-01-01"],
                "downloads": [100],
                "likes": [10],
                "library_name": ["pytorch"],
                "tags": [["tag1", "tag2"]],
                "pipeline_tag": ["test"],
                "card": ["test card"],
            }
        )

        return mock_df

    @pytest.fixture
    def mock_parser(self) -> Mock:
        """
        Create a mock parser with predefined behavior

        Returns:
            Mock: A mock parser object
        """
        mock_parser = Mock(spec=ModelCardQAParser)
        default_df = pd.DataFrame(
            {
                "modelId": ["model1"],
                "author": ["author1"],
                "createdAt": ["2024-01-01"],
                "last_modified": ["2024-01-01"],
                "downloads": [100],
                "likes": [10],
                "library_name": ["pytorch"],
                "tags": [["tag1", "tag2"]],
                "pipeline_tag": ["test"],
                "card": ["test card"],
            }
        )

        # Add base question columns to default_df
        for i in range(2):  # 2 questions as defined in the mock
            default_df[f"q_id_{i}"] = [
                [
                    {
                        "data": f"test_answer_{i}",
                        "confidence": 0.9,
                        "extraction_method": "mock",
                        "extraction_time": "2024-01-01",
                    }
                ]
            ]

        # Mock the methods that are called by HFExtractor
        mock_parser.parse_fields_from_tags_HF.return_value = default_df.copy()
        mock_parser.parse_known_fields_HF.return_value = default_df.copy()
        mock_parser.parse_fields_from_txt_HF_matching.return_value = default_df.copy()

        # Mock the questions property
        mock_parser.questions = ["Test question 1", "Test question 2"]

        return mock_parser

    @pytest.fixture
    def extractor_empty(self) -> HFExtractor:
        """
        Create an HFExtractor with empty questions and tags

        Returns:
            HFExtractor: An HFExtractor instance
        """
        extractor = HFExtractor(
            qa_model="Intel/dynamic_tinybert",
            questions=[],
            tags_language=[],
            tags_libraries=[],
            tags_other=[],
            tags_task=[],
        )

        return extractor

    @pytest.fixture
    def extractor_full(self) -> HFExtractor:
        """
        Create an HFExtractor with real questions and tags

        Returns:
            HFExtractor: An HFExtractor instance
        """

        def load_tsv_file_to_list(path: str) -> List[str]:
            return [val[0] for val in pd.read_csv(path, sep="\t").values.tolist()]

        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Navigate up 3 levels and into configuration
        config_path = os.path.join(
            current_dir, "..", "..", "..", "config", "hf", "extract"
        )

        questions = load_tsv_file_to_list(os.path.join(config_path, "questions.tsv"))
        tags_language = load_tsv_file_to_list(
            os.path.join(config_path, "tags_language.tsv")
        )
        tags_libraries = load_tsv_file_to_list(
            os.path.join(config_path, "tags_libraries.tsv")
        )
        tags_other = load_tsv_file_to_list(os.path.join(config_path, "tags_other.tsv"))
        tags_task = load_tsv_file_to_list(os.path.join(config_path, "tags_task.tsv"))

        # Initialize extractor with configuration
        extractor = HFExtractor(
            qa_model="Intel/dynamic_tinybert",
            questions=questions,
            tags_language=tags_language,
            tags_libraries=tags_libraries,
            tags_other=tags_other,
            tags_task=tags_task,
        )

        return extractor

    def test_download_models_basic(
        self, extractor_empty: HFExtractor, mock_parser: Mock
    ) -> None:
        """
        Test basic functionality of download_models method with mocked components
        """
        # Set the parser to the mock parser
        extractor_empty.parser = mock_parser

        # Call download_models
        df = extractor_empty.download_models(
            num_models=1,
            output_dir="./test_outputs",
            save_raw_data=False,
            save_result_in_json=False,
        )

        print(type(df))
        # Verify the results
        assert isinstance(df, pd.DataFrame)
        assert "q_id_0_Test question 1" in df.columns
        assert "q_id_1_Test question 2" in df.columns
        assert df.loc[0, "q_id_0_Test question 1"][0]["data"] == "test_answer_0"
        assert df.loc[0, "q_id_1_Test question 2"][0]["data"] == "test_answer_1"
        assert all(col.startswith("q_id_") for col in df.columns)

        # Verify that parser methods were called
        extractor_empty.parser.parse_fields_from_tags_HF.assert_called_once()
        extractor_empty.parser.parse_known_fields_HF.assert_called_once()
        extractor_empty.parser.parse_fields_from_txt_HF_matching.assert_called_once()

    def test_download_models_output_files(
        self,
        extractor_empty: HFExtractor,
        mock_parser: Mock,
        mock_dataset: Mock,
        tmp_path,
        monkeypatch,
    ) -> None:
        """
        Test that download_models creates expected output files
        """
        # Mock the load_dataset function
        monkeypatch.setattr(
            "mlentory_extract.hf_extract.HFDatasetManager.get_model_metadata_dataset",
            lambda self, update_recent=False, limit=2: mock_dataset,
        )
        extractor_empty.parser = mock_parser

        output_dir = tmp_path / "outputs"

        df = extractor_empty.download_models(
            num_models=1,
            output_dir=str(output_dir),
            save_raw_data=True,
            save_result_in_json=True,
        )
        # Add small wait to ensure files are written
        # time.sleep(0.5)
        # Check that output files were created
        files = list(output_dir.glob("*"))
        print(files)
        assert len(files) == 2
        assert any(f.name.endswith("Original_HF_Dataframe.csv") for f in files)
        assert any(f.name.endswith("Processed_HF_Dataframe.json") for f in files)

    def test_empty_dataset_handling(
        self, extractor_empty: HFExtractor, monkeypatch, mock_dataset: Mock
    ) -> None:
        """
        Test handling of empty dataset
        """
        # Mock the load_dataset function
        monkeypatch.setattr(
            "mlentory_extract.hf_extract.HFDatasetManager.get_model_metadata_dataset",
            lambda self, update_recent=False, limit=2: mock_dataset,
        )

        # Check an exception is raised when num_models is 0
        with pytest.raises(Exception):
            extractor_empty.download_models(num_models=0)

    def test_parser_extraction_results(self, extractor_full: HFExtractor) -> None:
        """
        Test that the parser extracts expected information

        Args:
            extractor (HFExtractor): The HFExtractor object
        """
        # Check that we're in the expected test folder

        df = extractor_full.download_models(
            num_models=2, save_raw_data=False, save_result_in_json=False
        )
        # Check that each cell contains the expected extraction info structure
        first_cell = df.iloc[0, 0][0]  # Get first cell's extraction info
        assert isinstance(first_cell, dict)
        assert all(
            key in first_cell
            for key in ["data", "extraction_method", "confidence", "extraction_time"]
        )
        assert isinstance(first_cell["confidence"], float)
        assert isinstance(first_cell["extraction_time"], str)

    def test_download_specific_datasets(
        self, extractor_empty: HFExtractor, monkeypatch, tmp_path
    ) -> None:
        """
        Test downloading specific datasets by name.

        Args:
            extractor_empty (HFExtractor): The HFExtractor fixture
            monkeypatch: pytest monkeypatch fixture
            tmp_path: pytest temporary directory fixture
        """
        # Mock sample dataset entries
        sample_data = {
            "dataset1": {
                "name": "Dataset 1",
                "description": "Test dataset 1",
            },
            "dataset2": {
                "name": "Dataset 2",
                "description": "Test dataset 2",
            },
        }
        
        # Create a mock DataFrame to return
        mock_df = pd.DataFrame([
            {
                "datasetId": "dataset1",
                "croissant_metadata": sample_data["dataset1"],
                "extraction_metadata": {
                    "extraction_method": "Downloaded_from_HF_Croissant_endpoint",
                    "confidence": 1.0,
                    "extraction_time": "2024-01-01_12-00-00",
                }
            },
            {
                "datasetId": "dataset2",
                "croissant_metadata": sample_data["dataset2"],
                "extraction_metadata": {
                    "extraction_method": "Downloaded_from_HF_Croissant_endpoint",
                    "confidence": 1.0,
                    "extraction_time": "2024-01-01_12-00-00",
                }
            }
        ])
        
        # Mock the get_specific_datasets_metadata method
        monkeypatch.setattr(
            "mlentory_extract.hf_extract.HFDatasetManager.get_specific_datasets_metadata",
            lambda self, dataset_names, threads=4: mock_df
        )
        
        output_dir = tmp_path / "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Test with valid dataset names
        dataset_names = ["dataset1", "dataset2", "nonexistent_dataset"]
        result_df = extractor_empty.download_specific_datasets(
            dataset_names=dataset_names,
            output_dir=str(output_dir),
            save_result_in_json=True,
        )
        
        # Check dataframe structure and content
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2  # Only two valid datasets should be processed
        assert "datasetId" in result_df.columns
        assert "croissant_metadata" in result_df.columns
        assert "extraction_metadata" in result_df.columns
        
        # Check that dataset IDs are correctly recorded
        dataset_ids = result_df["datasetId"].tolist()
        assert "dataset1" in dataset_ids
        assert "dataset2" in dataset_ids
        
        # Verify output file was created
        files = list(output_dir.glob("*"))
        assert any(f.name.endswith("Extracted_Specific_Datasets_HF_df.json") for f in files)
