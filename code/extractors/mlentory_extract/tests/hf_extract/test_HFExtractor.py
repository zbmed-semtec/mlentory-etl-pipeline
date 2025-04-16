import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import os
from datetime import datetime

from mlentory_extract.hf_extract.HFExtractor import HFExtractor
from mlentory_extract.core.ModelCardToSchemaParser import ModelCardToSchemaParser
from mlentory_extract.hf_extract.HFDatasetManager import HFDatasetManager


class TestHFExtractor:
    """
    Test suite for HFExtractor class, focusing on model and related entity extraction.
    """
    
    @pytest.fixture
    def mock_model_df(self):
        """
        Create a mock model dataframe with representative structure.
        
        Returns:
            pd.DataFrame: Mock dataframe containing model data
        """
        # Create first model entry
        model1_data = {
            "schema.org:identifier": [{"data": "distilbert/distilbert-base-uncased-finetuned-sst-2-english", 
                                     "extraction_method": "Parsed_from_HF_dataset", 
                                     "confidence": 1.0, 
                                     "extraction_time": "2025-04-16_14-02-02"}],
            "schema.org:keywords": [{"data": ["text classification"], 
                                   "extraction_method": "Parsed_from_HF_tags", 
                                   "confidence": 1.0, 
                                   "extraction_time": "2025-04-16_14-02-02"}],
            "fair4ml:mlTask": [{"data": ["text classification"], 
                              "extraction_method": "Parsed_from_HF_tags", 
                              "confidence": 1.0, 
                              "extraction_time": "2025-04-16_14-02-02"}],
            "fair4ml:fineTunedFrom": [{"data": "distilbert/distilbert-base-uncased", 
                                     "extraction_method": "Parsed_from_HF_tags", 
                                     "confidence": 1.0, 
                                     "extraction_time": "2025-04-16_14-02-02"}],
            "fair4ml:trainedOn": [{"data": ["sst2", "glue"], 
                                 "extraction_method": "Parsed_from_HF_tags", 
                                 "confidence": 1.0, 
                                 "extraction_time": "2025-04-16_14-02-02"}],
            "fair4ml:evaluatedOn": [{"data": ["sst2", "glue"], 
                                   "extraction_method": "Parsed_from_HF_tags", 
                                   "confidence": 1.0, 
                                   "extraction_time": "2025-04-16_14-02-02"}],
            "fair4ml:sharedBy": [{"data": "distilbert", 
                                "extraction_method": "Parsed_from_HF_dataset", 
                                "confidence": 1.0, 
                                "extraction_time": "2025-04-16_14-02-02"}]
        }
        
        # Create second model entry
        model2_data = {
            "schema.org:identifier": [{"data": "distilbert/distilbert-base-uncased", 
                                     "extraction_method": "Parsed_from_HF_dataset", 
                                     "confidence": 1.0, 
                                     "extraction_time": "2025-04-16_14-02-02"}],
            "schema.org:keywords": [{"data": ["fill mask"], 
                                   "extraction_method": "Parsed_from_HF_tags", 
                                   "confidence": 1.0, 
                                   "extraction_time": "2025-04-16_14-02-02"}],
            "fair4ml:mlTask": [{"data": ["fill mask"], 
                              "extraction_method": "Parsed_from_HF_tags", 
                              "confidence": 1.0, 
                              "extraction_time": "2025-04-16_14-02-02"}],
            "fair4ml:fineTunedFrom": [{"data": "Information not found", 
                                     "extraction_method": "Parsed_from_HF_tags", 
                                     "confidence": 1.0, 
                                     "extraction_time": "2025-04-16_14-02-02"}],
            "fair4ml:trainedOn": [{"data": ["bookcorpus", "wikipedia"], 
                                 "extraction_method": "Parsed_from_HF_tags", 
                                 "confidence": 1.0, 
                                 "extraction_time": "2025-04-16_14-02-02"}],
            "fair4ml:evaluatedOn": [{"data": ["bookcorpus", "wikipedia"], 
                                   "extraction_method": "Parsed_from_HF_tags", 
                                   "confidence": 1.0, 
                                   "extraction_time": "2025-04-16_14-02-02"}],
            "fair4ml:sharedBy": [{"data": "distilbert", 
                                "extraction_method": "Parsed_from_HF_dataset", 
                                "confidence": 1.0, 
                                "extraction_time": "2025-04-16_14-02-02"}]
        }
        
        # Create dataframe with both models
        df1 = pd.DataFrame([model1_data])
        df2 = pd.DataFrame([model2_data])
        return pd.concat([df1, df2], ignore_index=True)
    
    @pytest.fixture
    def mock_dataset_df(self):
        """
        Create a mock dataset dataframe.
        
        Returns:
            pd.DataFrame: Mock dataframe containing dataset information
        """
        datasets_data = [
            {
                "datasetId": "sst2",
                "crossaint_metadata": {
                    "name": "SST-2",
                    "description": "Stanford Sentiment Treebank v2",
                    "license": "CC BY-SA 4.0"
                }
            },
            {
                "datasetId": "glue",
                "crossaint_metadata": {
                    "name": "GLUE",
                    "description": "General Language Understanding Evaluation benchmark",
                    "license": "CC BY 4.0"
                }
            },
            {
                "datasetId": "bookcorpus",
                "crossaint_metadata": {
                    "name": "BookCorpus",
                    "description": "A large corpus of text from books",
                    "license": "Fair Use"
                }
            },
            {
                "datasetId": "wikipedia",
                "crossaint_metadata": {
                    "name": "Wikipedia",
                    "description": "English Wikipedia dump",
                    "license": "CC BY-SA 3.0"
                }
            }
        ]
        return pd.DataFrame(datasets_data)
    
    @pytest.fixture
    def mock_articles_df(self):
        """
        Create a mock articles dataframe.
        
        Returns:
            pd.DataFrame: Mock dataframe containing article information
        """
        articles_data = [
            {
                "arxiv_id": "1910.01108",
                "title": "DistilBERT, a distilled version of BERT",
                "authors": ["Victor Sanh", "Lysandre Debut", "Julien Chaumond", "Thomas Wolf"],
                "abstract": "This paper introduces DistilBERT, a distilled version of BERT."
            }
        ]
        return pd.DataFrame(articles_data)
    
    @pytest.fixture
    def mock_hf_extractor(self, mock_model_df, mock_dataset_df, mock_articles_df):
        """
        Create a mock HFExtractor with patched methods.
        
        Returns:
            HFExtractor: Mock HFExtractor instance
        """
        # Create mocks for dependencies
        mock_parser = Mock(spec=ModelCardToSchemaParser)
        mock_dataset_manager = Mock(spec=HFDatasetManager)
        
        # Create HFExtractor with mock dependencies
        extractor = HFExtractor(parser=mock_parser, dataset_manager=mock_dataset_manager)
        
        # Mock methods that would be called by download_models_with_related_entities
        extractor.download_models = MagicMock(return_value=mock_model_df.iloc[[0]])
        extractor.download_specific_models = MagicMock(return_value=mock_model_df.iloc[[1]])
        extractor.get_related_entities_names = MagicMock(return_value={
            "datasets": {"sst2", "glue", "bookcorpus", "wikipedia"},
            "base_models": {"distilbert/distilbert-base-uncased"},
            "licenses": {"CC BY-SA 4.0", "CC BY 4.0"},
            "keywords": {"text classification", "fill mask"},
            "articles": {"1910.01108"}
        })
        extractor.download_specific_datasets = MagicMock(return_value=mock_dataset_df)
        extractor.download_specific_arxiv_metadata = MagicMock(return_value=mock_articles_df)
        extractor.get_keywords = MagicMock(return_value=pd.DataFrame([
            {"keyword": "text classification", "type": "task"},
            {"keyword": "fill mask", "type": "task"}
        ]))
        
        # Mock download_related_entities to return dictionary with all entity types
        extractor.download_related_entities = MagicMock(return_value={
            "models": mock_model_df.iloc[[1]],  # Base model
            "datasets": mock_dataset_df,
            "articles": mock_articles_df,
            "keywords": pd.DataFrame([
                {"keyword": "text classification", "type": "task"},
                {"keyword": "fill mask", "type": "task"}
            ])
        })
        
        return extractor

    def test_download_models_with_related_entities_basic(self, mock_hf_extractor, tmp_path):
        """
        Test basic functionality of download_models_with_related_entities.
        
        Args:
            mock_hf_extractor: Mocked HFExtractor fixture
            tmp_path: Pytest temporary directory fixture
        """
        # Set up output directory
        output_dir = str(tmp_path / "output")
        
        # Call the method under test
        result = mock_hf_extractor.download_models_with_related_entities(
            num_models=1,
            update_recent=True,
            related_entities_to_download=["datasets", "base_models", "articles", "keywords"],
            output_dir=output_dir,
            save_initial_data=False,
            save_result_in_json=False,
            threads=1,
            depth=1
        )
        
        # Assert that download_models was called with the right parameters
        mock_hf_extractor.download_models.assert_called_once_with(
            num_models=1,
            update_recent=True,
            output_dir=output_dir,
            save_raw_data=False,
            save_result_in_json=False
        )
        
        # Assert that get_related_entities_names was called
        mock_hf_extractor.get_related_entities_names.assert_called()
        
        # Assert that download_related_entities was called with the right parameters
        mock_hf_extractor.download_related_entities.assert_called_once()
        
        # Verify the structure of the result
        assert isinstance(result, dict)
        assert "models" in result
        assert isinstance(result["models"], pd.DataFrame)
        
        # Check that the result contains the expected related entities
        for entity_type in ["datasets", "articles", "keywords"]:
            assert entity_type in result
            assert isinstance(result[entity_type], pd.DataFrame)
    
    def test_download_models_with_related_entities_depth_zero(self, mock_hf_extractor, tmp_path):
        """
        Test download_models_with_related_entities with depth=0.
        
        Args:
            mock_hf_extractor: Mocked HFExtractor fixture
            tmp_path: Pytest temporary directory fixture
        """
        output_dir = str(tmp_path / "output")
        
        # Call with depth=0, should return empty dict
        result = mock_hf_extractor.download_models_with_related_entities(
            num_models=1, 
            depth=0,
            output_dir=output_dir
        )
        
        assert result == {}
        mock_hf_extractor.download_models.assert_not_called()
    
    def test_download_models_with_related_entities_depth_two(self, mock_hf_extractor, tmp_path):
        """
        Test download_models_with_related_entities with depth=2.
        
        Args:
            mock_hf_extractor: Mocked HFExtractor fixture
            tmp_path: Pytest temporary directory fixture
        """
        output_dir = str(tmp_path / "output")
        
        # Call with depth=2
        result = mock_hf_extractor.download_models_with_related_entities(
            num_models=1,
            depth=2,
            output_dir=output_dir
        )
        
        # Should call download_models once and download_specific_models at least once
        mock_hf_extractor.download_models.assert_called_once()
        mock_hf_extractor.download_specific_models.assert_called()
        
        # Result should have models
        assert "models" in result
    
    def test_download_models_with_related_entities_select_entities(self, mock_hf_extractor, tmp_path):
        """
        Test download_models_with_related_entities with specific entity selection.
        
        Args:
            mock_hf_extractor: Mocked HFExtractor fixture
            tmp_path: Pytest temporary directory fixture
        """
        output_dir = str(tmp_path / "output")
        
        # Only select datasets and base_models
        result = mock_hf_extractor.download_models_with_related_entities(
            num_models=1,
            related_entities_to_download=["datasets", "base_models"],
            output_dir=output_dir
        )
        
        # Check that download_related_entities was called with the right parameters
        call_args = mock_hf_extractor.download_related_entities.call_args[1]
        assert call_args["related_entities_to_download"] == ["datasets", "base_models"]
        
        # Result should have models and datasets
        assert "models" in result
        assert "datasets" in result
    
    @patch('os.makedirs')
    def test_download_models_with_related_entities_save_json(self, mock_makedirs, mock_hf_extractor, tmp_path):
        """
        Test download_models_with_related_entities with save_result_in_json=True.
        
        Args:
            mock_makedirs: Mocked os.makedirs function
            mock_hf_extractor: Mocked HFExtractor fixture
            tmp_path: Pytest temporary directory fixture
        """
        output_dir = str(tmp_path / "output")
        
        # Call with save_result_in_json=True
        result = mock_hf_extractor.download_models_with_related_entities(
            num_models=1,
            output_dir=output_dir,
            save_result_in_json=True
        )
        
        # Should call download_models with save_result_in_json=True
        mock_hf_extractor.download_models.assert_called_once_with(
            num_models=1,
            update_recent=True,
            output_dir=output_dir,
            save_raw_data=False,
            save_result_in_json=True
        )
        
        # Should call download_related_entities with save_result_in_json=True
        call_args = mock_hf_extractor.download_related_entities.call_args[1]
        assert call_args["save_result_in_json"] is True
