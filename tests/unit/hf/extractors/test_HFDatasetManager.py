from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from datetime import datetime, timezone
from datasets import Dataset
from huggingface_hub import ModelInfo
from mlentory_extract.hf_extract import HFDatasetManager


@pytest.fixture
def mock_model_data():
    """
    Fixture providing sample model data that matches the expected schema.

    Returns:
        pd.DataFrame: Sample model data
    """
    return pd.DataFrame(
        [
            {
                "modelId": "bert-base-uncased",
                "author": "google",
                "last_modified": datetime(2023, 1, 1, tzinfo=timezone.utc),
                "downloads": 1000000,
                "likes": 1000,
                "pipeline_tag": "text-classification",
                "tags": ["transformers", "bert", "pytorch"],
                "task_categories": "text-classification",
                "createdAt": datetime(2022, 1, 1, tzinfo=timezone.utc),
                "card": "# BERT Base Uncased\nThis is a sample model card.",
            },
            # {
            #     "modelId": "gpt-3.5-turbo",
            #     "author": "openai",
            #     "last_modified": datetime(2023, 2, 1, tzinfo=timezone.utc),
            #     "downloads": 500000,
            #     "likes": 500,
            #     "pipeline_tag": "text-generation",
            #     "tags": ["transformers", "gpt", "openai"],
            #     "task_categories": "text-generation",
            #     "createdAt": datetime(2022, 2, 1, tzinfo=timezone.utc),
            #     "card": "# GPT-3.5 Turbo\nThis is a sample model card.",
            # }
        ]
    )


@pytest.fixture
def mock_hf_api():
    """
    Fixture providing a mocked HuggingFace API client.

    Returns:
        MagicMock: Mocked HF API instance
    """
    mock_api = MagicMock()
    mock_models = []

    # Create several mock models with different properties
    model_configs = [
        {
            "id": "bert-base-uncased",
            "author": "google",
            "last_modified": datetime(2023, 1, 1, tzinfo=timezone.utc),
            "downloads": 1000000,
            "likes": 1000,
            "pipeline_tag": "text-classification",
            "tags": ["transformers", "bert", "pytorch"],
            "created_at": datetime(2022, 1, 1, tzinfo=timezone.utc),
        },
        {
            "id": "gpt2",
            "author": "openai",
            "last_modified": datetime(2023, 2, 1, tzinfo=timezone.utc),
            "downloads": 500000,
            "likes": 800,
            "pipeline_tag": "text-generation",
            "tags": ["transformers", "gpt", "pytorch"],
            "created_at": datetime(2022, 6, 1, tzinfo=timezone.utc),
        },
        {
            "id": "t5-base",
            "author": "google",
            "last_modified": datetime(2023, 3, 1, tzinfo=timezone.utc),
            "downloads": 300000,
            "likes": 600,
            "pipeline_tag": "text2text-generation",
            "tags": ["transformers", "t5", "pytorch"],
            "created_at": datetime(2022, 9, 1, tzinfo=timezone.utc),
        },
    ]

    for config in model_configs:
        mock_model = MagicMock()
        for key, value in config.items():
            setattr(mock_model, key, value)
        mock_models.append(mock_model)

    mock_api.list_models.return_value = mock_models

    return mock_api


class TestHFDatasetManager:
    """Test suite for HFDatasetManager class."""

    @patch("mlentory_extract.hf_extract.HFDatasetManager.load_dataset")
    def test_get_model_metadata_dataset_without_update(
        self, mock_load_dataset, mock_model_data
    ):
        """
        Test getting model metadata without updating recent models.

        Args:
            mock_load_dataset: Mocked load_dataset function
            mock_model_data: Sample model data fixture
        """
        # Setup
        mock_dataset = MagicMock(spec=Dataset)
        mock_dataset.to_pandas.return_value = mock_model_data
        mock_load_dataset.return_value = {"train": mock_dataset}

        manager = HFDatasetManager()

        # Execute
        result = manager.get_model_metadata_dataset(update_recent=False)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mock_model_data)
        assert list(result.columns) == [
            "modelId",
            "author",
            "last_modified",
            "downloads",
            "likes",
            "pipeline_tag",
            "tags",
            "task_categories",
            "createdAt",
            "card",
        ]
        mock_load_dataset.assert_called_once_with(
            "librarian-bots/model_cards_with_metadata"
        )

    @patch("huggingface_hub.ModelCard")
    def test_get_recent_models_metadata(self, mock_model_card):
        """
        Test retrieving recent models metadata.
        Args:
            mock_model_card: Mocked ModelCard class
        """
        # Setup
        mock_card = MagicMock()
        mock_card.content = "# Test Model Card"
        mock_model_card.load.return_value = mock_card
        manager = HFDatasetManager()
        latest_modification = datetime(2023, 1, 1, tzinfo=timezone.utc)

        # Execute
        result = manager.get_recent_models_metadata(
            limit=3, latest_modification=latest_modification
        )
        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

        # Verify all models are newer than latest_modification
        assert all(date > latest_modification for date in result["last_modified"])
        expected_columns = [
            "modelId",
            "author",
            "last_modified",
            "downloads",
            "likes",
            "pipeline_tag",
            "tags",
            "library_name",
            "createdAt",
            "card",
        ]
        assert list(result.columns) == expected_columns

    @patch("mlentory_extract.hf_extract.HFDatasetManager.load_dataset")
    def test_get_model_metadata_dataset_with_error(self, mock_load_dataset):
        """
        Test error handling when loading dataset fails.

        Args:
            mock_load_dataset: Mocked load_dataset function
        """
        # Setup
        mock_load_dataset.side_effect = Exception("Dataset not found")
        manager = HFDatasetManager()

        # Execute and Assert
        with pytest.raises(Exception) as exc_info:
            manager.get_model_metadata_dataset(update_recent=False)
        assert "Error loading or updating model cards dataset" in str(exc_info.value)

    @patch("mlentory_extract.hf_extract.HFDatasetManager.load_dataset")
    def test_get_model_metadata_dataset_with_update(
        self, mock_load_dataset, mock_hf_api
    ):
        """
        Test getting model metadata with updating recent models.

        Args:
            mock_load_dataset: Mocked load_dataset function
            mock_hf_api: Mocked HfApi class
        """
        # Setup
        manager = HFDatasetManager()
        manager.api = mock_hf_api

        # Create historical dataset (older models)
        historical_data = pd.DataFrame(
            [
                {
                    "modelId": "old-model-1",
                    "author": "legacy-author",
                    "last_modified": datetime(2021, 6, 6, tzinfo=timezone.utc),
                    "downloads": 100000,
                    "likes": 300,
                    "pipeline_tag": "text-classification",
                    "tags": ["old", "legacy"],
                    "task_categories": "text-classification",
                    "createdAt": datetime(2021, 1, 1, tzinfo=timezone.utc),
                    "card": "# Old Model Card",
                },
                {
                    "modelId": "old-model-2",
                    "author": "legacy-author",
                    "last_modified": datetime(2021, 6, 1, tzinfo=timezone.utc),
                    "downloads": 200000,
                    "likes": 400,
                    "pipeline_tag": "text-generation",
                    "tags": ["old", "legacy"],
                    "task_categories": "text-generation",
                    "createdAt": datetime(2021, 3, 1, tzinfo=timezone.utc),
                    "card": "# Another Old Model Card",
                },
            ]
        )

        # Mock the dataset loading
        mock_dataset = MagicMock(spec=Dataset)
        mock_dataset.to_pandas.return_value = historical_data
        mock_load_dataset.return_value = {"train": mock_dataset}

        # Execute
        result = manager.get_model_metadata_dataset(update_recent=True, limit=3)

        # Assert
        assert isinstance(result, pd.DataFrame)
        # 2 historical models + 3 new models from mock_hf_api
        assert len(result) == 5
        # Verify no duplicate model IDs
        assert result["modelId"].duplicated().sum() == 0
        # Verify sorting by last_modified in descending order
        assert result["last_modified"].is_monotonic_decreasing

        # Verify the most recent models are from mock_hf_api
        assert result.iloc[0]["modelId"] == "t5-base"
        assert result.iloc[1]["modelId"] == "gpt2"
        assert result.iloc[2]["modelId"] == "bert-base-uncased"

        # Verify the older models are from historical data
        assert "old-model-1" in result["modelId"].values
        assert "old-model-2" in result["modelId"].values

        # Verify the dataset was loaded correctly
        mock_load_dataset.assert_called_once_with(
            "librarian-bots/model_cards_with_metadata"
        )
