from typing import Optional, Union, Literal, Dict, List
from datasets import load_dataset
import pandas as pd
from huggingface_hub import HfApi, ModelCard
from datetime import datetime
import os


class HFDatasetManager:
    """
    A class for managing HuggingFace dataset and model information.

    This class handles direct interactions with the HuggingFace platform,
    including downloading and creating datasets for both models and datasets information.

    Attributes:
        api (HfApi): HuggingFace API client instance
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
    ):
        """
        Initialize the HuggingFace Dataset Manager.

        Args:
            api_token (Optional[str]): HuggingFace API token for authenticated requests.
                Defaults to None.

        Raises:
            ValueError: If the model_cards_dataset is invalid or inaccessible
        """
        self.token = None
        if api_token:
            self.token = api_token
            self.api = HfApi(token=api_token)
        else:
            self.api = HfApi()

    def get_model_metadata_dataset(
        self, update_recent: bool = True, limit: int = 5
    ) -> pd.DataFrame:
        """
        Retrieve and optionally update the HuggingFace dataset containing model card information.

        The method first loads the existing dataset and then updates it with any models
        that have been modified since the most recent entry in the dataset.

        Args:
            update_recent (bool): Whether to fetch and append recent model updates.
                Defaults to True.
            limit (int): Maximum number of models to fetch. Defaults to 100.
        Returns:
            pd.DataFrame: DataFrame containing model card information

        Raises:
            Exception: If there's an error loading or updating the dataset
        """
        try:
            # Load base dataset
            dataset = load_dataset("librarian-bots/model_cards_with_metadata")[
                "train"
            ].to_pandas()

            # trim the dataset to the limit
            dataset = dataset[: min(limit, len(dataset))]

            if not update_recent:
                return dataset

            # Get the most recent modification date from the dataset
            latest_modification = dataset["last_modified"].max()

            recent_models = self.get_recent_models_metadata(limit, latest_modification)

            if len(recent_models) > 0:

                # Concatenate with original dataset and remove duplicates
                dataset = pd.concat([dataset, recent_models], ignore_index=True)
                dataset = dataset.drop_duplicates(subset=["modelId"], keep="last")

                # Sort by last_modified
                dataset = dataset.sort_values("last_modified", ascending=False)

            return dataset

        except Exception as e:
            raise Exception(f"Error loading or updating model cards dataset: {str(e)}")

    def get_recent_models_metadata(
        self, limit: int, latest_modification: datetime
    ) -> pd.DataFrame:
        """
        Retrieve recent models metadata from HuggingFace API.

        Args:
            limit (int): Maximum number of models to fetch.
            latest_modification (datetime): The latest modification date to filter models.

        Returns:
            pd.DataFrame: DataFrame containing model metadata
        """

        models = self.api.list_models(
            limit=limit, sort="lastModified", direction=-1, full=True
        )

        model_data = []

        for model in models:

            card = None
            try:
                if self.token:
                    card = ModelCard.load(model.modelId, token=self.token)
                else:
                    card = ModelCard.load(model.modelId)
            except Exception as e:
                print()
                print(f"Error loading model card for {model.id}: {e}")

            # Check if this model is newer than our latest modification
            if model.last_modified <= latest_modification:
                break

            model_data.append(
                {
                    "modelId": model.id,
                    "author": model.author,
                    "last_modified": model.last_modified,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "pipeline_tag": model.pipeline_tag,
                    "tags": model.tags,
                    "library_name": model.library_name,
                    "createdAt": model.created_at,
                    "card": card.content if card else "",
                }
            )

        return pd.DataFrame(model_data)

    def get_dataset_metadata_dataset(self) -> pd.DataFrame:
        """
        Retrieve the HuggingFace dataset containing dataset card information.
        """
        try:
            return load_dataset("librarian-bots/dataset_cards_with_metadata")[
                "train"
            ].to_pandas()
        except Exception as e:
            raise Exception(f"Error loading datasets card dataset: {str(e)}")

    def get_arxiv_metadata_dataset(self) -> pd.DataFrame:
        """
        Retrieve the HuggingFace dataset containing arxiv metadata.
        """
        try:
            return load_dataset("librarian-bots/arxiv-metadata-snapshot")[
                "train"
            ].to_pandas()
        except Exception as e:
            raise Exception(f"Error loading arxiv metadata dataset: {str(e)}")
