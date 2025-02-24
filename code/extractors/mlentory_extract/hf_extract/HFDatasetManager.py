from typing import Optional, Union, Literal, Dict, List
from datasets import load_dataset
import pandas as pd
from huggingface_hub import HfApi, ModelCard
from datetime import datetime
import requests
import itertools
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        if api_token != None:
            self.token = api_token
            self.api = HfApi(token=api_token)
        else:
            self.api = HfApi()

    def get_model_metadata_dataset(
        self, update_recent: bool = True, limit: int = 5, threads: int = 4
    ) -> pd.DataFrame:
        """
        Retrieve and optionally update the HuggingFace dataset containing model card information.

        The method first loads the existing dataset and then updates it with any models
        that have been modified since the most recent entry in the dataset.

        Args:
            update_recent (bool): Whether to fetch and append recent model updates.
                Defaults to True.
            limit (int): Maximum number of models to fetch. Defaults to 100.
            threads (int): Number of threads for parallel processing. Defaults to 4.
        Returns:
            pd.DataFrame: DataFrame containing model card information

        Raises:
            Exception: If there's an error loading or updating the dataset
        """
        try:
            # Load base dataset
            dataset = load_dataset(
                "librarian-bots/model_cards_with_metadata",
                # revision="0b3e7a79eae8a5dd28080f06065a988ca1fbf050",
            )["train"].to_pandas()
            
            # Discard all models with a card with a length less than 1000
            dataset = dataset[dataset["card"].str.len() > 1000]

            # trim the dataset to the limit
            dataset = dataset[: min(limit, len(dataset))]

            if not update_recent:
                return dataset

            # Get the most recent modification date from the dataset
            latest_modification = dataset["last_modified"].max()

            recent_models = self.get_recent_models_metadata(
                limit, latest_modification, threads
            )

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
        self, limit: int, latest_modification: datetime, threads: int = 4
    ) -> pd.DataFrame:
        """
        Retrieve recent models metadata from HuggingFace API.

        Args:
            limit (int): Maximum number of models to fetch.
            latest_modification (datetime): The latest modification date to filter models.
            threads (int): Number of threads for parallel processing. Defaults to 4.

        Returns:
            pd.DataFrame: DataFrame containing model metadata
        """
        models = self.api.list_models(
            limit=limit, sort="lastModified", direction=-1, full=True
        )

        def process_model(model):
            if model.last_modified <= latest_modification:
                return None

            card = None
            try:
                if self.token:
                    card = ModelCard.load(model.modelId, token=self.token)
                else:
                    card = ModelCard.load(model.modelId)
            except Exception as e:
                print()
                print(f"Error loading model card for {model.id}: {e}")

            model_info = {
                "modelId": model.id,
                "author": model.author,
                "last_modified": model.last_modified,
                "downloads": model.downloads,
                "likes": model.likes,
                "library_name": model.library_name,
                "tags": model.tags,
                "pipeline_tag": model.pipeline_tag,
                "createdAt": model.created_at,
                "card": card.content if card else "",
            }

            return model_info

        model_data = []
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_model = {
                executor.submit(process_model, model): model for model in models
            }
            for future in as_completed(future_to_model):
                result = future.result()
                if result is not None and result["card"] is not None and len(result["card"]) > 1000:
                    model_data.append(result)

        return pd.DataFrame(model_data)

    def get_datasets_metadata(
        self, limit: int, latest_modification: datetime, threads: int = 4
    ) -> pd.DataFrame:
        """
        Retrieve recent datasets metadata from HuggingFace API.

        Args:
            limit (int): Maximum number of datasets to fetch.
            latest_modification (datetime): The latest modification date to filter datasets.
            threads (int): Number of threads for parallel processing. Defaults to 4.

        Returns:
            pd.DataFrame: DataFrame containing dataset metadata with exactly 'limit' rows
                         (or fewer if not enough valid datasets are found)
        """
        # Fetch initial batch of datasets (100x limit to have enough valid ones)
        datasets = list(
            itertools.islice(
                self.api.list_datasets(sort="lastModified", direction=-1), limit + 1000
            )
        )

        dataset_data = []
        futures = []

        def process_dataset(dataset):
            if not (latest_modification is None):
                last_modified = dataset.last_modified.replace(
                    tzinfo=latest_modification.tzinfo
                )
                if last_modified <= latest_modification:
                    return None

            croissant_metadata = self.get_croissant_metadata(dataset.id)
            if croissant_metadata == {}:
                return None
            # Print all the datasets properties
            # print("\nDATASEEEEEEET\n")
            # print(dataset)
            return {
                "datasetId": dataset.id,
                "croissant_metadata": croissant_metadata,
                "extraction_metadata": {
                    "extraction_method": "Downloaded_from_HF_Croissant_endpoint",
                    "confidence": 1.0,
                    "extraction_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                },
            }

        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Submit all tasks
            for dataset in datasets:
                future = executor.submit(process_dataset, dataset)
                futures.append(future)

            # Process results as they complete
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    dataset_data.append(result)
                    # If we've reached the limit, cancel remaining futures
                    if len(dataset_data) >= limit:
                        for f in futures:
                            f.cancel()
                        break

        # Trim results to exact limit if we got more than needed
        dataset_data = dataset_data[:limit]
        return pd.DataFrame(dataset_data)

    def get_croissant_metadata(self, dataset_id: str) -> Dict:
        """
        Retrieve croissant metadata for a given dataset.

        Args:
            dataset_id (str): The ID of the dataset to retrieve metadata for.

        Returns:
            Dict: The croissant metadata for the dataset, or an empty dictionary if not found.
        """
        API_URL = f"https://huggingface.co/api/datasets/{dataset_id}/croissant"
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
        else:
            headers = {}
        response = requests.get(API_URL, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return {}

    def get_arxiv_metadata_dataset(self) -> pd.DataFrame:
        """
        Retrieve the HuggingFace dataset containing arxiv metadata.

        Returns:
            pd.DataFrame: DataFrame containing arxiv metadata
        """
        try:
            return load_dataset("librarian-bots/arxiv-metadata-snapshot")[
                "train"
            ].to_pandas()
        except Exception as e:
            raise Exception(f"Error loading arxiv metadata dataset: {str(e)}")
