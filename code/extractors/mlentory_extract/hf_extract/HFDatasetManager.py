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
        default_card: Optional[str] = None,
    ):
        """
        Initialize the HuggingFace Dataset Manager.

        Args:
            api_token (Optional[str]): HuggingFace API token for authenticated requests.
                Defaults to None.
            default_card (Optional[str]): Default HF card content.
                Defaults to None (uses the standard path).

        Raises:
            ValueError: If the model_cards_dataset is invalid or inaccessible
        """
        self.token = None
        if api_token != None:
            self.token = api_token
            self.api = HfApi(token=api_token)
        else:
            self.api = HfApi()
            
        # Set default card path
        self.default_card = default_card

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
            

            if update_recent:
                # Get the most recent modification date from the dataset
                latest_modification = dataset["last_modified"].max()
                
                recent_models = self.get_recent_models_metadata(
                    limit, latest_modification, threads
                )

                # Concatenate with original dataset and remove duplicates
                dataset = pd.concat([dataset, recent_models], ignore_index=True)
                dataset = dataset.drop_duplicates(subset=["modelId"], keep="last")

                # Sort by last_modified
                dataset = dataset.sort_values("last_modified", ascending=False)
            
            # print("GOT HERREEEEE")
            # Discard models with not enough information
            dataset = self.filter_models(dataset)

            # trim the dataset to the limit
            dataset = dataset[: min(limit, len(dataset))]
            
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
                print(f"Error loading model card for {model.id}: {e}")
                return None

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
            
            if self.has_model_enough_information(model_info):
                return model_info
            else:
                return None

        model_data = []
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_model = {
                executor.submit(process_model, model): model for model in models
            }
            for future in as_completed(future_to_model):
                result = future.result()
                if result is not None:
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

    def filter_models(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out models with default card content.

        This method checks model cards against the default HuggingFace model card template
        and removes entries that are essentially unchanged from the default template.

        Args:
            dataset (pd.DataFrame): The input dataset containing model card information.

        Returns:
            pd.DataFrame: A DataFrame with models that do not have default card content.

        Raises:
            FileNotFoundError: If the default card file is not found.
        """
        # Filter the dataset
        filtered_dataset = dataset[dataset.apply(self.has_model_enough_information, axis=1)]
        
        # Log how many models were filtered out
        removed_count = len(dataset) - len(filtered_dataset)
        if removed_count > 0:
            print(f"Filtered out {removed_count} models with default card content.")
        
        return filtered_dataset

    def has_model_enough_information(self, model_info: Dict):
        """
        Check if a model has enough information to be considered for extraction.

        Args:
            model_info (Dict): A dictionary containing model information.

        Returns:
            bool: True if the model has enough information, False otherwise.
        """
        
        # Discard all models with no pipeline_tag
        if type(model_info["pipeline_tag"]) == str:
            if model_info["pipeline_tag"] == "" or model_info["pipeline_tag"] == None:
                return False
        else:
            if model_info["pipeline_tag"] == None or model_info["pipeline_tag"].isna():
                return False
            
        # Discard models with no modeltags 
        if len(model_info["tags"]) == 0:
            return False
        
            
        # Discard all models with a card with a length less than 200
        if len(model_info["card"]) < 200:
            return False
        
        # Define key phrases that indicate a default card
        default_indicators = [
            "<!-- Provide a quick summary of what the model is/does. -->",
            "This is the model card of a 🤗 transformers model that has been pushed on the Hub. This model card has been automatically generated.",
            "[More Information Needed]",
            "<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->",
            "<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->",
            "<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->",
            "<!-- Provide the basic links for the model. -->",
            "## Model Card Contact"
        ]
        
        # Create a function to check if a card is default
        def is_default_card(card_text):
                
            # Check if at least 4 key phrases are present (this indicates a mostly default card)
            indicator_count = sum(1 for indicator in default_indicators if indicator in card_text)
            
            # Count number of "[More Information Needed]" occurrences
            more_info_needed_count = card_text.count("[More Information Needed]")
            
            # If the card contains many "[More Information Needed]" phrases or most default indicators,
            # consider it a default card
            return more_info_needed_count >= 38 and indicator_count >= 7
        
        # Discard models with the default card
        if is_default_card(model_info["card"]):
            return False
        
        return True