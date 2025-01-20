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
        print("api_token")
        print(api_token)
        if api_token:
            self.token = api_token
            self.api = HfApi(token=api_token)
        else:
            self.api = HfApi()

    def get_model_metadata_dataset(
        self, update_recent: bool = True, limit: int = 100
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

            if not update_recent:
                return dataset

            print("Dataset before update:")
            print(dataset.head(10))
            # Get the most recent modification date from the dataset
            latest_modification = dataset["last_modified"].max()

            recent_models = self.get_recent_models_metadata(limit, latest_modification)

            if len(recent_models) > 0:

                # Concatenate with original dataset and remove duplicates
                dataset = pd.concat([dataset, recent_models], ignore_index=True)
                dataset = dataset.drop_duplicates(subset=["modelId"], keep="last")

                # Sort by last_modified
                dataset = dataset.sort_values("last_modified", ascending=False)

                print("Dataset after update:")
                # Select the modelId and pipeline_tag columns
                print(dataset[["modelId", "pipeline_tag"]].head(10))

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
            limit=limit,
            sort="lastModified",
            direction=-1,
        )

        model_data = []

        for model in models:

            model_info = None
            card = None
            try:
                if self.token:
                    model_info = self.api.model_info(model.modelId, token=self.token)
                    card = ModelCard.load(model.modelId, token=self.token)
                else:
                    model_info = self.api.model_info(model.modelId)
                    card = ModelCard.load(model.modelId)
            except Exception as e:
                print()
                print(f"Error loading model card for {model.id}: {e}")

            if model_info == None:
                continue
            # Check if this model is newer than our latest modification
            if model_info.last_modified <= latest_modification:
                break

            model_data.append(
                {
                    "modelId": model_info.id,
                    "author": model_info.author,
                    "last_modified": model_info.last_modified,
                    "downloads": model_info.downloads,
                    "likes": model_info.likes,
                    "pipeline_tag": model_info.pipeline_tag,
                    "tags": model_info.tags,
                    "task_categories": model_info.pipeline_tag,
                    "createdAt": model_info.created_at,
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

    def save_to_file(
        self,
        df: pd.DataFrame,
        output_dir: str,
        filename: str,
        format: Literal["csv", "json", "parquet"] = "parquet",
    ) -> str:
        """
        Save the DataFrame to a file.

        Args:
            df (pd.DataFrame): DataFrame to save
            output_dir (str): Directory to save the file
            filename (str): Base filename without extension
            format (Literal["csv", "json", "parquet"]): Output format.
                Defaults to "parquet".

        Returns:
            str: Path to the saved file

        Raises:
            Exception: If there's an error saving the file
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(output_dir, f"{filename}_{timestamp}.{format}")

            if format == "csv":
                df.to_csv(filepath, index=False)
            elif format == "json":
                df.to_json(filepath, orient="records", indent=2)
            else:  # parquet
                df.to_parquet(filepath, index=False)

            return filepath

        except Exception as e:
            raise Exception(f"Error saving file: {str(e)}")
