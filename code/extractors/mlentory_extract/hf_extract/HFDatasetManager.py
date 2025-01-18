from typing import Optional, Union, Literal, Dict, List
from datasets import load_dataset
import pandas as pd
from huggingface_hub import HfApi
from datetime import datetime
import os


class HFDatasetManager:
    """
    A class for managing HuggingFace dataset and model information.
    
    This class handles direct interactions with the HuggingFace platform,
    including downloading and creating datasets for both models and datasets information.
    
    Attributes:
        api (HfApi): HuggingFace API client instance
        model_cards_dataset (str): Name of the default model cards dataset
    """

    def __init__(
        self,
        model_cards_dataset: str = "librarian-bots/model_cards_with_metadata",
        api_token: Optional[str] = None,
    ):
        """
        Initialize the HuggingFace Dataset Manager.

        Args:
            model_cards_dataset (str): Name of the dataset containing model cards.
                Defaults to "librarian-bots/model_cards_with_metadata".
            api_token (Optional[str]): HuggingFace API token for authenticated requests.
                Defaults to None.

        Raises:
            ValueError: If the model_cards_dataset is invalid or inaccessible
        """
        self.api = HfApi(token=api_token)
        self.model_cards_dataset = model_cards_dataset

    def get_model_cards_dataset(self) -> pd.DataFrame:
        """
        Retrieve the HuggingFace dataset containing model card information.

        Returns:
            pd.DataFrame: DataFrame containing model card information

        Raises:
            Exception: If there's an error loading the dataset
        """
        try:
            return load_dataset(self.model_cards_dataset)["train"].to_pandas()
        except Exception as e:
            raise Exception(
                f"Error loading model cards dataset {self.model_cards_dataset}: {str(e)}"
            )

    def fetch_models(
        self,
        limit: Optional[int] = None,
        filter_criteria: Optional[Dict] = None,
        author: Optional[str] = None,
        search: Optional[str] = None,
        sort: Optional[str] = "lastModified",
        direction: Optional[Literal[-1, 1]] = -1,
        full_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch model information directly from the HuggingFace API.

        Args:
            limit (Optional[int]): Maximum number of models to fetch. Defaults to None.
            filter_criteria (Optional[Dict]): Filter criteria for models. Defaults to None.
            author (Optional[str]): Filter by model author. Defaults to None.
            search (Optional[str]): Search query string. Defaults to None.
            sort (Optional[str]): Sort field. Defaults to "lastModified".
            direction (Optional[Literal[-1, 1]]): Sort direction (-1 for desc, 1 for asc).
                Defaults to -1.
            full_metadata (bool): Whether to fetch full model metadata. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame containing model information

        Raises:
            Exception: If there's an error fetching data from the API
        """
        try:
            models = self.api.list_models(
                limit=limit,
                filter=filter_criteria,
                author=author,
                search=search,
                sort=sort,
                direction=direction,
            )
            
            model_data = []
            for model in models:
                model_info = model.to_dict()
                if full_metadata:
                    try:
                        model_info.update(
                            self.api.model_info(model.modelId).to_dict()
                        )
                    except Exception:
                        pass  # Skip additional metadata if unavailable
                model_data.append(model_info)

            df = pd.DataFrame(model_data)
            df["fetch_date"] = datetime.now()
            return df

        except Exception as e:
            raise Exception(f"Error fetching models from API: {str(e)}")

    def fetch_datasets(
        self,
        limit: Optional[int] = None,
        filter_criteria: Optional[Dict] = None,
        author: Optional[str] = None,
        search: Optional[str] = None,
        sort: Optional[str] = "lastModified",
        direction: Optional[Literal[-1, 1]] = -1,
        full_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch dataset information directly from the HuggingFace API.

        Args:
            limit (Optional[int]): Maximum number of datasets to fetch. Defaults to None.
            filter_criteria (Optional[Dict]): Filter criteria for datasets. Defaults to None.
            author (Optional[str]): Filter by dataset author. Defaults to None.
            search (Optional[str]): Search query string. Defaults to None.
            sort (Optional[str]): Sort field. Defaults to "lastModified".
            direction (Optional[Literal[-1, 1]]): Sort direction (-1 for desc, 1 for asc).
                Defaults to -1.
            full_metadata (bool): Whether to fetch full dataset metadata. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame containing dataset information

        Raises:
            Exception: If there's an error fetching data from the API
        """
        try:
            datasets = self.api.list_datasets(
                limit=limit,
                filter=filter_criteria,
                author=author,
                search=search,
                sort=sort,
                direction=direction,
            )
            
            dataset_data = []
            for dataset in datasets:
                dataset_info = dataset.to_dict()
                if full_metadata:
                    try:
                        dataset_info.update(
                            self.api.dataset_info(dataset.id).to_dict()
                        )
                    except Exception:
                        pass  # Skip additional metadata if unavailable
                dataset_data.append(dataset_info)

            df = pd.DataFrame(dataset_data)
            df["fetch_date"] = datetime.now()
            return df

        except Exception as e:
            raise Exception(f"Error fetching datasets from API: {str(e)}")

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