from typing import Optional, Union, Literal, Dict, List
from datasets import load_dataset
import pandas as pd
from huggingface_hub import HfApi, ModelCard
from datetime import datetime
import requests
import itertools
import arxiv
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
            logger.info(f"Loading models from HuggingFace dataset")
            
            # Load base dataset
            # If this section ever freezes, just delete the cache and try again.
            dataset = load_dataset(
                "librarian-bots/model_cards_with_metadata",
                revision="4e7edd391342ee5c182afd08a6f62bff38f44535",
            )["train"].to_pandas()
            
            logger.info(f"Loaded {len(dataset)} models from HuggingFace dataset")

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
    
    def get_specific_models_metadata(
        self, model_ids: List[str], threads: int = 2
    ) -> pd.DataFrame:
        """
        Retrieve metadata for specific HuggingFace models by ID.

        This method processes a list of model IDs, retrieving their metadata from HuggingFace.

        Args:
            model_ids (List[str]): List of model IDs to retrieve metadata for
            threads (int, optional): Number of threads to use for downloading.
                Defaults to 4.

        Returns:
            pd.DataFrame: DataFrame containing model metadata for the requested models
        """
        model_data = []
        futures = []
        
        def process_model(model_id: str):
            
            models_to_process = self.api.list_models(model_name=model_id,limit=1,full=True)
            results = []
            
            for model in models_to_process:
                card = None
                try:
                    if self.token:
                        card = ModelCard.load(model.modelId, token=self.token)
                    else:
                        card = ModelCard.load(model.modelId)
                except Exception as e:
                    print(f"Error loading model card for {model_id}: {e}")
                    continue
                
                # model_id = model.id
                # if model_id is None:
                #     model_id = 
                
                # print("\n\n\n =================== \n\n\n")
                # print(f"Model ID: {model.id}")
                # print(f"Model card: {card.content}")
                # print("\n\n\n =================== \n\n\n")
                
                
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
                    results.append(model_info)
            return results

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(process_model, model_id) for model_id in model_ids]
            for future in as_completed(futures):
                results = future.result()
                if len(results) > 0:
                    model_data.extend(results)
                
        model_data = pd.DataFrame(model_data)
        # print(f"Model data!!!!!!!!: {model_data}")
        model_data = model_data.drop_duplicates(subset=["modelId"], keep="last")
        return model_data
                
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

    def get_specific_datasets_metadata(
        self, 
        dataset_names: List[str], 
        threads: int = 4
    ) -> pd.DataFrame:
        """
        Retrieve metadata for specific HuggingFace datasets by name in the croissant format.
        
        This method processes a list of dataset names, retrieving their metadata from HuggingFace.
        It differs from get_datasets_metadata as it targets specific datasets rather than
        a number of recent ones.

        Args:
            dataset_names (List[str]): List of dataset names/IDs to process.
            threads (int, optional): Number of threads to use for downloading.
                Defaults to 4.

        Returns:
            pd.DataFrame: DataFrame containing dataset metadata for the requested datasets
                
        Raises:
            ValueError: If the dataset_names parameter is empty
            
        Example:
            >>> manager = HFDatasetManager()
            >>> # Get metadata for specific datasets
            >>> dataset_df = manager.get_specific_datasets_metadata(
            ...     dataset_names=["squad", "glue", "mnist"]
            ... )
            >>> # Check the dataset IDs in the result
            >>> dataset_df["datasetId"].tolist()
            ['squad', 'glue', 'mnist']
        """
        if not dataset_names:
            raise ValueError("dataset_names list cannot be empty")
            
        dataset_data = []
        futures = []
        
        def process_dataset(dataset_id: str):
            try:
                croissant_metadata = self.get_croissant_metadata(dataset_id)
                if croissant_metadata == {}:
                    print(f"Warning: No croissant metadata found or access restricted for dataset '{dataset_id}'")
                    return None
                    
                return {
                    "datasetId": dataset_id,
                    "croissant_metadata": croissant_metadata,
                    "extraction_metadata": {
                        "extraction_method": "Downloaded_from_HF_Croissant_endpoint",
                        "confidence": 1.0,
                        "extraction_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    },
                }
            except Exception as e:
                print(f"Error processing dataset '{dataset_id}': {str(e)}")
                return None
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Submit all tasks
            for dataset_id in dataset_names:
                future = executor.submit(process_dataset, dataset_id)
                futures.append(future)
            
            # Process results as they complete
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    dataset_data.append(result)
        
        return pd.DataFrame(dataset_data)

    def get_specific_arxiv_metadata_dataset(
        self,
        arxiv_ids: List[str],
        batch_size: int = 200,
    ) -> pd.DataFrame:
        """
        Retrieve metadata for specific arXiv IDs using the arXiv API.

        This method takes a list of arXiv IDs and retrieves metadata for each paper,
        including title, authors, abstract, categories, and other information.
        Processing is done in batches with a 4-second delay between batches.

        Args:
            arxiv_ids (List[str]): List of arXiv IDs to retrieve metadata for
            batch_size (int): Number of arXiv IDs to process per batch. Defaults to 5000.

        Returns:
            pd.DataFrame: DataFrame containing arXiv metadata for the requested papers

        Raises:
            ValueError: If the arxiv_ids parameter is empty or invalid
            ImportError: If the arxiv package is not installed

        Example:
            >>> manager = HFDatasetManager()
            >>> arxiv_df = manager.get_specific_arxiv_metadata_dataset(
            ...     arxiv_ids=["2106.09685", "1706.03762"],
            ...     batch_size=1000
            ... )
            >>> print(arxiv_df.columns)
            ['arxiv_id', 'title', 'published', 'updated', 'summary', 'authors', ...]
        """
        temp_arxiv_ids = []
        
        for arxiv_id in arxiv_ids:
            if "." in arxiv_id:
                arxiv_id = arxiv_id.split("/")[-1]
                temp_arxiv_ids.append(arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id)
        
        arxiv_ids = temp_arxiv_ids
        
        # Use the arXiv API to search for the specific IDs in batches
        client = arxiv.Client(page_size = batch_size)
        arxiv_data = []
        
        # Process arXiv IDs in batches
        for i in range(0, len(arxiv_ids), batch_size):
            batch_ids = arxiv_ids[i:i + batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(arxiv_ids) + batch_size - 1)//batch_size} with {len(batch_ids)} arXiv IDs")
            
            search = arxiv.Search(
                id_list=batch_ids,
                max_results=batch_size
            )
            
            results = list(client.results(search))

            # Process arXiv papers sequentially for this batch
            for paper in results:
                try:
                    # Extract all available authors with affiliations
                    arxiv_id = str(paper).split("/")[-1].strip()
                    
                    authors_data = []
                    if hasattr(paper, "authors") and paper.authors:
                        for author in paper.authors:
                            author_name = author.name if hasattr(author, "name") else str(author)
                            affiliation = None
                            # The arXiv API through this package doesn't directly provide affiliations
                            # This would require additional processing if needed
                            authors_data.append({"name": author_name, "affiliation": affiliation})
                    
                    # Extract categories
                    categories = []
                    if hasattr(paper, "categories") and paper.categories:
                        categories = paper.categories
                    
                    # Process links
                    links = []
                    if hasattr(paper, "links") and paper.links:
                        for link in paper.links:
                            if isinstance(link, dict) and "href" in link:
                                links.append(link["href"])
                            elif hasattr(link, "href"):
                                links.append(link.href)
                            else:
                                links.append(str(link))
                    
                    # Extract DOI if available
                    doi = paper.doi if hasattr(paper, "doi") and paper.doi else None
                    
                    # Extract journal reference if available
                    journal_ref = paper.journal_ref if hasattr(paper, "journal_ref") and paper.journal_ref else None
                    
                    # Extract comment if available
                    comment = paper.comment if hasattr(paper, "comment") and paper.comment else None
                    
                    # Determine primary category
                    primary_category = categories[0] if categories else None
                    
                    # Parse dates
                    published = paper.published.strftime("%Y-%m-%d") if paper.published else None
                    updated = paper.updated.strftime("%Y-%m-%d") if paper.updated else None
                    
                    # Build the paper metadata dictionary
                    paper_metadata = {
                        "arxiv_id": arxiv_id,
                        "title": paper.title,
                        "published": published,
                        "updated": updated,
                        "summary": paper.summary,
                        "authors": authors_data,
                        "categories": categories,
                        "primary_category": primary_category,
                        "comment": comment,
                        "journal_ref": journal_ref,
                        "doi": doi,
                        #Ask about this urls
                        "links": links,
                        "pdf_url": paper.pdf_url if hasattr(paper, "pdf_url") else None,
                        "extraction_metadata": {
                            "extraction_method": "arXiv_API",
                            "confidence": 1.0,
                            "extraction_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        },
                    }
                    
                    arxiv_data.append(paper_metadata)
                    
                except Exception as e:
                    print(f"Error processing arXiv paper '{arxiv_id}': {str(e)}")
                    import traceback
                    print(f"Full stack trace: {traceback.format_exc()}")
            
            # Wait 4 seconds between batches (except for the last batch)
            if i + batch_size < len(arxiv_ids):
                print(f"Waiting 4 seconds before processing next batch...")
                time.sleep(5)
        
        if not arxiv_data:
            print("No arXiv papers could be successfully retrieved")
            return pd.DataFrame()
            
        return pd.DataFrame(arxiv_data)

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