from typing import Any, Dict, List, Set, Union, Optional
from datasets import load_dataset
from datetime import datetime
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from mlentory_extract.core.ModelCardToSchemaParser import ModelCardToSchemaParser
from mlentory_extract.hf_extract.HFDatasetManager import HFDatasetManager


class HFExtractor:
    """
    A class for extracting and processing model information from HuggingFace.

    This class provides functionality to:
    - Download model information from HuggingFace
    - Process model cards using QA techniques
    - Extract structured information from model metadata
    - Save results in various formats

    Attributes:
        parser (ModelCardQAParser): Parser instance for extracting information
        dataset_manager (HFDatasetManager): Dataset manager instance
    """

    def __init__(
        self,
        parser: Optional[ModelCardToSchemaParser] = None,
        dataset_manager: Optional[HFDatasetManager] = None
    ):
        """
        Initialize the HuggingFace extractor.

        Args:
            parser (Optional[ModelCardToSchemaParser], optional): Parser instance for extracting information.
            dataset_manager (Optional[HFDatasetManager], optional): Dataset manager instance.
        """
        self.parser = parser or ModelCardToSchemaParser()
        self.dataset_manager = dataset_manager or HFDatasetManager()
        
    def download_models_with_related_entities(
        self,
        num_models: int = 10,
        update_recent: bool = True,
        related_entities_to_download: List[str] = ["datasets", "base_models", "licenses", "keywords", "articles"],
        output_dir: str = "./outputs",
        save_initial_data: bool = False,
        save_result_in_json: bool = False,
        from_date: str = None,
        threads: int = 4,
        depth: int = 1,
    ) -> pd.DataFrame:
        """
        Download models with all related entities specified, then repeat the process for the base models found.
        That process will be repeated depth times.
        """
        if depth <= 0:
            print("Warning: Depth is 0 or negative, no download will be performed")
            return {}
        
        extracted_entities = {}
        processed_models = list()
        processed_models_ids = set()
        related_entities_names = {}
        related_entities = {}
        models_to_process = list()
        
        for current_depth in range(depth):
            current_models_df = pd.DataFrame()
            if current_depth == 0:
                current_models_df = self.download_models(
                    num_models=num_models,
                    update_recent=update_recent,
                    output_dir=output_dir,
                    save_raw_data=save_initial_data,
                    save_result_in_json=save_result_in_json,
                )
            else:
                current_models_df = self.download_specific_models(
                    model_ids=models_to_process,
                    output_dir=output_dir,
                    save_result_in_json=save_result_in_json,
                )
                
            processed_models.append(current_models_df)
            for index, row in current_models_df.iterrows():
                processed_models_ids.add(row["schema.org:identifier"][0]["data"])
            
            current_related_entities_names = self.get_related_entities_names(current_models_df)
            
            models_to_process = list()
            
            for model_id in current_related_entities_names["base_models"]:
                if model_id not in processed_models_ids:
                    models_to_process.append(model_id)
            
            for key, value in current_related_entities_names.items():
                if key in related_entities_to_download:
                    if key not in related_entities:
                        related_entities[key] = value
                    else:
                        related_entities[key].update(value)
        
        #Concatenate all processed models
        models_df = pd.concat(processed_models, ignore_index=True)
        models_df = models_df.drop_duplicates(subset=["schema.org:identifier"], keep="last")
        
        # Download related entities
        extracted_entities = self.download_related_entities(
            related_entities=related_entities,
            related_entities_to_download=related_entities_to_download,
            output_dir=output_dir,
            save_result_in_json=save_result_in_json,
            threads=threads,
        )
        
        #merge base models with models
        models_df = pd.concat([models_df, extracted_entities["models"]], ignore_index=True)
        models_df = models_df.drop_duplicates(subset=["schema.org:identifier"], keep="last")
        
        extracted_entities["models"] = models_df
        
        return extracted_entities
    
    
    def download_models(
        self,
        num_models: int = 10,
        update_recent: bool = True,
        output_dir: str = "./outputs",
        save_raw_data: bool = False,
        save_result_in_json: bool = False,
        from_date: str = None,
        threads: int = 4,
    ) -> pd.DataFrame:
        """
        Download and process model cards from HuggingFace.

        This method performs the following steps:
        1. Downloads model card information from HuggingFace
        2. Processes the specified number of models
        3. Extracts information using the QA model
        4. Saves results in the specified format

        Args:
            num_models (int, optional): Number of models to process.
                Defaults to 10.
            update_recent (bool, optional): Whether to update recent models.
                Defaults to True.
            output_dir (str, optional): Directory to save output files.
                Defaults to "./outputs".
            save_raw_data (bool, optional): Whether to save original dataset.
                Defaults to False.
            save_result_in_json (bool, optional): Whether to save results as JSON.
                Defaults to True.
            from_date (str, optional): Filter models by date.
                Defaults to None.

        Returns:
            pd.DataFrame: Processed DataFrame containing extracted information

        """
        # Load dataset
        original_HF_df = self.dataset_manager.get_model_metadata_dataset(
            update_recent=update_recent, limit=num_models, threads=threads
        )
        
        # Parse fields
        HF_df = self.parser.process_dataframe(original_HF_df)
        
        # Save results
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if save_raw_data:
            original_path = os.path.join(
                output_dir, f"{timestamp}_Original_HF_Dataframe.csv"
            )
            original_HF_df.to_csv(original_path, sep="\t")

        if save_result_in_json:
            processed_path = os.path.join(
                output_dir, f"{timestamp}_Extracted_Models_HF_df.json"
            )
            HF_df.to_json(path_or_buf=processed_path, orient="records", indent=4)

        return HF_df
    
    def download_specific_models(
        self,
        model_ids: List[str],
        output_dir: str = "./outputs",
        save_result_in_json: bool = False,
        threads: int = 4,
    ) -> pd.DataFrame:
        """
        Download specific models from HuggingFace.
        """
        result_df = self.dataset_manager.get_specific_models_metadata(
            model_ids=model_ids, threads=threads
        )
        
        result_df = self.parser.process_dataframe(result_df)
        
        if save_result_in_json:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            processed_path = os.path.join(
                output_dir, f"{timestamp}_Extracted_Specific_Models_HF_df.json"
            )
            result_df.to_json(path_or_buf=processed_path, orient="records", indent=4)
        
        return result_df

    def get_related_entities_names(self, HF_models_df: pd.DataFrame) -> Dict[str, Set[str]]:
        """
        Get the related entities for all models in the dataframe.

        Args:
            HF_models_df (pd.DataFrame): The dataframe containing the models.

        Returns:
            Dict[str, List[str]]: A dictionary containing the related entities for each model.
        """
        
        related_entities = {}
        related_entities_names = {"datasets": ["fair4ml:trainedOn"],
                                  "base_models": ["fair4ml:fineTunedFrom"],
                                  "licenses": ["schema.org:license"],
                                  "keywords": ["schema.org:keywords","fair4ml:mlTask"],
                                  "articles": ["codemeta:referencePublication"]}
        
        for name in related_entities_names.keys():
            related_entities[name] = set()
        
        
        # Get all unique values for each column
        for index, row in HF_models_df.iterrows():
            for name in related_entities_names.keys():
                for property in related_entities_names[name]:
                    for list_item in row[property]:
                        if isinstance(list_item, dict) and "data" in list_item:
                            if isinstance(list_item["data"], list):
                                # Add each element from the list
                                related_entities[name].update(list_item["data"])
                            else:
                                # Add single item
                                related_entities[name].add(list_item["data"])
        
        return related_entities
    
    def download_related_entities(
        self,
        related_entities_to_download: List[str],
        related_entities: Dict[str, Set[str]],
        output_dir: str = "./outputs",
        save_result_in_json: bool = False,
        threads: int = 4,
    ) -> pd.DataFrame:
        
        extracted_entities = {}
        
        # Download base models
        if len(related_entities["base_models"]) > 0:
            # Use download_specific_datasets if base model names are provided
            extracted_base_models_df = self.download_specific_models(
                model_ids=related_entities["base_models"],
                output_dir=output_dir,
                save_result_in_json=False,
                threads=threads,
            )
            extracted_entities["models"] = extracted_base_models_df
            print(f"Downloaded {len(extracted_base_models_df)} base models")
        else:
            print("No base models found to download")
        
        
        
        if "datasets" in related_entities_to_download:
            # Download datasets
            if len(related_entities["datasets"]) > 0:
                # Use download_specific_datasets if dataset names are provided
                extracted_datasets_df = self.download_specific_datasets(
                    dataset_names=related_entities["datasets"],
                    output_dir=output_dir+"/datasets",
                    save_result_in_json=False,
                    threads=threads,
                )
                print(f"Downloaded {len(extracted_datasets_df)} datasets")
                extracted_entities["datasets"] = extracted_datasets_df
            else:
                print("No datasets found to download")
        
        
        if "articles" in related_entities_to_download:
            # Download arxiv articles
            if len(related_entities["articles"]) > 0:
                # Use download_specific_arxiv_metadata if arxiv ids are provided
                extracted_arxiv_df = self.download_specific_arxiv_metadata(
                    arxiv_ids=related_entities["articles"],
                    output_dir=output_dir+"/articles",
                    save_result_in_json=False,
                    threads=threads,
                )
                print(f"Downloaded {len(extracted_arxiv_df)} arxiv articles")
                extracted_entities["articles"] = extracted_arxiv_df
            else:
                print("No arxiv articles found to download")
        
            if "keywords" in related_entities_to_download:
                # Download keywords
                extracted_keywords_df = self.get_keywords()
                print(f"Processed {len(extracted_keywords_df)} keywords")
                extracted_entities["keywords"] = extracted_keywords_df
        
        return extracted_entities
    
    def download_datasets(
        self,
        num_datasets: int = 10,
        from_date: str = None,
        output_dir: str = "./outputs",
        save_result_in_json: bool = False,
        update_recent: bool = True,
        threads: int = 4,
    ) -> pd.DataFrame:
        """
        Download metadata from HuggingFace datasets in the croissant format.

        Args:
            num_datasets (int, optional): Number of datasets to process.
                Defaults to 10.
            from_date (str, optional): Filter datasets by date.
                Defaults to None.
            output_dir (str, optional): Directory to save output files.
                Defaults to "./outputs".
            save_result_in_json (bool, optional): Whether to save results as JSON.
                Defaults to True.
            update_recent (bool, optional): Whether to update recent datasets.
                Defaults to True.
            threads (int, optional): Number of threads to use for downloading.
                Defaults to 4.

        Returns:
            pd.DataFrame: Processed DataFrame containing extracted information
        """

        result_df = self.dataset_manager.get_datasets_metadata(
            limit=num_datasets, latest_modification=from_date, threads=threads
        )

        if save_result_in_json:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            processed_path = os.path.join(
                output_dir, f"{timestamp}_Extracted_Datasets_HF_df.json"
            )
            result_df.to_json(path_or_buf=processed_path, orient="records", indent=4)

        return result_df
        
    def download_specific_datasets(
        self,
        dataset_names: List[str],
        output_dir: str = "./outputs",
        save_result_in_json: bool = False,
        threads: int = 4,
    ) -> pd.DataFrame:
        """
        Download metadata for specific HuggingFace datasets by name in the croissant format.
        
        This method processes a list of dataset names, retrieving their metadata from HuggingFace.
        It differs from download_datasets as it targets specific datasets rather than
        a number of recent ones.

        Args:
            dataset_names (List[str]): List of dataset names/IDs to process.
            output_dir (str, optional): Directory to save output files.
                Defaults to "./outputs".
            save_result_in_json (bool, optional): Whether to save results as JSON.
                Defaults to True.
            threads (int, optional): Number of threads to use for downloading.
                Defaults to 4.

        Returns:
            pd.DataFrame: Processed DataFrame containing extracted information for the
                requested datasets
                
        Raises:
            ValueError: If the dataset_names parameter is empty
            
        Example:
            >>> extractor = HFExtractor()
            >>> # Download metadata for specific datasets
            >>> dataset_df = extractor.download_specific_datasets(
            ...     dataset_names=["squad", "glue", "mnist"],
            ...     output_dir="./outputs/datasets",
            ...     save_result_in_json=True
            ... )
            >>> # Check the dataset IDs in the result
            >>> dataset_df["datasetId"].tolist()
            ['squad', 'glue', 'mnist']
        """
        # Get dataset metadata using HFDatasetManager's method
        result_df = self.dataset_manager.get_specific_datasets_metadata(
            dataset_names=dataset_names,
            threads=threads
        )
        
        # Save results if requested
        if save_result_in_json and not result_df.empty:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            processed_path = os.path.join(
                output_dir, f"{timestamp}_Extracted_Specific_Datasets_HF_df.json"
            )
            result_df.to_json(path_or_buf=processed_path, orient="records", indent=4)
        
        return result_df

    def download_specific_arxiv_metadata(
        self,
        arxiv_ids: List[str],
        output_dir: str = "./outputs",
        save_result_in_json: bool = False,
        threads: int = 4,   
    ) -> pd.DataFrame:
        """
        Download metadata for specific arxiv ids.
        """
        result_df = self.dataset_manager.get_specific_arxiv_metadata_dataset(
            arxiv_ids=arxiv_ids
        )
        
        if save_result_in_json:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            processed_path = os.path.join(
                output_dir, f"{timestamp}_Extracted_Specific_Arxiv_Metadata_HF_df.json"
            )
            result_df.to_json(path_or_buf=processed_path, orient="records", indent=4)
        
        return result_df

    def get_keywords(self) -> pd.DataFrame:
        """
        Get keywords from HuggingFace.
        """
        keywords_df = pd.concat([self.parser.tags_other_df, self.parser.tags_task_df, self.parser.tags_libraries_df])
        return keywords_df
    
    def print_detailed_dataframe(self, HF_df: pd.DataFrame):

        print("\n**DATAFRAME**")
        print("\nColumns:", HF_df.columns.tolist())
        print("\nShape:", HF_df.shape)
        print("\nSample Data:")
        for col in HF_df.columns:
            print(f"\n{col}:")
            for row in HF_df[col]:
                # Limit the text to 100 characters
                if isinstance(row, list):
                    row_data = row[0]["data"]
                    if isinstance(row_data, str):
                        print(row_data[:100])
                    else:
                        print(row_data)
                else:
                    print(row)

            print()
        print("\nDataFrame Info:")
        print(HF_df.info())
