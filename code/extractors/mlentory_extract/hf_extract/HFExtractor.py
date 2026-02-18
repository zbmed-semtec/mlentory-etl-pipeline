from typing import Any, Dict, List, Set, Union, Optional
from datasets import load_dataset
from datetime import datetime
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import spdx_lookup
import logging

from mlentory_extract.core.ModelCardToSchemaParser import ModelCardToSchemaParser
from mlentory_extract.hf_extract.HFDatasetManager import HFDatasetManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        update_recent: bool = False,
        related_entities_to_download: List[str] = ["datasets", "base_models", "licenses", "keywords", "articles"],
        output_dir: str = "./outputs",
        save_initial_data: bool = False,
        save_result_in_json: bool = False,
        from_date: str = None,
        unstructured_text_strategy: str = "None",
        threads: int = 4,
        depth: int = 1,
        offset: int = 0,
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
        
        logger.info(f"Downloading models with related entities for depth {depth}")
        
        for current_depth in range(depth):
            current_models_df = pd.DataFrame()
            if current_depth == 0:
                current_models_df = self.download_models(
                    num_models=num_models,
                    update_recent=update_recent,
                    output_dir=output_dir,
                    save_raw_data=save_initial_data,
                    save_result_in_json=save_result_in_json,
                    unstructured_text_strategy=unstructured_text_strategy,
                    offset=offset,
                )
            else:
                current_models_df = self.download_specific_models(
                    model_ids=models_to_process,
                    output_dir=output_dir,
                    save_result_in_json=save_result_in_json,
                    unstructured_text_strategy=unstructured_text_strategy,
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
    
    def download_specific_models_with_related_entities(
        self,
        model_ids: List[str],
        related_entities_to_download: List[str] = ["datasets", "articles", "base_models", "keywords"],
        output_dir: str = "./outputs",
        save_result_in_json: bool = False,
        threads: int = 4,
        unstructured_text_strategy: str = "None",
    ) -> Dict[str, pd.DataFrame]:
        """
        Downloads specific models and their specified related entities.

        Args:
            model_ids (List[str]): A list of model IDs to download.
            related_entities_to_download (List[str], optional): List of related entity types to download.
                Defaults to ["datasets", "articles", "base_models", "keywords"].
            output_dir (str, optional): Directory to save output files. Defaults to "./outputs".
            save_result_in_json (bool, optional): Whether to save intermediate results as JSON.
                Defaults to False.
            threads (int, optional): Number of threads for parallel downloads. Defaults to 4.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where keys are entity types (e.g., "models", "datasets")
                                     and values are DataFrames of the extracted entities.
                                     Returns an empty dict if no models are processed.
        """
        extracted_entities = {}
        final_models_df = pd.DataFrame()

        if not model_ids:
            print("Warning: No model IDs provided to download_specific_models_with_related_entities.")
            return extracted_entities

        # 1. Download the specific models listed
        initial_models_df = self.download_specific_models(
            model_ids=model_ids,
            output_dir=os.path.join(output_dir, "models"), # Save models in their subdir
            save_result_in_json=save_result_in_json,
            threads=threads,
            unstructured_text_strategy=unstructured_text_strategy,
        )
        
        

        if initial_models_df.empty:
            print("Warning: No valid models could be downloaded from the provided list.")
            extracted_entities["models"] = pd.DataFrame() # Ensure models key exists
            return extracted_entities
        
        # Even if initial_models_df is empty, proceed to ensure the "models" key is in extracted_entities
        # This makes the downstream processing in the ETL script more robust.
        processed_models_ids = set()
        if not initial_models_df.empty:
            # Ensure 'schema.org:identifier' exists and is not empty before trying to access its elements
            if "schema.org:identifier" in initial_models_df.columns and not initial_models_df["schema.org:identifier"].empty:
                try:
                    processed_models_ids = set(initial_models_df["schema.org:identifier"].dropna().apply(lambda x: x[0]['data'] if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict) and 'data' in x[0] else None))
                    processed_models_ids.discard(None) # Remove None if any conversion failed
                except Exception as e:
                    print(f"Error processing schema.org:identifier: {e}. Processed_models_ids will be empty.")
            else:
                print("Warning: 'schema.org:identifier' column is missing or empty in initial_models_df.")
        
        final_models_df = initial_models_df.copy() if not initial_models_df.empty else pd.DataFrame()
        
        # 2. Find related entities for these initial models
        # Only proceed if there are models to get related entities from
        related_entities_names = {} # type: Dict[str, Set[str]]
        if not initial_models_df.empty:
            related_entities_names = self.get_related_entities_names(initial_models_df)
        else:
            # Initialize with empty sets if no initial models
            for entity_type in related_entities_to_download:
                related_entities_names[entity_type] = set()

        # Filter base models to only those not already downloaded
        # And ensure 'base_models' key exists even if not in related_entities_to_download initially for safety
        current_base_models_set = related_entities_names.get("base_models", set())
        base_models_to_download_set = {
            bm for bm in current_base_models_set
            if bm not in processed_models_ids
        }
        related_entities_names["base_models"] = base_models_to_download_set # Update the set

        # 3. Prepare to download the related entities
        entities_to_actually_download = {}
        for entity_type in related_entities_to_download:
            if entity_type in related_entities_names and related_entities_names[entity_type]:
                entities_to_actually_download[entity_type] = related_entities_names[entity_type]
        
        downloaded_related_entities = {} # type: Dict[str, pd.DataFrame]
        if entities_to_actually_download: # Only call if there's something to download
            
            downloaded_related_entities = self.download_related_entities(
                related_entities_to_download=list(entities_to_actually_download.keys()),
                related_entities=entities_to_actually_download, # Pass the filtered dict
                output_dir=output_dir, # Use base output_dir for subfolders like 'datasets', 'articles'
                save_result_in_json=save_result_in_json,
                threads=threads,
            )

        # 4. Combine results
        # Ensure final_models_df is a DataFrame, even if empty
        if not isinstance(final_models_df, pd.DataFrame):
            final_models_df = pd.DataFrame()
            
        if "models" in downloaded_related_entities and not downloaded_related_entities["models"].empty:
            final_models_df = pd.concat([final_models_df, downloaded_related_entities["models"]], ignore_index=True)\
                                    .drop_duplicates(subset=["schema.org:identifier"], keep="last")

        # Prepare the final extracted_entities dict
        extracted_entities = downloaded_related_entities
        extracted_entities["models"] = final_models_df
        
        # Ensure all requested entity types are keys in extracted_entities, even if empty
        for entity_type in related_entities_to_download:
            if entity_type not in extracted_entities:
                extracted_entities[entity_type] = pd.DataFrame()
        if "models" not in extracted_entities: # Crucial for downstream
             extracted_entities["models"] = pd.DataFrame()

        return extracted_entities
    
    def download_models(
        self,
        num_models: int = 10,
        update_recent: bool = False,
        output_dir: str = "./outputs",
        save_raw_data: bool = False,
        save_result_in_json: bool = False,
        from_date: str = None,
        unstructured_text_strategy: str = "None",
        threads: int = 4,
        offset: int = 0,
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
            unstructured_text_strategy (str, optional): Strategy to use for unstructured text extraction.
                Options: "context_matching", "grouped", "individual". Defaults to "None".

        Returns:
            pd.DataFrame: Processed DataFrame containing extracted information

        """
        
        # Load dataset
        original_HF_df = self.dataset_manager.get_model_metadata_dataset(
            update_recent=update_recent,
            limit=num_models,
            threads=threads,
            offset=offset,
        )
        
        logger.info(f"Downloaded {len(original_HF_df)} models from HuggingFace dataset")
        
        # Parse fields
        HF_df = self.parser.process_dataframe(original_HF_df, unstructured_text_strategy=unstructured_text_strategy)
        
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
        unstructured_text_strategy: str = "None",
        threads: int = 4,
    ) -> pd.DataFrame:
        """
        Download specific models from HuggingFace.
        """
        result_df = self.dataset_manager.get_specific_models_metadata(
            model_ids=model_ids, threads=threads
        )
        
        if result_df.empty:
            print("No models found to download")
            return pd.DataFrame()
        
        result_df = self.parser.process_dataframe(result_df, unstructured_text_strategy=unstructured_text_strategy)
        
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
        unstructured_text_strategy: str = "None",
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
                unstructured_text_strategy=unstructured_text_strategy,
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
        
        if "licenses" in related_entities_to_download:
            extracted_licenses_df = self.download_specific_spdx_licenses(related_entities["licenses"])
            print(f"Processed {len(extracted_licenses_df)} licenses")
            extracted_entities["licenses"] = extracted_licenses_df
        
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
    
    def download_specific_spdx_licenses(self, license_ids: List[str]) -> pd.DataFrame:
        """
        Get the SPDX license information for a given license string.

        Args:
            license_ids (List[str]): The license identifier string (e.g., "mit", "apache-2.0").

        Returns:
            pd.DataFrame: A DataFrame containing the license information. 
                          Columns include "Name", "Identifier", "OSI Approved", 
                          "Deprecated", "Notes", and "URL".
                          Returns an empty DataFrame if the license is not found.
        
        Example:
            >>> extractor = HFExtractor()
            >>> license_df = extractor.get_spdx_license_info("apache-2.0")
            >>> print(license_df)
        """
        all_license_data = []
        for license_id in license_ids:
            license_data = {
                "Name": license_id,
                "Identifier": None,
                "OSI Approved": None,
                "Deprecated": None,
                "Notes": None,
                "Text": None,
                "URL": None,
                "extraction_metadata": {"extraction_method": "Extracted from SPDX API", "confidence": 1.0}
            }
            # search by id and by name
            spdx_license_from_id = spdx_lookup.by_id(license_id)
            spdx_license_from_name = spdx_lookup.by_name(license_id)
            
            spdx_license = spdx_license_from_id or spdx_license_from_name

            if spdx_license:
                
                if hasattr(spdx_license, 'id'):
                    license_data["Identifier"] = spdx_license.id
                    url = f"https://spdx.org/licenses/{spdx_license.id}.html"
                    license_data["URL"] = url
                if hasattr(spdx_license, 'osi_approved'):
                    license_data["OSI Approved"] = spdx_license.osi_approved
                # Note: spdx_lookup might use 'sources' for deprecation info or other details
                # User should verify if 'sources' attribute correctly maps to 'Deprecated'
                if hasattr(spdx_license, 'sources'): 
                    license_data["Deprecated"] = spdx_license.sources
                if hasattr(spdx_license, 'notes'):
                    license_data["Notes"] = spdx_license.notes
                if hasattr(spdx_license, 'text'):
                    license_data["Text"] = spdx_license.text
                        
                all_license_data.append(license_data)
        
        return pd.DataFrame(all_license_data)

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

    