from typing import Any, Dict, List, Set, Union, Optional
from datasets import load_dataset
from datetime import datetime
import pandas as pd
import os

from mlentory_extract.core.ModelCardQAParser import ModelCardQAParser
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
        qa_model: str = "Intel/dynamic_tinybert",
        questions: List[str] = None,
        tags_language: List[str] = None,
        tags_libraries: List[str] = None,
        tags_other: List[str] = None,
        tags_task: List[str] = None,
        dataset_manager: Optional[HFDatasetManager] = None,
    ):
        """
        Initialize the HuggingFace extractor.

        Args:
            qa_model (str, optional): The model to use for text extraction.
                Defaults to "Intel/dynamic_tinybert".
            questions (List[str], optional): List of questions for extraction.
                Defaults to None.
            tags_language (List[str], optional): List of language tags.
                Defaults to None.
            tags_libraries (List[str], optional): List of library tags.
                Defaults to None.
            tags_other (List[str], optional): List of other tags.
                Defaults to None.
            tags_task (List[str], optional): List of task tags.
                Defaults to None.
            dataset_manager (Optional[HFDatasetManager], optional): Dataset manager instance.
                Defaults to None.
        """
        self.parser = ModelCardQAParser(
            qa_model=qa_model,
            questions=questions,
            tags_language=tags_language,
            tags_libraries=tags_libraries,
            tags_other=tags_other,
            tags_task=tags_task,
        )
        self.dataset_manager = dataset_manager or HFDatasetManager()

    def download_models(
        self,
        num_models: int = 10,
        update_recent: bool = True,
        output_dir: str = "./outputs",
        save_raw_data: bool = False,
        save_result_in_json: bool = False,
        from_date: str = None,
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
            update_recent=update_recent, limit=num_models
        )
        # Slice dataframe if num_models specified
        HF_df = original_HF_df

        # Create new columns for each question
        new_columns = {
            f"q_id_{idx}": [None for _ in range(len(HF_df))]
            for idx in range(len(self.parser.questions))
        }
        HF_df = HF_df.assign(**new_columns)

        # Parse fields
        HF_df = self.parser.parse_fields_from_tags_HF(HF_df=HF_df)
        HF_df = self.parser.parse_known_fields_HF(HF_df=HF_df)
        HF_df = self.parser.parse_fields_from_txt_HF_matching(HF_df=HF_df)
        

        # Clean up columns
        HF_df = HF_df.drop(
            columns=[
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
        )

        # Improve column naming
        HF_df.columns = HF_df.columns.map(self._augment_column_name)

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

    def download_datasets(self, num_datasets: int = 10, from_date: str = None, output_dir: str = "./outputs", save_result_in_json: bool = False, update_recent: bool = True):
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
        """
        
        result_df = self.dataset_manager.get_dataset_metadata_dataset(
            limit=num_datasets,
            latest_modification=from_date,
            threads=4
        )
        
        if save_result_in_json:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            processed_path = os.path.join(
                output_dir, f"{timestamp}_Extracted_Datasets_HF_df.json"
            )
            result_df.to_json(path_or_buf=processed_path, orient="records", indent=4)
        
        return result_df

    def print_detailed_dataframe(self, HF_df: pd.DataFrame):
        print("\n**DATAFRAME**")
        print("\nColumns:", HF_df.columns.tolist())
        print("\nShape:", HF_df.shape)
        print("\nSample Data:")
        for col in HF_df.columns:
            print(f"\n{col}:")
            for row in HF_df[col]:
                # Limit the text to 100 characters
                if isinstance(row, list ):
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
        
    def _augment_column_name(self, name: str) -> str:
        """
        Add question text to column names for better readability.

        This method transforms column names like 'q_id_0' to include the actual question text,
        making the output more human-readable.

        Args:
            name (str): Original column name

        Returns:
            str: Augmented column name including the question text if applicable
        """
        if "q_id" in name:
            num_id = int(name.split("_")[2])
            return name + "_" + self.parser.questions[num_id]
        return name
