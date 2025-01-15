from typing import Any, Dict, List, Set, Union
from datasets import load_dataset
from datetime import datetime
import pandas as pd
import os

from mlentory_extract.core.ModelCardQAParser import ModelCardQAParser


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
    """

    def __init__(
        self,
        qa_model: str = "Intel/dynamic_tinybert",
        questions: List[str] = None,
        tags_language: List[str] = None,
        tags_libraries: List[str] = None,
        tags_other: List[str] = None,
        tags_task: List[str] = None,
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
        """
        self.parser = ModelCardQAParser(
            qa_model=qa_model,
            questions=questions,
            tags_language=tags_language,
            tags_libraries=tags_libraries,
            tags_other=tags_other,
            tags_task=tags_task,
        )

    def get_hf_dataset(self) -> pd.DataFrame:
        """
        Retrieve the HuggingFace dataset containing model card information.

        Returns:
            pd.DataFrame: DataFrame containing model card information from HuggingFace
        """
        return load_dataset("librarian-bots/model_cards_with_metadata")[
            "train"
        ].to_pandas()

    def download_models(
        self,
        num_models: int = 10,
        questions: List[str] = None,
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
            questions (List[str], optional): Custom questions for extraction.
                Defaults to None.
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
        original_HF_df = self.get_hf_dataset()

        # Slice dataframe if num_models specified
        HF_df = original_HF_df.iloc[0:num_models] if num_models else original_HF_df

        # Update parser questions if custom questions provided
        if questions:
            self.parser.questions = questions

        # Create new columns for each question
        new_columns = {
            f"q_id_{idx}": [None for _ in range(len(HF_df))]
            for idx in range(len(self.parser.questions))
        }
        HF_df = HF_df.assign(**new_columns)

        print("**CHECKING COLUMNS**")
        print(HF_df.columns)

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
                "library_name",
                "tags",
                "pipeline_tag",
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
                output_dir, f"{timestamp}_Processed_HF_Dataframe.json"
            )
            HF_df.to_json(path_or_buf=processed_path, orient="records", indent=4)

        return HF_df

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
