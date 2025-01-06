from extractors.core.ModelCardQAParser import ModelCardQAParser
from typing import Any, Dict, List, Set, Union
from datasets import load_dataset
from datetime import datetime
import pandas as pd
import os

class HFExtractor:
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
        Initialize the HuggingFace extractor
        
        Args:
            qa_model (str): The model to use for text extraction
            questions (list[str]): List of questions for extraction
            tags_language (list[str]): List of language tags
            tags_libraries (list[str]): List of library tags
            tags_other (list[str]): List of other tags
            tags_task (list[str]): List of task tags
        """
        self.parser = ModelCardQAParser(
            qa_model=qa_model,
            questions=questions,
            tags_language=tags_language,
            tags_libraries=tags_libraries,
            tags_other=tags_other,
            tags_task=tags_task,
        )
    
    def get_hf_dataset(self):
        return load_dataset("librarian-bots/model_cards_with_metadata")["train"].to_pandas()
    
    def download_models(
        self, 
        num_models: int = 10, 
        questions: List[str] = None,
        output_dir: str = "./outputs",
        save_original: bool = True,
        save_result_in_json: bool = True
    ) -> pd.DataFrame:
        """
        Download and process model cards from HuggingFace
        
        Args:
            num_models (int): Number of models to process
            questions (List[str]): List of questions to use for extraction. If None, uses default questions
            output_dir (str): Directory to save the output files
            save_original (bool): Whether to save the original dataset
            save_result_in_json (bool): Whether to download the dataset in json format in the output directory
            
        Returns:
            pd.DataFrame: Processed dataframe with extracted information
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
        HF_df = self.parser.parse_fields_from_txt_HF(HF_df=HF_df)
        
        
        # Clean up columns
        HF_df = HF_df.drop(columns=[
            "modelId", "author", "last_modified", "downloads",
            "likes", "library_name", "tags", "pipeline_tag",
            "createdAt", "card"
        ])
        
        
        # Improve column naming
        HF_df.columns = HF_df.columns.map(self._augment_column_name)
        
        # Save results
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        if save_original:
            original_path = os.path.join(output_dir, f"{timestamp}_Original_HF_Dataframe.csv")
            original_HF_df.to_csv(original_path, sep="\t")
        
        if save_result_in_json:
            processed_path = os.path.join(output_dir, f"{timestamp}_Processed_HF_Dataframe.json")
            HF_df.to_json(path_or_buf=processed_path, orient="records", indent=4)
        
        return HF_df
    
    def _augment_column_name(self, name: str) -> str:
        """Add question text to column names for better readability"""
        if "q_id" in name:
            num_id = int(name.split("_")[2])
            return name + "_" + self.parser.questions[num_id]
        return name 