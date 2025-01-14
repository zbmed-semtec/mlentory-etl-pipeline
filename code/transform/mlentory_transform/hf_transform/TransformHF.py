import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime

from .FieldProcessorHF import FieldProcessorHF


class TransformHF:
    """
    A class for transforming HuggingFace model metadata into a standardized schema.

    This class provides functionality to:
    - Transform extracted model data into a target schema
    - Apply field-level transformations
    - Save transformed data in various formats
    - Track transformation progress

    Attributes:
        new_schema (pd.DataFrame): Target schema for the transformed data
        transformations (pd.DataFrame): Transformation rules to apply
        fields_processor (FieldProcessorHF): Processor for field-level transformations
    """

    def __init__(self, new_schema: pd.DataFrame, transformations: pd.DataFrame):
        """
        Initialize the HuggingFace transformer with schema and transformations.

        Args:
            new_schema (pd.DataFrame): DataFrame containing the target schema
            transformations (pd.DataFrame): DataFrame containing column transformations
        """
        self.new_schema = new_schema
        self.transformations = transformations
        self.fields_processor = FieldProcessorHF(self.new_schema, self.transformations)

    def transform(
        self,
        extracted_df: pd.DataFrame,
        save_output_in_json: bool = False,
        output_dir: str = None,
    ) -> pd.DataFrame:
        """
        Transform the extracted data into the target schema.

        This method:
        1. Processes each row of the input DataFrame
        2. Applies the specified transformations
        3. Optionally saves the results to a file
        4. Shows progress using tqdm

        Args:
            extracted_df (pd.DataFrame): DataFrame containing extracted model data
            save_output_in_json (bool, optional): Whether to save the transformed data.
                Defaults to False.
            output_dir (str, optional): Directory to save the transformed data.
                Required if save_output_in_json is True.

        Returns:
            pd.DataFrame: Transformed DataFrame conforming to the target schema

        Raises:
            ValueError: If save_output_in_json is True but output_dir is not provided
        """
        if (save_output_in_json == True) and (output_dir is None):
            raise ValueError("output_dir must be provided when save_output is True")

        processed_models = []

        for row_num, row in tqdm(
            extracted_df.iterrows(),
            total=len(extracted_df),
            desc="Transforming progress",
        ):
            model_data = self.fields_processor.process_row(row)
            processed_models.append(model_data)

        transformed_df = pd.DataFrame(list(processed_models))

        if save_output_in_json:
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = os.path.join(
                output_dir, f"{current_date}_transformation_results.csv"
            )
            transformed_df.to_json(output_path, index=False)
            print(f"Saved transformation results to {output_path}")

        return transformed_df
