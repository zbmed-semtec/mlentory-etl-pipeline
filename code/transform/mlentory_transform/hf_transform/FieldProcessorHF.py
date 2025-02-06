from typing import Callable, List, Dict, Any
import logging
import time
import pandas as pd
import json
import ast
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FieldProcessorHF:
    """
    A class for processing fields from HuggingFace model data and mapping them to a target schema.

    This class provides functionality to:
    - Process individual fields using configurable transformations
    - Apply custom transformation functions to fields
    - Handle metadata extraction and formatting
    - Support batch processing of model data

    Attributes:
        M4ML_schema (pd.DataFrame): Target schema for the transformed data
        transformations (pd.DataFrame): Mapping of source to target fields with transformation rules
        transformation_functions (dict): Available transformation functions
        current_row (pd.Series): Currently processed row of data
    """

    def __init__(self, new_schema: pd.DataFrame, transformations: pd.DataFrame):
        """
        Initialize the FieldProcessorHF with configuration data.

        Args:
            new_schema (pd.DataFrame): The target schema to transform data into
            transformations (pd.DataFrame): The transformation rules to apply
        """
        self.M4ML_schema = new_schema
        self.transformations = transformations
        # Map of available transformation functions
        self.transformation_functions = {
            "find_value_in_HF": self.find_value_in_HF,
            "build_HF_link": self.build_HF_link,
            "process_trainedOn": self.process_trainedOn,
            "process_softwareRequirements": self.process_softwareRequirements,
            "process_not_extracted": self.process_not_extracted,
        }

        self.current_row = None

    def process_row(self, row: pd.Series) -> pd.Series:
        """
        Process a single row of data using the defined transformation mappings.

        Args:
            row (pd.Series): Input row containing source data

        Returns:
            pd.Series: Transformed row conforming to target schema
        """
        result = pd.Series()

        # Apply each transformation
        for _, transform in self.transformations.iterrows():
            target_column = transform["target_column"]
            result[target_column] = self.apply_transformation(row, transform)

        return result

    def apply_transformation(self, row: pd.Series, transformation: pd.Series) -> Any:
        """
        Apply a specific transformation to a row based on the transformation configuration.

        Args:
            row (pd.Series): Input row containing source data
            transformation (pd.Series): Series containing transformation configuration

        Returns:
            pd.DataFrame: Transformed value according to the specified transformation
        """
        self.current_row = row
        func_name = transformation["transformation_function"]
        func = self.transformation_functions[func_name]

        # Parse parameters if they exist
        params = {}
        if pd.notna(transformation["parameters"]):
            params = json.loads(transformation["parameters"])

        return func(**params)

    def find_value_in_HF(self, property_name: str):
        """
        Find the value of a property in a HuggingFace object.

        Args:
            property_name (str): The name of the property to find

        Returns:
            str: The value of the property
        """

        prefix = property_name
        column_with_prefix = list(
            filter(lambda x: x.startswith(prefix), self.current_row.index)
        )
        processed_value = self.current_row[column_with_prefix[0]]
        return processed_value

    def build_HF_link(self, tail_info: str) -> str:
        """
        Build the distribution link of a HuggingFace model.

        Args:
            tail_info (str): Additional path information to append to the base URL

        Returns:
            str: Complete HuggingFace model link
        """
        print("Checking building links: ",self.find_value_in_HF("q_id_0"),"\n")
        model_name = self.find_value_in_HF("q_id_0")[0]["data"]
        link = "https://huggingface.co/" + model_name + tail_info
        # print("Link: ",link)
        return [self.add_default_extraction_info(link, "Built in transform stage", 1.0)]

    def process_softwareRequirements(self) -> List:
        """
        Process software requirements for the model.

        Returns:
            List: List of software requirements with extraction metadata
        """

        q17_values = self.find_value_in_HF("q_id_17")

        values = [q17_values[0]]

        values.append(
            self.add_default_extraction_info(
                data="Python",
                extraction_method="Added in transform stage",
                confidence=1.0,
            )
        )

        return values

    def process_trainedOn(self) -> List:
        """
        Process the trainedOn property by aggregating the values from Q4, Q6, and Q7.

        Processes three different values:
        1. Q4: What datasets was the model trained on?
        2. Q6: What datasets were used to finetune the model?
        3. Q7: What datasets were used to retrain the model?

        Returns:
            List: Combined list of datasets used for training
        """
        q4_values = self.find_value_in_HF("q_id_4")
        q6_values = self.find_value_in_HF("q_id_6")
        q7_values = self.find_value_in_HF("q_id_7")

        processed_values = []

        processed_values.extend(q4_values)
        processed_values.extend(q6_values)
        processed_values.extend(q7_values)

        return processed_values

    def process_not_extracted(self) -> Dict:
        """
        Handle fields that couldn't be extracted.

        Returns:
            Dict: Default metadata for non-extracted fields
        """
        return [
            self.add_default_extraction_info(
                data="Not extracted",
                extraction_method="None",
                confidence=1.0,
            )
        ]

    def add_default_extraction_info(
        self, data: str, extraction_method: str, confidence: float
    ) -> Dict:
        """
        Create standardized metadata for extracted information.

        Args:
            data (str): The extracted information
            extraction_method (str): Method used for extraction
            confidence (float): Confidence score of the extraction

        Returns:
            Dict: Standardized metadata dictionary
        """
        return {
            "data": data,
            "extraction_method": extraction_method,
            "confidence": confidence,
            "extraction_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }
