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
    This class processes the fields of the incoming tsv files and maps it to the M4ML schema.
    """

    def __init__(self, new_schema: pd.DataFrame, transformations: pd.DataFrame):
        """
        Initialize the FieldProcessorHF with configuration data.
        
        Args:
            new_schema (pd.DataFrame): The new schema to be used
            transformations (pd.DataFrame): The transformations to be applied
        """
        self.M4ML_schema = new_schema
        self.transformations = transformations
        # Map of available transformation functions
        self.transformation_functions = {
            'find_value_in_HF': self.find_value_in_HF,
            'build_HF_link': self.build_HF_link,
            'process_trainedOn': self.process_trainedOn,
            'process_softwareRequirements': self.process_softwareRequirements,
            'process_not_extracted': self.process_not_extracted
        }
        
        self.current_row = None
    
    def process_row(self, row: pd.Series) -> pd.Series:
        """
        Process a row using the transformation mappings.
        """
        result = pd.Series()
        
        # Apply each transformation
        for _, transform in self.transformations.iterrows():
            target_column = transform['target_column']
            result[target_column] = self.apply_transformation(row, transform)
            
        return result

    def apply_transformation(self, row: pd.Series, transformation: pd.Series) -> Any:
        """
        Apply a transformation to a row based on the transformation configuration.
        
        Args:
            row: Input row containing source data
            transformation: Series containing transformation configuration
        Returns:
            Transformed value
        """
        self.current_row = row
        func_name = transformation['transformation_function']
        func = self.transformation_functions[func_name]
        
        # Parse parameters if they exist
        params = {}
        if pd.notna(transformation['parameters']):
            params = json.loads(transformation['parameters'])
        
        return func(**params)


    def find_value_in_HF(self, property_name: str):
        """
        Find the value of a property in a HF object.
        Args:
            property_name (str): The name of the property to find
        Returns:
            str: The value of the property
        """

        prefix = property_name
        column_with_prefix = list(filter(lambda x: x.startswith(prefix), self.current_row.index))
        processed_value = self.current_row[column_with_prefix[0]]
        return processed_value

    def build_HF_link(self, tail_info: str) -> str:
        """
        Build the distribution link of a HF model.
        """

        model_name = self.find_value_in_HF("q_id_0")[0]["data"]
        link = "https://huggingface.co/" + model_name + tail_info
        # print("Link: ",link)
        return [self.add_default_extraction_info(link, "Built in transform stage", 1.0)]


    def process_softwareRequirements(self) -> List:

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
        Process the trainedOn property of a HF object.
        To process this proper we take into account 3 different values.
        1. Q4 What datasets was the model trained on?
        2. Q6 What datasets were used to finetune the model?
        3. Q7 What datasets were used to retrain the model?

        Return:
            str -- A string representing the list of datasets used to train the model.
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
        return self.add_default_extraction_info(
            data="Not extracted",
            extraction_method="None",
            confidence=1.0,
        )

    def add_default_extraction_info(
        self, data: str, extraction_method: str, confidence: float
    ) -> Dict:
        return {
            "data": data,
            "extraction_method": extraction_method,
            "confidence": confidence,
            "extraction_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }
