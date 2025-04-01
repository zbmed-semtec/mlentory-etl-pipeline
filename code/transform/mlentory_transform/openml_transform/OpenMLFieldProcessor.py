import json
import pandas as pd
from typing import Callable, List, Dict, Any

class OpenMLFieldProcessor:
    """
    A class for processing fields from OpenML model data and mapping them to a target schema.

    This class provides functionality to:
    - Process individual fields using configurable transformations
    - Apply custom transformation functions to fields
    - Handle metadata extraction and formatting
    - Support batch processing of model data

    Attributes:
        FAIR4ML_schema (pd.DataFrame): Target schema for the transformed data
        transformations (pd.DataFrame): Mapping of source to target fields with transformation rules
        transformation_functions (dict): Available transformation functions
        current_row (pd.Series): Currently processed row of data
    """

    def __init__(self, new_schema: pd.DataFrame, transformations: pd.DataFrame):
        """
        Initialize the OpenMLFieldProcessor with configuration data.

        Args:
            new_schema (pd.DataFrame): The target schema to transform data into
            transformations (pd.DataFrame): The transformation rules to apply
        """
        self.M4ML_schema = new_schema
        self.transformations = transformations
        # Map of available transformation functions
        self.transformation_functions = {
            "find_value_in_OpenML": self.find_value_in_OpenML,
            "format_id": self.format_id, 
            "construct_dataset_object": self.construct_dataset_object,
            "construct_profile_object": self.construct_profile_object
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

    def find_value_in_OpenML(self, property_name: str):
        """
        Find the value of a property in a OpenML object.

        Args:
            property_name (str): The name of the property to find

        Returns:
            str: The value of the property
        """
        if property_name not in self.current_row:
            print(f"Property '{property_name}' not found in the current row.")
            return None
        
        return self.current_row.get(property_name)
         
    
    def format_id(self, property_name:str):
        """
        Format the value of the specified property as 'Run_{value}'.

        Args:
            property_name (str): The name of the property to retrieve and format.

        Returns:
            str: The formatted value as 'Run_{property_value}'.
        """
        if property_name not in self.current_row:
            print(f"Property '{property_name}' not found in the current row.")
            return None

        property_value = self.current_row.get(property_name)

        return f"Run_{property_value}"
    
    def construct_profile_object(self, property_names:List[str]):
        """
        Constructs the profile object of the given property

        Args:
            property_names (List[str]): The list of the properties to retrieve and format.

        Returns:
            dict: The constructed profile object.
        """

        for property_name in property_names: 
            if property_name not in self.current_row:
                print(f"Property '{property_name}' not found in the current row.")
                return None
            
        name_property, profile_property = property_names

        name_value = self.current_row.get(name_property)
        profile_value = self.current_row.get(profile_property)
            
        return {
            "name": name_value,
            "profile": f"https://www.openml.org/u/{profile_value}"
        }
    
    def construct_dataset_object(self, property_names:List[str]):
        """
        Constructs the dataset object of the given property

        Args:
            property_names (List[str]): The list of the properties to retrieve and format.

        Returns:
            dict: The constructed dataset object.
        """
        for property_name in property_names: 
            if property_name not in self.current_row:
                print(f"Property '{property_name}' not found in the current row.")
                return None
            
        datasetName_property, datasetPage_property, estimationProcedure_property = property_names

        datasetName = self.current_row.get(datasetName_property)
        datasetPage = self.current_row.get(datasetPage_property)
        estimationProcedure = self.current_row.get(estimationProcedure_property)

        return {
            "datasetName" : datasetName, 
            "datasetPage": datasetPage, 
            "estimationProcedure" : estimationProcedure
        }
