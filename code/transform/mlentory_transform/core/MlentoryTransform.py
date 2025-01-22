from typing import List, Tuple, Dict, Any
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
from hf_transform.FieldProcessorHF import FieldProcessorHF

class MlentoryTransform:
    """
    A class for transforming data from different sources into a unified knowledge graph.
    
    This class provides functionality to:
    - Transform data from multiple sources (HF, OpenML, etc.)
    - Standardize data formats
    - Handle metadata extraction
    - Create unified knowledge representations
    
    Attributes:
        sources (List[Tuple[str, pd.DataFrame]]): List of data sources and their dataframes
        schema (pd.DataFrame): Target schema for the transformed data
        transformations (pd.DataFrame): Mapping rules for data transformation
    """

    def __init__(self):
        """
        Initialize the transformer with schema and transformation rules.

        Args:
            schema (pd.DataFrame): Target schema for data transformation
            transformations (pd.DataFrame): Rules for mapping source fields to target schema

        Example:
            >>> schema_df = pd.read_csv("schema.tsv", sep="\t")
            >>> transform_df = pd.read_csv("transformations.csv")
            >>> transformer = MlentoryTransform(schema_df, transform_df)
        """
        self.processed_data = []
        self.current_sources = {}

    def transform_HF_models(self,
                            new_schema: pd.DataFrame,
                            transformations: pd.DataFrame,
                            extracted_df: pd.DataFrame,
                            save_output_in_json: bool = False,
                            output_dir: str = None) -> pd.DataFrame:
        """
        Transform the extracted data into the target schema.

        This method:
        1. Processes each row of the input DataFrame
        2. Applies the specified transformations
        3. Optionally saves the results to a file

        Args:
            new_schema (pd.DataFrame): Target schema for the transformed data
            transformations (pd.DataFrame): Transformation rules to apply
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
        
        fields_processor = FieldProcessorHF(new_schema, transformations)
        
        if save_output_in_json and output_dir is None:
            raise ValueError("output_dir must be provided when save_output is True")

        processed_models = []

        for row_num, row in tqdm(
            extracted_df.iterrows(),
            total=len(extracted_df),
            desc="Transforming progress",
        ):
            model_data = fields_processor.process_row(row)
            processed_models.append(model_data)

        transformed_df = pd.DataFrame(list(processed_models))
        
        #Transform the dataframe to a knowledge graph
        knowledge_graph = self.transform_df_to_knowledge_graph(transformed_df)
        
        self.current_sources["HF_models"] = knowledge_graph
        
        return transformed_df

    def save_indiviual_sources(self, output_dir: str):
        """
        Save each transformed source in a separate file in json format.
        """
        current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        
        for source in self.current_sources.keys():
            
            output_path = os.path.join(
                output_dir, f"{current_date}_{source}_transformation_result.json"
            )
            
            self.current_sources[source].to_json(output_path, index=False)
    
    def unify_knowledge_graph(self):
        """
        Unify the knowledge graph from the current sources.
        """
        pass