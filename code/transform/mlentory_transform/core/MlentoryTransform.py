from typing import List, Tuple, Dict, Any
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
from ..hf_transform.TransformHF import TransformHF
from .KnoledgeGraphHandler import KnowledgeGraphHandler

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
        self.kg_handler = KnowledgeGraphHandler()

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
        
        transform_hf = TransformHF(new_schema, transformations)

        transformed_df = transform_hf.transform_models(extracted_df)
        
        self.print_detailed_dataframe(transformed_df)
        
        #Transform the dataframe to a knowledge graph
        knowledge_graph = self.kg_handler.dataframe_to_graph_M4ML_schema(
            df=transformed_df,
            identifier_column="schema.org:name",
            platform="HF"
        )
        
        self.current_sources["HF_models"] = knowledge_graph
        
        return transformed_df

    def print_detailed_dataframe(self, df: pd.DataFrame):
        """
        Print the detailed dataframe
        """
        print("\n**DATAFRAME**")
        print("\nColumns:", df.columns.tolist())
        print("\nShape:", df.shape)
        print("\nSample Data:")
        for col in df.columns:
            print("--------------------------------------------")
            print(f"\n{col}:")
            for row in df[col]:
                # Limit the text to 100 characters
                if isinstance(row, list ):
                    row_data = row[0]["data"]
                    if isinstance(row_data, str):
                        print(row_data[:100])
                    else:
                        print(row_data)
                else:
                    print(row)
            print("--------------------------------------------")
            print()
        print("\nDataFrame Info:")
        print(df.info())
    
    
    
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