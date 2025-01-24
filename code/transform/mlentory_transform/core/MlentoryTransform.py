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

    def __init__(self, kg_handler, transform_hf):
        """
        Initialize the transformer with schema and transformation rules.

        Args:
            kg_handler (KnowledgeGraphHandler): Knowledge graph handler
            transform_hf (TransformHF): Transform HF
        Example:
            >>> schema_df = pd.read_csv("schema.tsv", sep="\t")
            >>> transform_df = pd.read_csv("transformations.csv")
            >>> transformer = MlentoryTransform(schema_df, transform_df)
        """
        self.processed_data = []
        self.current_sources = {}
        self.kg_handler = kg_handler
        self.transform_hf = transform_hf

    def transform_HF_models(self,
                            extracted_df: pd.DataFrame,
                            save_output_in_json: bool = False,
                            output_dir: str = None) -> pd.DataFrame:
        """
        Transform the extracted data into a knowledge graph.

        This method:
        1. Processes each row of the input DataFrame
        2. Applies the specified transformations
        3. Optionally saves the results to a file
        4. Returns the transformed knowledge graph and metadata graph
        Args:
            extracted_df (pd.DataFrame): DataFrame containing extracted model data
            save_output_in_json (bool, optional): Whether to save the transformed data.
                Defaults to False.
            output_dir (str, optional): Directory to save the transformed data.
                Required if save_output_in_json is True.

        Returns:
            Tuple[rdflib.Graph, rdflib.Graph]: Transformed knowledge graph and metadata graph

        Raises:
            ValueError: If save_output_in_json is True but output_dir is not provided
        """
        # Reset the knowledge graph handler before processing new data
        self.kg_handler.reset_graphs()
        
        transformed_df = self.transform_hf.transform_models(extracted_df)
        
        # self.print_detailed_dataframe(transformed_df)
        
        #Transform the dataframe to a knowledge graph
        knowledge_graph, metadata_graph = self.kg_handler.dataframe_to_graph_M4ML_schema(
            df=transformed_df,
            identifier_column="schema.org:name",
            platform="HF"
        )
        
        self.current_sources["HF"] = knowledge_graph
        self.current_sources["HF_metadata"] = metadata_graph
        
        if save_output_in_json:
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            kg_output_path = os.path.join(
                output_dir, f"{current_date}_Transformed_HF_kg.json"
            )
            metadata_output_path = os.path.join(
                output_dir, f"{current_date}_Transformed_HF_kg_metadata.json"
            )
            knowledge_graph.serialize(destination=kg_output_path, format="json-ld")
            metadata_graph.serialize(destination=metadata_output_path, format="json-ld")
            
            
        return knowledge_graph, metadata_graph

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