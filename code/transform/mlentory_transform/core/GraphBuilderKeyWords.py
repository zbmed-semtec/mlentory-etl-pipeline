from typing import List, Optional, Union, Dict, Tuple, Any
import pandas as pd
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from ..utils.enums import ExtractionMethod
from .GraphBuilderBase import GraphBuilderBase

class GraphBuilderKeyWords(GraphBuilderBase):
    """
    A Knowledge Graph Handler specifically for keyword/tag data.

    Inherits common functionalities from BaseKnowledgeGraphHandler and implements
    methods to convert DataFrames containing keyword/tag information into an RDF graph,
    creating schema:DefinedTerm entities.

    Args:
        base_namespace (str): The base URI namespace for the knowledge graph entities.
            Default: "https://example.org/"
    """
    def __init__(self, base_namespace: str = "https://example.org/") -> None:
        """
        Initialize the KeywordsKnowledgeGraphHandler.

        Args:
            base_namespace (str): The base URI namespace.
        """
        super().__init__(base_namespace)

    def hf_dataframe_to_graph(
        self,
        df: pd.DataFrame,
        identifier_column: Optional[str] = None,
        platform: str = "MLentoryKeywords", # Specific platform for these terms
    ) -> Tuple[Graph, Graph]:
        """
        Convert a DataFrame containing keyword/tag data to a Knowledge Graph.

        Maps columns like 'tag_name' and 'description' to schema.org properties
        for DefinedTerm entities.

        Args:
            df (pd.DataFrame): The DataFrame with keyword data.
            identifier_column (Optional[str]): The column containing the unique identifier for the keyword/tag.
                                           Must be provided.
            platform (str): The platform name, defaults to "MLentoryKeywords".

        Returns:
            Tuple[Graph, Graph]: A tuple containing:
                - The Knowledge Graph created from the DataFrame
                - The Metadata Graph (assuming fixed metadata for these terms)

        Raises:
            ValueError: If the DataFrame is empty or the identifier column is missing/invalid.
        """
        if df.empty:
            return self.graph, self.metadata_graph

        if not identifier_column or identifier_column not in df.columns:
            raise ValueError(
                f"Identifier column '{identifier_column}' must be provided and exist in DataFrame for keywords schema."
            )

        # Fixed metadata for keywords collected by the MLentory team
        metadata_dict = {
             "extraction_method": ExtractionMethod.ETL.value,
             "confidence": 1.0
        }
        extraction_time = None # Or set a fixed date if applicable

        for idx, row in df.iterrows():
            entity_id = row[identifier_column]
            if not entity_id or not isinstance(entity_id, str):
                 print(f"Warning: Skipping row {idx} due to missing or invalid identifier '{entity_id}'.")
                 continue

            entity_id = entity_id.strip().lower()
            if not entity_id:
                 print(f"Warning: Skipping row {idx} due to empty identifier after stripping/lowercasing.")
                 continue

            id_hash = self.generate_entity_hash(platform, "DefinedTerm", entity_id)
            defined_term_uri = self.base_namespace[id_hash]

            # Add entity type
            self.add_triple_with_metadata(
                defined_term_uri,
                RDF.type,
                self.namespaces["schema"]["DefinedTerm"],
                metadata_dict,
                extraction_time
            )

            # Add tag name
            if "tag_name" in row and pd.notna(row["tag_name"]):
                self.add_triple_with_metadata(
                    defined_term_uri,
                    self.namespaces["schema"]["name"],
                    Literal(row["tag_name"], datatype=XSD.string),
                    metadata_dict,
                    extraction_time
                )
            else:
                 # Use the identifier as the name if tag_name is missing
                  self.add_triple_with_metadata(
                    defined_term_uri,
                    self.namespaces["schema"]["name"],
                    Literal(entity_id, datatype=XSD.string),
                    metadata_dict,
                    extraction_time
                )


            # Add description
            if "description" in row and pd.notna(row["description"]):
                self.add_triple_with_metadata(
                    defined_term_uri,
                    self.namespaces["schema"]["description"],
                    Literal(row["description"], datatype=XSD.string),
                     metadata_dict,
                     extraction_time
                )

        return self.graph, self.metadata_graph 