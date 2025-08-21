from typing import List, Optional, Union, Dict, Tuple, Any
import pandas as pd
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from .GraphBuilderBase import GraphBuilderBase

class GraphBuilderLicense(GraphBuilderBase):
    """
    A Knowledge Graph Handler specifically for License data using SPDX information.

    Inherits common functionalities from GraphBuilderBase and implements
    methods to convert DataFrames containing license metadata into an RDF graph,
    primarily creating schema:CreativeWork entities for licenses.

    Args:
        base_namespace (str): The base URI namespace for the knowledge graph entities.
            Default: "https://example.org/"
    """
    def __init__(self, base_namespace: str = "https://example.org/") -> None:
        """
        Initialize the GraphBuilderLicense.

        Args:
            base_namespace (str): The base URI namespace.
        """
        super().__init__(base_namespace)

    def hf_dataframe_to_graph(
        self,
        df: pd.DataFrame,
        identifier_column: Optional[str] = None,
        platform: str = "spdx",  # Default platform for license data
    ) -> Tuple[Graph, Graph]:
        """
        Convert a DataFrame containing SPDX license data to a Knowledge Graph.

        Maps columns like 'Name', 'Identifier', 'URL', 'Text' to
        schema.org properties for CreativeWork entities.

        Args:
            df (pd.DataFrame): The DataFrame with license data.
            identifier_column (Optional[str]): The column containing the license identifier (e.g., SPDX ID).
                                           If None, it might default or raise error depending on usage.
            platform (str): The platform name (e.g., "spdx", "huggingface"). Defaults to "spdx".

        Returns:
            Tuple[Graph, Graph]: A tuple containing:
                - The Knowledge Graph created from the DataFrame
                - The Metadata Graph containing provenance information

        Raises:
            ValueError: If the DataFrame is empty or the identifier column is missing/invalid when provided.
        """
        if df.empty:
            return self.graph, self.metadata_graph

        if identifier_column and identifier_column not in df.columns:
            raise ValueError(
                f"Identifier column '{identifier_column}' not found in DataFrame"
            )

        for idx, row in df.iterrows():
            # Use the provided identifier_column or a fallback if sensible (e.g. 'Name')
            # For this implementation, we stick to the provided identifier_column
            if not identifier_column or pd.isna(row[identifier_column]):
                # Fallback to 'Name' if identifier is missing, or skip row
                if "Name" in row and not pd.isna(row["Name"]):
                    entity_id_source = row["Name"]
                else:
                    print(f"Skipping row {idx} due to missing identifier and Name.")
                    continue # Skip this row if no usable identifier
            else:
                entity_id_source = row[identifier_column]

            entity_id = str(entity_id_source).strip().lower()
            print(f"Creating in License CreativeWork for {entity_id}")
            # Using "License" as a more specific type for hash generation within the platform context
            id_hash = self.generate_entity_hash(platform, "CreativeWork", entity_id) 
            creative_work_uri = self.base_namespace[id_hash]
            
            # Default extraction metadata if not present in the row
            extraction_meta = row.get("extraction_metadata", {"extraction_method": "Unknown", "confidence": 1.0})
            if not isinstance(extraction_meta, dict): # Ensure it's a dict
                extraction_meta = {"extraction_method": "Unknown (invalid format)", "confidence": 1.0}
            
            # Add platform to metadata
            extraction_meta["platform"] = platform

            self.add_triple_with_metadata(
                creative_work_uri,
                RDF.type,
                self.namespaces["schema"]["CreativeWork"], # Licenses are a form of creative work
                extraction_meta 
            )

            if "Name" in row and row["Name"] is not None and not pd.isna(row["Name"]):
                self.add_triple_with_metadata(
                    creative_work_uri,
                    self.namespaces["schema"]["name"],
                    Literal(row["Name"], datatype=XSD.string),
                    extraction_meta
                )
            
            if "Identifier" in row and row["Identifier"] is not None and not pd.isna(row["Identifier"]):
                self.add_triple_with_metadata(
                    creative_work_uri,
                    self.namespaces["schema"]["identifier"],
                    Literal(row["Identifier"], datatype=XSD.string),
                    extraction_meta
                )
            
            if "URL" in row and row["URL"] is not None and not pd.isna(row["URL"]):
                self.add_triple_with_metadata(
                    creative_work_uri,
                    self.namespaces["schema"]["url"],
                    URIRef(row["URL"]), # Assuming URL is a valid URI
                    extraction_meta
                )
            
            if "Text" in row and row["Text"] is not None and not pd.isna(row["Text"]):
                self.add_triple_with_metadata(
                    creative_work_uri,
                    self.namespaces["schema"]["description"], # Or schema:text if more appropriate
                    Literal(row["Text"], datatype=XSD.string),
                    extraction_meta 
                )

        return self.graph, self.metadata_graph
