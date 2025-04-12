from typing import List, Optional, Union, Dict, Tuple, Any
import pandas as pd
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from .GraphBuilderBase import GraphBuilderBase

class GraphBuilderArxiv(GraphBuilderBase):
    """
    A Knowledge Graph Handler specifically for arXiv data.

    Inherits common functionalities from BaseKnowledgeGraphHandler and implements
    methods to convert DataFrames containing arXiv metadata into an RDF graph,
    primarily creating schema:ScholarlyArticle entities.

    Args:
        base_namespace (str): The base URI namespace for the knowledge graph entities.
            Default: "http://example.org/"
    """
    def __init__(self, base_namespace: str = "http://example.org/") -> None:
        """
        Initialize the ArXivKnowledgeGraphHandler.

        Args:
            base_namespace (str): The base URI namespace.
        """
        super().__init__(base_namespace)

    def hf_dataframe_to_graph(
        self,
        df: pd.DataFrame,
        identifier_column: Optional[str] = None,
        platform: str = "arXiv", # Default platform
    ) -> Tuple[Graph, Graph]:
        """
        Convert a DataFrame containing arXiv metadata to a Knowledge Graph.

        Maps columns like 'title', 'summary', 'authors', 'published', etc., to
        schema.org properties for ScholarlyArticle entities.

        Args:
            df (pd.DataFrame): The DataFrame with arXiv data.
            identifier_column (Optional[str]): The column containing the arXiv ID (e.g., '2301.00001v1').
                                           Must be provided.
            platform (str): The platform name, defaults to "arXiv".

        Returns:
            Tuple[Graph, Graph]: A tuple containing:
                - The Knowledge Graph created from the DataFrame
                - The Metadata Graph containing provenance information (using row metadata)

        Raises:
            ValueError: If the DataFrame is empty or the identifier column is missing/invalid.
        """
        if df.empty:
            return self.graph, self.metadata_graph

        if not identifier_column or identifier_column not in df.columns:
            raise ValueError(
                f"Identifier column '{identifier_column}' must be provided and exist in DataFrame for arXiv schema."
            )
        if "extraction_metadata" not in df.columns:
             raise ValueError("DataFrame must contain 'extraction_metadata' column for arXiv schema.")

        for idx, row in df.iterrows():
            raw_entity_id = row[identifier_column]
            if not raw_entity_id or not isinstance(raw_entity_id, str):
                 print(f"Warning: Skipping row {idx} due to missing or invalid identifier.")
                 continue

            # Extract the core arXiv ID (remove version)
            entity_id = raw_entity_id.split("v")[0].strip()
            if not entity_id:
                 print(f"Warning: Skipping row {idx} due to invalid arXiv ID format '{raw_entity_id}'.")
                 continue

            # print(f"arXiv ENTITY_ID: {entity_id}") # Keep for debugging if needed
            id_hash = self.generate_entity_hash(platform, "ScholarlyArticle", entity_id)
            scholarly_article_uri = self.base_namespace[id_hash]

            extraction_metadata = row["extraction_metadata"]
            if not isinstance(extraction_metadata, dict):
                 print(f"Warning: Skipping row {idx} due to invalid extraction_metadata format.")
                 continue
            extraction_time = extraction_metadata.get("extraction_time")
            metadata_dict = {
                "extraction_method": extraction_metadata.get("extraction_method", "arXiv API"), # Default method
                "confidence": extraction_metadata.get("confidence", 1.0), # Assume high confidence
            }

            # Add entity type with metadata
            self.add_triple_with_metadata(
                scholarly_article_uri,
                RDF.type,
                self.namespaces["schema"]["ScholarlyArticle"],
                metadata_dict,
                extraction_time)

            # Map DataFrame columns to schema properties
            property_map = {
                "title": (self.namespaces["schema"]["name"], XSD.string),
                "summary": (self.namespaces["schema"]["abstract"], XSD.string),
                "doi": (self.namespaces["schema"]["sameAs"], XSD.anyURI), # Assuming DOI is a URI
                "published": (self.namespaces["schema"]["datePublished"], XSD.date), # Check if dateTime needed
                # Handle list types separately below
            }

            for col, (predicate, datatype) in property_map.items():
                if col in row and pd.notna(row[col]):
                    value = row[col]
                    literal_value = Literal(value, datatype=datatype)
                    if datatype == XSD.anyURI:
                        # Attempt to create URIRef for DOI, fallback to literal if invalid
                        try:
                           object_value = URIRef(value)
                        except Exception:
                            print(f"Warning: Invalid URI for DOI '{value}' in row {idx}. Storing as Literal.")
                            object_value = Literal(value, datatype=XSD.string)
                    else:
                        object_value = Literal(value, datatype=datatype)

                    self.add_triple_with_metadata(
                        scholarly_article_uri,
                        predicate,
                        object_value,
                        metadata_dict,
                        extraction_time)

            # Handle list-based properties (categories, authors)
            if "categories" in row and row["categories"] is not None:
                if isinstance(row["categories"], list):
                     for category in row["categories"]:
                        if category and isinstance(category, str):
                             self.add_triple_with_metadata(
                                scholarly_article_uri,
                                self.namespaces["schema"]["keywords"], # Using keywords for categories
                                Literal(category, datatype=XSD.string),
                                metadata_dict,
                                extraction_time)
                else: 
                     print(f"Warning: 'categories' column in row {idx} is not a list.")

            if "authors" in row and row["authors"] is not None:
                if isinstance(row["authors"], list):
                     for author_name in row["authors"]:
                        if author_name and isinstance(author_name, str):
                            # Simple approach: add author names as literals.
                            # Enhancement: Could create schema:Person entities if needed.
                            self.add_triple_with_metadata(
                                scholarly_article_uri,
                                self.namespaces["schema"]["author"],
                                Literal(author_name, datatype=XSD.string),
                                metadata_dict,
                                extraction_time)
                else:
                    print(f"Warning: 'authors' column in row {idx} is not a list.")

            # Add the canonical arXiv URL
            arxiv_url = f"https://arxiv.org/abs/{entity_id}"
            self.add_triple_with_metadata(
                scholarly_article_uri,
                self.namespaces["schema"]["url"],
                Literal(arxiv_url, datatype=XSD.anyURI), # Store as URI literal
                metadata_dict,
                extraction_time)

        return self.graph, self.metadata_graph 