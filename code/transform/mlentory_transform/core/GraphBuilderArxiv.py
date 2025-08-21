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
            Default: "https://example.org/"
    """
    def __init__(self, base_namespace: str = "https://example.org/") -> None:
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

        if identifier_column and identifier_column not in df.columns:
            raise ValueError(
                f"Identifier column '{identifier_column}' not found in DataFrame"
            )

        for idx, row in df.iterrows():
            entity_id = row[identifier_column].split("v")[0].strip()
            print(f"arXiv ENTITY_ID: {entity_id}")
            id_hash = self.generate_entity_hash(platform, "ScholarlyArticle", entity_id)
            scholarly_article_uri = self.base_namespace[id_hash]

            # Ensure metadata includes platform information
            metadata = dict(row["extraction_metadata"])
            metadata["platform"] = platform

            # Add entity type with metadata
            self.add_triple_with_metadata(
                scholarly_article_uri,
                RDF.type,
                self.namespaces["schema"]["ScholarlyArticle"],
                metadata)
            
            if row["title"] is not None:
                self.add_triple_with_metadata(
                    scholarly_article_uri,
                    self.namespaces["schema"]["name"],
                    Literal(row["title"], datatype=XSD.string),
                    metadata)
            
            
            self.add_triple_with_metadata(
                scholarly_article_uri,
                self.namespaces["schema"]["url"],
                Literal("https://arxiv.org/abs/"+entity_id, datatype=XSD.string),
                metadata)
            
            if row["summary"] is not None:
                self.add_triple_with_metadata(
                    scholarly_article_uri,
                    self.namespaces["schema"]["abstract"],
                    Literal(row["summary"], datatype=XSD.string),
                    metadata)
            
            if row["doi"] is not None:
                self.add_triple_with_metadata(
                    scholarly_article_uri,
                    self.namespaces["schema"]["sameAs"],
                    URIRef(row["doi"]),
                    metadata)
            
            if row["published"] is not None:
                self.add_triple_with_metadata(
                    scholarly_article_uri,
                    self.namespaces["schema"]["datePublished"],
                    Literal(row["published"], datatype=XSD.date),
                    metadata)
            
            if row["categories"] is not None:
                for category in row["categories"]:
                    self.add_triple_with_metadata(
                        scholarly_article_uri,
                        self.namespaces["schema"]["keywords"],
                        Literal(category, datatype=XSD.string),
                        metadata)
            
            if row["authors"] is not None:
                for author in row["authors"]:
                    self.add_triple_with_metadata(
                        scholarly_article_uri,
                        self.namespaces["schema"]["author"],
                        Literal(author, datatype=XSD.string),
                        metadata)
            
            
        return self.graph, self.metadata_graph