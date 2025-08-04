from typing import List, Optional, Union, Dict, Tuple, Any
import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD
from datetime import datetime
import hashlib
import json
from ..utils.enums import Platform, SchemasURL, EntityType, ExtractionMethod


class GraphBuilderBase:
    """
    A base class for converting pandas DataFrames to RDF Knowledge Graphs.

    This class provides common functionality for handling namespaces, graphs,
    and adding triples with metadata. Subclasses should implement specific
    schema conversion logic.

    Args:
        base_namespace (str): The base URI namespace for the knowledge graph entities.
            Default: "https://example.org/"

    Raises:
        ValueError: If the base_namespace is not a valid URI string.
    """

    def __init__(
        self,
        base_namespace: str = "https://example.org/",
    ) -> None:
        """
        Initialize the GraphBuilderBase.

        Args:
            base_namespace (str): The base URI namespace for the knowledge graph entities.
                Default: "https://example.org/"

        Raises:
            ValueError: If the base_namespace is not a valid URI string.
        """
        # Initialize base namespace and graph
        self.base_namespace = Namespace(base_namespace)
        self.meta_namespace = Namespace(f"{base_namespace}meta/")

        self.graph = Graph()

        # Create separate graph for metadata
        self.metadata_graph = Graph()

        # Define and bind all required namespaces
        self.namespaces = {
            "base": self.base_namespace,
            "schema": Namespace(SchemasURL.SCHEMA.value),
            "fair4ml": Namespace(SchemasURL.FAIR4ML.value),
            "codemeta": Namespace(SchemasURL.CODEMETA.value),
            "cr": Namespace(SchemasURL.CROISSANT.value),
            "rdf": RDF,
            "rdfs": RDFS,
            "xsd": XSD,
            "mlentory": self.meta_namespace
        }

        # Bind all namespaces to the graphs
        for prefix, namespace in self.namespaces.items():
            self.graph.bind(prefix, namespace)
            self.metadata_graph.bind(prefix, namespace)

    def get_predicate_uri(self, predicate: str) -> URIRef:
        """
        Convert a predicate string to its corresponding URIRef with proper namespace.

        Args:
            predicate (str): The predicate string (e.g., "schema.org:name" or "fair4ml:mlTask")

        Returns:
            URIRef: The properly namespaced predicate URI

        Example:
            >>> uri = kg_handler.get_predicate_uri("schema.org:name")
            >>> print(uri)
            https://schema.org/name
        """
        if ":" not in predicate:
            return self.base_namespace[predicate]

        prefix, local_name = predicate.split(":", 1)

        # Map source prefixes to namespace prefixes
        prefix_mapping = {
            "schema.org": "schema",
            "FAIR4ML": "fair4ml",
            "codemeta": "codemeta",
        }

        namespace_prefix = prefix_mapping.get(prefix, prefix)
        namespace = self.namespaces.get(namespace_prefix)

        if namespace is None:
            raise ValueError(f"Unknown namespace prefix: {prefix}")

        return namespace[local_name]

    def add_triple_with_metadata(
        self,
        subject: URIRef,
        predicate: URIRef,
        object_: Union[URIRef, Literal],
        metadata: Dict[str, any],
        extraction_time: Optional[str] = None,
    ) -> None:
        """
        Add a triple to the main graph and its metadata to the metadata graph.

        Args:
            subject (URIRef): Subject of the triple
            predicate (URIRef): Predicate of the triple
            object_ (Union[URIRef, Literal]): Object of the triple
            metadata (Dict[str, any]): Dictionary containing metadata about the triple
            extraction_time (Optional[str]): Timestamp of when the triple was extracted
        """
        #Check if the triple already exists
        if (subject, predicate, object_) in self.graph:
            return

        # Add the main triple to the knowledge graph
        self.graph.add((subject, predicate, object_))

        # Create a unique identifier for this triple assertion
        statement_id = BNode()
        meta = self.meta_namespace

        # Add metadata about the triple
        self.metadata_graph.add((statement_id, RDF.type, meta["StatementMetadata"]))
        self.metadata_graph.add((statement_id, meta["subject"], subject))
        self.metadata_graph.add((statement_id, meta["predicate"], predicate))
        self.metadata_graph.add((statement_id, meta["object"], object_))

        # Add extraction metadata
        if "extraction_method" in metadata:
            self.metadata_graph.add(
                (
                    statement_id,
                    meta["extractionMethod"],
                    Literal(metadata["extraction_method"]),
                )
            )

        if "confidence" in metadata:
            self.metadata_graph.add(
                (
                    statement_id,
                    meta["confidence"],
                    Literal(float(metadata["confidence"]), datatype=XSD.float),
                )
            )

        # Format extraction time to ISO 8601
        if extraction_time:
            try:
                # Convert from format like '2025-01-24_20-12-20' to datetime object
                dt = datetime.strptime(extraction_time, "%Y-%m-%d_%H-%M-%S")
                # Convert to ISO format
                iso_time = dt.isoformat()
            except ValueError:
                # If parsing fails, use current time
                iso_time = datetime.now().isoformat()
        else:
            iso_time = datetime.now().isoformat()

        self.metadata_graph.add(
            (
                statement_id,
                meta["extractionTime"],
                Literal(iso_time, datatype=XSD.dateTime),
            )
        )

    def integrate_graphs(self, graphs: List[Graph]) -> Graph:
        """
        Integrates multiple RDF graphs into a single knowledge graph.

        Args:
            graphs (List[Graph]): List of RDFlib Graph objects to integrate.

        Returns:
            Graph: A new graph containing the integrated knowledge.
        Example:
            >>> graph1 = kg_handler.dataframe_to_graph(df1, "Person")
            >>> graph2 = kg_handler.dataframe_to_graph(df2, "Organization")
            >>> integrated_graph = kg_handler.integrate_graphs([graph1, graph2])
        """
        if not graphs:
            raise ValueError("No graphs provided for integration")

        integrated_graph = Graph()

        for graph in graphs:
            for triple in graph:
                integrated_graph.add(triple)

        return integrated_graph


    def generate_entity_hash(self, platform: str, entity_type: str, entity_id: str) -> str:
        """
        Generate a consistent hash from entity properties.

        Args:
            platform (str): The platform identifier (e.g., 'HF')
            entity_type (str): The type of entity (e.g., 'Dataset', 'Person')
            entity_id (str): The unique identifier for the entity

        Returns:
            str: A SHA-256 hash of the concatenated properties

        Example:
            >>> hash = kg_handler.generate_entity_hash('HF', 'Dataset', 'dataset1')
            >>> print(hash)
            '8a1c0c50e3e4f0b8a9d5c9e8b7a6f5d4c3b2a1'
        """
        # Create a sorted dictionary of properties to ensure consistent hashing
        properties = {
            "platform": platform,
            "type": entity_type,
            "id": entity_id
        }

        # Convert to JSON string to ensure consistent serialization
        properties_str = json.dumps(properties, sort_keys=True)

        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(properties_str.encode())
        return hash_obj.hexdigest()

    def replace_node(
        self,
        old_id: Union[str, URIRef],
        new_id: Union[str, URIRef],
        graph: Graph,
        platform: str = "HF",
        type: str = "https://schema.org/Dataset",
        node_type: Optional[URIRef] = None,
    ) -> None:
        """
        Replace a node in the graph using SPARQL queries.
        It will replace it from the object and subject side of any triples that contains it.

        Args:
            old_id (Union[str, URIRef]): The old identifier to replace (can be string or URIRef)
            new_id (Union[str, URIRef]): The new identifier to use (can be string or URIRef)
            graph (Graph): The RDF graph where the replacements should be made
            platform (str): The platform prefix to use in the URIs
            type (str): The type of entity being replaced (Dataset/Organization/Person)
            node_type (Optional[URIRef]): Direct RDF type for the node (if provided)

        Raises:
            ValueError: If replacement fails or old node still exists after replacement

        Example:
            >>> # Old style usage
            >>> replace_node("_:b0", "dataset1", graph, "HF", "Dataset")
            >>> # New style usage with URIRefs
            >>> replace_node(old_uri, new_uri, graph, node_type=field_type)
        """
        # Handle both string IDs and URIRefs
        old_uri = old_id if isinstance(old_id, URIRef) else None
        new_uri = new_id if isinstance(new_id, URIRef) else None

        # If string IDs provided, create proper URIRef (old style)
        if old_uri is None and isinstance(old_id, str):
            if old_id.startswith("_:"):
                old_uri = BNode(old_id[2:])
            else:
                old_uri = URIRef(old_id)

        if new_uri is None and isinstance(new_id, str):
            if not new_id.startswith("https"):
                new_id = new_id.replace(' ', '_')
                entity_type = type.split('/')[-1]
                # Generate hash for the entity
                entity_hash = self.generate_entity_hash(platform, entity_type, new_id)
                new_uri = self.base_namespace[entity_hash]
            else:
                new_uri = URIRef(new_id.replace(" ", "_"))

        # Add type declaration
        if node_type:
            graph.add((new_uri, RDF.type, node_type))
        else:
            graph.add((new_uri, RDF.type, URIRef(type)))

        # Update all references in the graph
        for s, p, o in graph.triples((old_uri, None, None)):
            graph.remove((s, p, o))
            graph.add((new_uri, p, o))

        for s, p, o in graph.triples((None, None, old_uri)):
            graph.remove((s, p, o))
            graph.add((s, p, new_uri))


    def reset_graphs(self) -> None:
        """
        Reset the internal graphs while preserving namespace and schema configurations.

        This method clears both the main graph and metadata graph while maintaining
        the initialized namespaces.

        Example:
            >>> kg_handler.reset_graphs()
        """
        # Create fresh graphs
        self.graph = Graph()
        self.metadata_graph = Graph()

        # Rebind all namespaces to the new graphs
        for prefix, namespace in self.namespaces.items():
            self.graph.bind(prefix, namespace)
            self.metadata_graph.bind(prefix, namespace) 