from typing import List, Optional, Union, Dict, Tuple
import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD
from datetime import datetime

class KnowledgeGraphHandler:
    """
    A class for converting pandas DataFrames to RDF Knowledge Graphs and handling graph integration.
    
    This class provides functionality to transform tabular data into RDF triples and combine
    multiple knowledge graphs while maintaining proper namespacing and relationships.
    
    Args:
        base_namespace (str): The base URI namespace for the knowledge graph entities.
            Default: "http://example.org/"
        predicate_categories (Optional[dict[str, set[str]]]): Dictionary mapping predicate
            categories to sets of predicates. If None, uses default categories.
            Expected format: {
                "date": {"created", "modified"},
                "float": {"accuracy", "loss"},
                "string": {"name", "description"},
                "dataset": {"training_data", "validation_data"},
                "article": {"paper_reference", "citation"}
            }
    
    Raises:
        ValueError: If the base_namespace is not a valid URI string or if predicate_categories
            has an invalid format.
    
    Example:
        >>> df = pd.DataFrame({
        ...     "name": ["Entity1", "Entity2"],
        ...     "property": ["value1", "value2"]
        ... })
        >>> kg_handler = KnowledgeGraphHandler("http://myontology.org/")
        >>> graph = kg_handler.dataframe_to_graph(df, "TestEntity")
    """

    def __init__(
        self, 
        base_namespace: str = "http://example.org/",
        M4ML_schema: pd.DataFrame = None
    ) -> None:
        """
        Initialize the KnowledgeGraphHandler.

        Args:
            base_namespace (str): The base URI namespace for the knowledge graph entities.
                Default: "http://example.org/"
            M4ML_schema (pd.DataFrame): DataFrame containing the M4ML schema with columns
                'Source', 'Property', and 'Range'.

        Raises:
            ValueError: If the base_namespace is not a valid URI string or if M4ML_schema
                has an invalid format.
        """
        self.M4ML_schema = M4ML_schema
        
        # Initialize base namespace and graph
        self.base_namespace = Namespace(base_namespace)
        self.meta_namespace = Namespace(f"{base_namespace}/meta/")
        
        self.graph = Graph()
        
        # Create separate graph for metadata
        self.metadata_graph = Graph()
        
        # Define and bind all required namespaces
        self.namespaces = {
            "base": self.base_namespace,
            "schema": Namespace("http://schema.org/"),
            "fair4ml": Namespace("http://w3id.org/fair4ml/"),
            "codemeta": Namespace("https://w3id.org/codemeta/"),
            "cr": Namespace("https://w3id.org/croissant/"),
            "rdf": RDF,
            "rdfs": RDFS,
            "xsd": XSD
        }
        
        # Bind all namespaces to the graphs
        for prefix, namespace in self.namespaces.items():
            self.graph.bind(prefix, namespace)
            self.metadata_graph.bind(prefix, namespace)
    
    def dataframe_to_graph_Croissant_schema(
        self,
        df: pd.DataFrame,
        identifier_column: Optional[str] = None,
        platform: str = None,
    ) -> Tuple[Graph, Graph]:
        """
        Convert a DataFrame to a Knowledge Graph using the Croissant schema.
        
        Args:
            df (pd.DataFrame): The DataFrame to convert to a Knowledge Graph.
            identifier_column (Optional[str]): The column to use as the identifier for the entities.
            platform (str): The platform name for the entities in the DataFrame.
        
        Returns:
            Tuple[Graph, Graph]: A tuple containing:
                - The Knowledge Graph created from the DataFrame
                - The Metadata Graph containing provenance information
                
        Raises:
            ValueError: If the DataFrame is empty or if the identifier column is not found.
        """
        if df.empty:
            raise ValueError("Cannot convert empty DataFrame to graph")

        if identifier_column and identifier_column not in df.columns:
            raise ValueError(f"Identifier column '{identifier_column}' not found in DataFrame")

        for idx, row in df.iterrows():
            item_json_ld = row['croissant_metadata']
            temp_graph = Graph()
            temp_graph.parse(data=item_json_ld, format='json-ld', base=URIRef(self.base_namespace))
            entity_uri = self.base_namespace[f"{platform}_Dataset_{row['datasetId']}"]
            # creator_uri = self.base_namespace[f"{platform}_User_{}"]
            
            #Go through the triples and add them
            for triple in temp_graph:
                print("TRIPLE\n", triple)
                #Transform the triple to the correct format
                
                self.add_triple_with_metadata(
                    triple[0],
                    triple[1],
                    triple[2],
                    {
                        "extraction_method": row['extraction_metadata']['extraction_method'],
                        "confidence": row['extraction_metadata']['confidence']
                    },
                    row['extraction_metadata']['extraction_time']
            )
                
            
       

        return self.graph, self.metadata_graph

    def dataframe_to_graph_M4ML_schema(
        self,
        df: pd.DataFrame,
        identifier_column: Optional[str] = None,
        platform: str = None,
    ) -> Graph:
        """
        Function to convert a DataFrame to a Knowledge Graph using the M4ML schema.
        
        Args:
            df (pd.DataFrame): The DataFrame to convert to a Knowledge Graph.
            identifier_column (Optional[str]): The column to use as the identifier for the entities.
            platform (str): The platform name for the entities in the DataFrame.
        
        Returns:
            Graph: The Knowledge Graph created from the DataFrame.
            Graph: The Metadata Graph created from the DataFrame.
        """
        
        if df.empty:
            raise ValueError("Cannot convert empty DataFrame to graph")

        if identifier_column and identifier_column not in df.columns:
            raise ValueError(f"Identifier column '{identifier_column}' not found in DataFrame")

        for idx, row in df.iterrows():
            entity_id = row[identifier_column][0]['data'] if identifier_column else str(idx)
            entity_uri = self.base_namespace[f"{platform}_Model_{entity_id}"]
            
            # Add entity type with metadata
            self.add_triple_with_metadata(
                entity_uri, 
                RDF.type, 
                self.base_namespace[f"{platform}_Model"],
                {"extraction_method": "System", "confidence": 1.0}
            )
            
            # Add properties
            for column in df.columns:
                if column != identifier_column:
                    predicate = self.get_predicate_uri(column)
                    values = row[column]
                    
                    for value_info in values:
                        rdf_objects = self.generate_objects_for_M4ML_schema(column,value_info["data"],platform)
                        for rdf_object in rdf_objects:
                            self.add_triple_with_metadata(
                                entity_uri,
                                predicate,
                                rdf_object,
                                {
                                    "extraction_method": value_info["extraction_method"],
                                    "confidence": value_info["confidence"]
                                },
                                value_info["extraction_time"]
                            )

        return self.graph,self.metadata_graph

    def generate_objects_for_M4ML_schema(
        self, predicate: str, values: List[Dict],platform: str
    ) -> List[Literal]:
        """
        Generates appropriate RDF objects based on the predicate type and input value.

        Args:
            predicate (str): The predicate name.
            values (List[Dict]): The value to convert to RDF format.
            platform (str): The platform name for the entities in the DataFrame.
        Returns:
            List[Literal]: A list of RDF literals or URI references.

        Raises:
            ValueError: If the value cannot be properly converted for the given predicate.

        Example:
            >>> predicate = "schema.org:dateCreated"
            >>> values = [{'data': 'user/model_1', 'extraction_method': 'Parsed_from_HF_dataset', 'confidence': 1.0, 'extraction_time': '2025-01-24_07-43-32'}]
            >>> objects = self.generate_objects_for_M4ML_schema(predicate, values,"HF")
            >>> print(objects)
            [Literal('2023-01-01', datatype=XSD.date)]
        """
        if values is None:
            return []
        
        if isinstance(values, list) == False:
            values = [values]
        
        # Find the range where the predicate contains the Property value
        predicate_info = self.M4ML_schema.loc[self.M4ML_schema['Property'].apply(lambda x: x in predicate)]
        
        range_value = predicate_info['Real_Range'].values[0]
        
        objects = []
        
        for value in values:
            
            print("VALUE\n", value)
            
            if predicate_info.empty:
                # Default case: treat as string if predicate not found in schema
                objects.append(Literal(value, datatype=XSD.string))

            # If the predicate is in the schema, use the range value Handle different range types
            if isinstance(value, str):
                if value == "" or value == "None" or value == "No context to answer the question":
                    continue
                
            if 'Text' in range_value or 'String' in range_value:
                objects.append(Literal(value, datatype=XSD.string))
                
            elif 'Date' in range_value or 'DateTime' in range_value:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                formatted_dt = dt.isoformat()
                objects.append(Literal(formatted_dt, datatype=XSD.dateTime))
                
            elif 'Dataset' in range_value:
                objects.append(Literal(value, datatype=XSD.string))
                
            elif 'ScholarlyArticle' in range_value:
                objects.append(self.base_namespace[f"{platform}_Article_{value}"])
                
            elif 'Boolean' in range_value:
                objects.append(Literal(bool(value), datatype=XSD.boolean))
                
            elif 'URL' in range_value:
                objects.append(URIRef(value))
                
            elif 'Person' in range_value or 'Organization' in range_value:
                objects.append(Literal(value, datatype=XSD.string))
            
            
        return objects

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
            http://schema.org/name
        """
        if ":" not in predicate:
            return self.base_namespace[predicate]
        
        prefix, local_name = predicate.split(":", 1)
        
        # Map source prefixes to namespace prefixes
        prefix_mapping = {
            "schema.org": "schema",
            "FAIR4ML": "fair4ml",
            "codemeta": "codemeta"
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
        extraction_time: Optional[str] = None
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
            self.metadata_graph.add((
                statement_id, 
                meta["extractionMethod"], 
                Literal(metadata["extraction_method"])
            ))
        
        if "confidence" in metadata:
            self.metadata_graph.add((
                statement_id, 
                meta["confidence"], 
                Literal(float(metadata["confidence"]), datatype=XSD.float)
            ))
        
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
        
        self.metadata_graph.add((
            statement_id, 
            meta["extractionTime"], 
            Literal(iso_time, datatype=XSD.dateTime)
        ))

    def integrate_graphs(
        self, graphs: List[Graph]
    ) -> Graph:
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
    
    def get_entity_graph_at_time(
        self, 
        entity_uri: URIRef, 
        timestamp: str
    ) -> Graph:
        """
        Reconstruct the graph for a specific entity at a given point in time.

        Args:
            entity_uri (URIRef): URI of the entity to reconstruct
            timestamp (str): ISO format timestamp to reconstruct the graph at

        Returns:
            Graph: A graph containing all valid triples for the entity at the given time
        """
        result_graph = Graph()
        
        # Query metadata graph to find all assertions about this entity
        meta = self.namespaces["meta"]
        
        query = f"""
        SELECT ?s ?p ?o ?time
        WHERE {{
            ?assertion rdf:type meta:TripletMetadata ;
                      meta:subject ?s ;
                      meta:predicate ?p ;
                      meta:object ?o ;
                      meta:extractionTime ?time .
            FILTER(?s = <{str(entity_uri)}>)
            FILTER(?time <= "{timestamp}"^^xsd:dateTime)
        }}
        ORDER BY ?time
        """
        
        # Get the latest version of each triple before the given timestamp
        latest_triples = {}
        for row in self.metadata_graph.query(query):
            s, p, o, time = row
            triple_key = (s, p)
            latest_triples[triple_key] = (o, time)
        
        # Add the latest version of each triple to the result graph
        for (s, p), (o, _) in latest_triples.items():
            result_graph.add((s, p, o))
        
        return result_graph

    def get_current_graph(self) -> Graph:
        """
        Reconstruct the current version of the entire graph based on metadata.

        Returns:
            Graph: The current version of the graph
        """
        current_time = datetime.now().isoformat()
        result_graph = Graph()
        
        # Query to get the latest version of each triple
        query = f"""
        SELECT ?s ?p ?o
        WHERE {{
            {{
                SELECT ?s ?p (MAX(?time) as ?latest_time)
                WHERE {{
                    ?assertion rdf:type meta:TripletMetadata ;
                              meta:subject ?s ;
                              meta:predicate ?p ;
                              meta:extractionTime ?time .
                }}
                GROUP BY ?s ?p
            }}
            ?assertion meta:subject ?s ;
                      meta:predicate ?p ;
                      meta:object ?o ;
                      meta:extractionTime ?latest_time .
        }}
        """
        
        for row in self.metadata_graph.query(query):
            result_graph.add(row)
        
        return result_graph

    def reset_graphs(self) -> None:
        """
        Reset the internal graphs while preserving namespace and schema configurations.
        
        This method clears both the main graph and metadata graph while maintaining
        the initialized namespaces and M4ML schema.
        
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