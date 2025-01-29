from typing import List, Optional, Union, Dict, Tuple, Any
import pprint
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
            
            
            self.replace_blank_nodes_with_type(temp_graph,row,platform)
            # self.replace_blank_nodes_with_no_type(temp_graph,row,platform)
            self.replace_default_nodes(temp_graph,row,platform)
            self.delete_remaining_blank_nodes(temp_graph)
            
                    
            #Go through the triples and add them
            for triple in temp_graph:
                # print("TRIPLE:", triple)
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
    

            
    def replace_blank_nodes_with_type(self, graph: Graph, row: pd.Series, platform: str) -> None:
        """
        Replace blank nodes in the graph with the correct values.
        
        Args:
            graph (Graph): The RDF graph to replace blank nodes in
            row (pd.Series): The row containing the datasetId and other metadata
            platform (str): The platform prefix to use in the URIs
        """
        
        
        blank_nodes = self.identify_blank_nodes_with_type(graph)
        
        #blank_nodes is a dictionary with the type of the blank node as the key and the properties as the value
        if blank_nodes.get("https://schema.org/Dataset", []) != []:
            self.replace_node(old_id=blank_nodes.get("https://schema.org/Dataset", [])[0]["node_id"],
                                new_id=row['datasetId'],
                                graph=graph,
                                platform=platform,
                                type="https://schema.org/Dataset")
            
        if blank_nodes.get("https://schema.org/Organization", []) != []:   
            name = blank_nodes.get("https://schema.org/Organization", [])[0]["properties"]["https://schema.org/name"]
            self.replace_node(old_id=blank_nodes.get("https://schema.org/Organization", [])[0]["node_id"],
                                new_id=name,
                                graph=graph,
                                platform=platform,
                                type="https://schema.org/Organization")
        
        if blank_nodes.get("https://schema.org/Person", []) != []:
            name = blank_nodes.get("https://schema.org/Person", [])[0]["properties"]["https://schema.org/name"]
            self.replace_node(old_id=blank_nodes.get("https://schema.org/Person", [])[0]["node_id"],
                                new_id=name,
                                graph=graph,
                                platform=platform,
                                type="https://schema.org/Person")
        

    def identify_blank_nodes_with_type(self, graph: Graph) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify all blank nodes in the graph that have a type and replace them with a node with an ID.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary where:
                - key is the type of the blank node
                - value is a list of dictionaries containing:
                    - node: The blank node identifier
                    - properties: Dict of properties and their values for this node
        """
        blank_nodes = {}
        
        # Query to find all blank nodes and their types
        query = """
        SELECT DISTINCT ?node ?type
        WHERE {
            ?node a ?type .
            FILTER(isBlank(?node))
        }
        """
        
        # First get all blank nodes and their types
        for row in graph.query(query):
            node, node_type = row
            type_str = str(node_type)
            
            if type_str not in blank_nodes:
                blank_nodes[type_str] = []
            
            # Now get all properties for this blank node
            properties = {}
            for pred, obj in graph.predicate_objects(node):
                pred_str = str(pred)
                if isinstance(obj, Literal):
                    properties[pred_str] = str(obj)
                else:
                    properties[pred_str] = str(obj)
            
            blank_nodes[type_str].append({
                "type": type_str,
                "node_id": node.n3(),
                "properties": properties
            })
        
        return blank_nodes
    
    def replace_blank_nodes_with_no_type(self, graph: Graph, row: pd.Series, platform: str) -> None:
        """
        Identify blank nodes that don't have a type and create a new node with an unique ID.
        
        Args:
            graph (Graph): The RDF graph to replace blank nodes in
            row (pd.Series): The row containing the datasetId and other metadata
            platform (str): The platform prefix to use in the URIs
        """
        
        blank_nodes = self.identify_blank_nodes_with_no_type(graph)
        
        for blank_node in blank_nodes:
            new_id = blank_node["parent_id"] + "/" + blank_node["relation_type"].split("/")[-1]
            self.replace_node(old_id=blank_node,
                                new_id=new_id,
                                graph=graph,
                                platform=platform,
                                type="https://schema.org/Dataset")

    def replace_default_nodes(self, temp_graph: Graph, row: pd.Series, platform: str) -> None:
        """
        Identify and update field nodes in the Croissant schema with default types IDs.
        
        This method finds all nodes of types like http://mlcommons.org/croissant/Field, http://mlcommons.org/croissant/FileSet,
        http://mlcommons.org/croissant/FileObject, http://mlcommons.org/croissant/FileObjectSet, http://mlcommons.org/croissant/RecordSet
        and updates their IDs to include the dataset ID, ensuring uniqueness across datasets.
        
        Args:
            temp_graph (Graph): The temporary graph parsed from JSON-LD
            row (pd.Series): DataFrame row containing dataset metadata
            platform (str): Platform prefix (e.g., 'HF' for Hugging Face)
            
        Example:
            Original ID: http://test_example.org/default/split
            New ID: http://test_example.org/dataset_name/default/split
        """
        # Find all Field nodes using SPARQL query
        field_types = [URIRef("http://mlcommons.org/croissant/Field"),
                       URIRef("http://mlcommons.org/croissant/FileSet"),
                       URIRef("http://mlcommons.org/croissant/File"),
                       URIRef("http://mlcommons.org/croissant/FileObject"),
                       URIRef("http://mlcommons.org/croissant/FileObjectSet"),
                       URIRef("http://mlcommons.org/croissant/RecordSet"),
                       ]
        
        for field_type in field_types:
            # Get all nodes of type Field
            for field_node in temp_graph.subjects(RDF.type, field_type):
                if isinstance(field_node, URIRef):
                    # Extract the original ID path component
                    original_path = str(field_node).split("/")[-1]
                    
                    # Create new ID with dataset prefix
                    dataset_id = row['datasetId'].replace("/", "_")
                    new_id = f"{dataset_id}/{original_path}"
                    new_uri = URIRef(f"{self.base_namespace}{new_id}")
                    
                    # Replace all occurrences in the graph
                    self.replace_node(
                        old_id=field_node,
                        new_id=new_uri,
                        graph=temp_graph,
                        node_type=field_type
                    )              
    
    def replace_node(
        self,
        old_id: Union[str, URIRef],
        new_id: Union[str, URIRef],
        graph: Graph,
        platform: str = "HF",
        type: str = "https://schema.org/Dataset",
        node_type: Optional[URIRef] = None
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
            if not new_id.startswith("http"):
                new_uri = self.base_namespace[f"{platform}_{type.split('/')[-1]}_{new_id.replace(' ', '_')}"]
            else:
                new_uri = URIRef(new_id.replace(' ', '_'))
        
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
     
    
    def delete_remaining_blank_nodes(self,graph:Graph):
        """
        Delete all triples related to a blank node
        
        Args:
            graph (Graph): The RDF graph to delete blank nodes from
        """
        
        for triple in graph:
            s, p, o = triple
            if isinstance(s, BNode) or isinstance(o, BNode) or isinstance(p, BNode):
                graph.remove(triple)

        
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
