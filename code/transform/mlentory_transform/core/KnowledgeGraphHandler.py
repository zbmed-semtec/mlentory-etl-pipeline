"""
This module serves as the entry point for different Knowledge Graph Handlers.

It imports the specialized handlers for various schemas (FAIR4ML, Croissant, arXiv, Keywords)
from their respective modules.
"""

from typing import List, Optional, Union, Dict, Tuple, Any
import pprint
import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD
from datetime import datetime
import hashlib
import json
from ..utils.enums import Platform, SchemasURL, EntityType, ExtractionMethod

class KnowledgeGraphHandler:
    """
    A class for converting pandas DataFrames to RDF Knowledge Graphs and handling graph integration.

    This class provides functionality to transform tabular data into RDF triples and combine
    multiple knowledge graphs while maintaining proper namespacing and relationships.

    Args:
        base_namespace (str): The base URI namespace for the knowledge graph entities.
            Default: "https://example.org/"

    Raises:
        ValueError: If the base_namespace is not a valid URI string or if predicate_categories
            has an invalid format.

    Example:
        >>> df = pd.DataFrame({
        ...     "name": ["Entity1", "Entity2"],
        ...     "property": ["value1", "value2"]
        ... })
        >>> kg_handler = KnowledgeGraphHandler("https://myontology.org/")
        >>> graph = kg_handler.dataframe_to_graph(df, "TestEntity")
    """

    def __init__(
        self,
        base_namespace: str = "https://example.org/",
        FAIR4ML_schema_data: pd.DataFrame = None,
    ) -> None:
        """
        Initialize the KnowledgeGraphHandler.

        Args:
            base_namespace (str): The base URI namespace for the knowledge graph entities.
                Default: "https://example.org/"
            FAIR4ML_schema_data (pd.DataFrame): DataFrame containing the FAIR4ML schema with columns
                'Source', 'Property', and 'Range'.

        Raises:
            ValueError: If the base_namespace is not a valid URI string or if FAIR4ML_schema_data
                has an invalid format.
        """
        self.FAIR4ML_schema_data = FAIR4ML_schema_data

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
            print("Warning: Cannot convert empty DataFrame to graph")
            return self.graph, self.metadata_graph

        if identifier_column and identifier_column not in df.columns:
            raise ValueError(
                f"Identifier column '{identifier_column}' not found in DataFrame"
            )

        for idx, row in df.iterrows():
            item_json_ld = row["croissant_metadata"]
            temp_graph = Graph()
            temp_graph.parse(
                data=item_json_ld, format="json-ld", base=URIRef(self.base_namespace)
            )

            self.delete_unwanted_nodes(temp_graph)
            self.replace_blank_nodes_with_type(temp_graph, row, platform)
            self.delete_remaining_blank_nodes(temp_graph)
            # self.replace_blank_nodes_with_no_type(temp_graph,row,platform)
            # self.delete_remaining_blank_nodes(temp_graph)
            # self.replace_default_nodes(temp_graph, row, platform)
            

            # Go through the triples and add them
            for triple in temp_graph:
                # print("TRIPLE:", triple)
                # Transform the triple to the correct format

                self.add_triple_with_metadata(
                    triple[0],
                    triple[1],
                    triple[2],
                    {
                        "extraction_method": row["extraction_metadata"][
                            "extraction_method"
                        ],
                        "confidence": row["extraction_metadata"]["confidence"],
                    },
                    row["extraction_metadata"]["extraction_time"],
                )

        return self.graph, self.metadata_graph

    def dataframe_to_graph_FAIR4ML_schema(
        self,
        df: pd.DataFrame,
        identifier_column: Optional[str] = None,
        platform: str = None,
    ) -> Graph:
        """
        Function to convert a DataFrame to a Knowledge Graph using the FAIR4ML schema.

        Args:
            df (pd.DataFrame): The DataFrame to convert to a Knowledge Graph.
            identifier_column (Optional[str]): The column to use as the identifier for the entities.
            platform (str): The platform name for the entities in the DataFrame.

        Returns:
            Graph: The Knowledge Graph created from the DataFrame.
            Graph: The Metadata Graph created from the DataFrame.
        """

        if df.empty:
            return self.graph, self.metadata_graph

        if identifier_column and identifier_column not in df.columns:
            raise ValueError(
                f"Identifier column '{identifier_column}' not found in DataFrame"
            )
        
        # Determine entity type and extraction method based on platform
        if platform == "open_ml":
            extraction_method = "openml_python_package"
            if identifier_column == "schema.org:name":
                entity_type = "MLModel"
            elif identifier_column == "schema.org:identifier":
                entity_type = "Dataset"
        else:
            entity_type = "MLModel"
            extraction_method = "System"

        for idx, row in df.iterrows():
            entity_id = (
                row[identifier_column][0]["data"] if identifier_column else str(idx)
            )
            
            id_hash = self.generate_entity_hash(platform, entity_type, entity_id)
            entity_uri = self.base_namespace[id_hash]

            # Add entity type with metadata
            self.add_triple_with_metadata(
                entity_uri,
                RDF.type,
                self.namespaces["fair4ml"][entity_type],
                {"extraction_method": extraction_method, "confidence": 1.0},
            )

            # Go through the properties of the model
            for column in df.columns:
                predicate = self.get_predicate_uri(column)
                values = row[column]

                # Check if the values are a list
                for value_info in values:
                    rdf_objects = self.generate_objects_for_FAIR4ML_schema(
                        column, value_info["data"], platform
                    )
                    for rdf_object in rdf_objects:
                        self.add_triple_with_metadata(
                            entity_uri,
                            predicate,
                            rdf_object,
                            {
                                "extraction_method": value_info[
                                    "extraction_method"
                                ],
                                "confidence": value_info["confidence"],
                            },
                            value_info["extraction_time"],
                        )

        return self.graph, self.metadata_graph

    def generate_objects_for_FAIR4ML_schema(
        self, predicate: str, values: List[Dict], platform: str
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
            >>> objects = self.generate_objects_for_FAIR4ML_schema(predicate, values,"HF")
            >>> print(objects)
            [Literal('2023-01-01', datatype=XSD.date)]
        """
        if values is None:
            return []

        if isinstance(values, list) == False:
            values = [values]

        # Find the range where the predicate contains the Property value
        # print("START, PREDICATE:", predicate)
        predicate_info = self.FAIR4ML_schema_data.loc[
            self.FAIR4ML_schema_data["Property"].apply(lambda x: x in predicate)
        ]
        # print("PREDICATE_INFO: \n", predicate_info)
        # print("RANGE: \n", predicate_info["Real_Range"].values[0])
        range_value = predicate_info["Real_Range"].values[0]

        objects = []

        if platform == "open_ml":
            extraction_method = "openml_python_package"
        else:
            extraction_method = "System"

        for value in values:
            if isinstance(value, dict):
                value = value.get('data', value)  # Handle OpenML-style dicts

            if predicate_info.empty:
                # Default case: treat as string if predicate not found in schema
                objects.append(Literal(value, datatype=XSD.string))

            # If the predicate is in the schema, use the range value Handle different range types
            if isinstance(value, str):
                if (
                    value == ""
                    or value == "None"
                    or value == "No context to answer the question"
                    or value == "Information not found"
                ):
                    objects.append(Literal("Information not found", datatype=XSD.string))
                    continue

            if "Text" in range_value or "String" in range_value:
                objects.append(Literal(value, datatype=XSD.string))

            elif "Date" in range_value or "DateTime" in range_value:
                try:
                    # Try to parse the value as a datetime
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    formatted_dt = dt.isoformat()
                    objects.append(Literal(formatted_dt, datatype=XSD.dateTime))
                except ValueError:
                    # If parsing fails, treat it as a regular string
                    print(f"Warning: Could not parse '{value}' as a date for property '{predicate}'. Treating as string.")
                    objects.append(Literal(value, datatype=XSD.string))

            elif "DatasetObject" in range_value: 
                id_hash = self.generate_entity_hash(platform, "DatasetObject", value)
                dataset_object_uri = self.base_namespace[id_hash]

                self.add_triple_with_metadata(
                    dataset_object_uri, 
                    RDF.type, 
                    self.namespaces["fair4ml"]["DatasetObject"], 
                    {"extraction_method": "openml_python_package", "confidence": 1.0}
                )

                self.add_triple_with_metadata(
                    dataset_object_uri, 
                    self.namespaces["schema"]["name"], 
                    Literal(value["name"], datatype=XSD.string), 
                    {"extraction_method": extraction_method, "confidence": 1.0})
                
                self.add_triple_with_metadata(
                    dataset_object_uri, 
                    self.namespaces["schema"]["url"], 
                    Literal(value["url"], datatype=XSD.string), 
                    {"extraction_method": "openml_python_package", "confidence": 1.0}
                )

                sub_id_hash = self.generate_entity_hash(platform, "estimationProcedure"+str(id_hash), value["estimationProcedure"])
                est_proc_uri = self.base_namespace[sub_id_hash]

                # Link estimation procedure to dataset
                self.add_triple_with_metadata(
                    dataset_object_uri,
                    self.namespaces["fair4ml"]["estimationProcedure"],
                    est_proc_uri,
                    {"extraction_method": "openml_python_package", "confidence": 1.0}
                )

                self.add_triple_with_metadata(
                    est_proc_uri,
                    RDF.type,
                    self.namespaces["fair4ml"]["estimationProcedure"],
                    {"extraction_method": "openml_python_package", "confidence": 1.0}
                )

                self.add_triple_with_metadata(
                    est_proc_uri,
                    self.namespaces["schema"]["type"],
                    Literal(value["estimationProcedure"]["type"], datatype=XSD.string),
                    {"extraction_method": "openml_python_package", "confidence": 1.0}
                )

                self.add_triple_with_metadata(
                    est_proc_uri,
                    self.namespaces["schema"]["url"],
                    Literal(value["estimationProcedure"]["data_splits_url"], datatype=XSD.anyURI),
                    {"extraction_method": "openml_python_package", "confidence": 1.0}
                )

                params = value['estimationProcedure']['parameters']
                for param_key, param_val in params.items():
                    self.add_triple_with_metadata(
                        est_proc_uri,
                        self.namespaces["fair4ml"][param_key],
                        Literal(param_val, datatype=XSD.string),
                        {"extraction_method": "openml_python_package", "confidence": 1.0}
                    )

                objects.append(dataset_object_uri)

            elif "Dataset" in range_value:
                #Check if the value can be encoded in a URI
                try:
                    id_hash = self.generate_entity_hash(platform, "Dataset", value)
                    dataset_uri = self.base_namespace[id_hash]
                    self.add_triple_with_metadata(
                        dataset_uri, 
                        RDF.type, 
                        self.namespaces["fair4ml"]["Dataset"], 
                        {"extraction_method": extraction_method, "confidence": 1.0})
                    if( len(value) < 100):
                        self.add_triple_with_metadata(
                            dataset_uri, 
                            self.namespaces["schema"]["name"], 
                            Literal(value, datatype=XSD.string), 
                            {"extraction_method": extraction_method, "confidence": 1.0})
                        self.add_triple_with_metadata(
                            dataset_uri, 
                            self.namespaces["schema"]["url"], 
                            Literal("https://huggingface.co/"+value, datatype=XSD.string), 
                            {"extraction_method": extraction_method, "confidence": 1.0})
                    else:
                        self.add_triple_with_metadata(
                            dataset_uri, 
                            self.namespaces["schema"]["description"], 
                            Literal(value, datatype=XSD.string), 
                            {"extraction_method": extraction_method, "confidence": 1.0})
                        self.add_triple_with_metadata(
                            dataset_uri, 
                            self.namespaces["schema"]["name"], 
                            Literal("Extracted model info: "+value[:50]+"...", datatype=XSD.string), 
                            {"extraction_method": extraction_method, "confidence": 1.0})
                    
                    objects.append(dataset_uri)
                except:
                    objects.append(Literal(value, datatype=XSD.string))

            elif "EvalutionObject" in range_value:
                # Generate a unique URI for the evaluation result            
                id_hash = self.generate_entity_hash(platform, "EvaluationObject", value)
                evaluation_uri = self.base_namespace[id_hash]

                # Add RDF type
                self.add_triple_with_metadata(
                    evaluation_uri,
                    RDF.type,
                    self.namespaces["fair4ml"]["EvaluationObject"],
                    {"extraction_method": "openml_python_package", "confidence": 1.0}
                )

                # Add all evaluation metrics as triples
                for metric_key, metric_val in value.items():
                    self.add_triple_with_metadata(
                        evaluation_uri,
                        self.namespaces["fair4ml"][metric_key],
                        Literal(metric_val, datatype=XSD.double if isinstance(metric_val, float) else XSD.string),
                        {"extraction_method": "openml_python_package", "confidence": 1.0}
                    )

                objects.append(evaluation_uri)
 
            elif "ScholarlyArticle" in range_value:
                value = value.split("/")[-1].split("v")[0].strip()
                id_hash = self.generate_entity_hash(platform, "ScholarlyArticle", value)
                scholarly_article_uri = self.base_namespace[id_hash]
                self.add_triple_with_metadata(
                    scholarly_article_uri, 
                    RDF.type, 
                    self.namespaces["schema"]["ScholarlyArticle"], 
                    {"extraction_method": extraction_method, "confidence": 1.0})
                self.add_triple_with_metadata(
                    scholarly_article_uri, 
                    self.namespaces["schema"]["url"], 
                    Literal("https://arxiv.org/abs/"+value, datatype=XSD.string), 
                    {"extraction_method": extraction_method, "confidence": 1.0})
                objects.append(scholarly_article_uri)

            elif "Boolean" in range_value:
                objects.append(Literal(bool(value), datatype=XSD.boolean))

            elif "Integer" in range_value:
                objects.append(Literal(int(value), datatype=XSD.integer))

            elif "URL" in range_value:
                # Special handling for license values that might not be valid URIs
                if "license" in predicate.lower():
                    # Check if it's a valid URI
                    try:
                        # Try to validate if it's a URI-like string
                        if isinstance(value, str) and value.startswith(('http://', 'https://', 'ftp://', 'file://')):
                            # It looks like a URI, try to create URIRef
                            objects.append(URIRef(value))
                        else:
                            # It's a license identifier like "CC BY-NC 4.0" or "Open Database License (ODbL)", treat as literal
                            objects.append(Literal(value, datatype=XSD.string))
                    except Exception as e:
                        print(f"Warning: Could not process license value '{value}' as URI for predicate '{predicate}': {e}. Treating as string.")
                        objects.append(Literal(value, datatype=XSD.string))
                else:
                    # For non-license URLs, try to create URIRef with fallback to Literal
                    try:
                        objects.append(URIRef(value))
                    except Exception as e:
                        print(f"Warning: Value '{value}' is not a valid URI for predicate '{predicate}': {e}. Treating as string.")
                        objects.append(Literal(value, datatype=XSD.string))

            elif "Person" in range_value:
                id_hash = self.generate_entity_hash(platform, "Person", value)
                person_uri = self.base_namespace[id_hash]
                
                self.add_triple_with_metadata(
                    person_uri, 
                    RDF.type, 
                    self.namespaces["schema"]["Person"], 
                    {"extraction_method": extraction_method, "confidence": 1.0})
                self.add_triple_with_metadata(
                    person_uri, 
                    self.namespaces["schema"]["name"], 
                    Literal(value['name'], datatype=XSD.string), 
                    {"extraction_method": extraction_method, "confidence": 1.0})
                url_value = f"https://huggingface.co/{value}" if platform == "HF" else value['url']
                self.add_triple_with_metadata(
                    person_uri, 
                    self.namespaces["schema"]["url"], 
                    Literal(url_value, datatype=XSD.string), 
                    {"extraction_method": extraction_method, "confidence": 1.0})
                objects.append(person_uri)
                
            elif "Organization" in range_value:
                id_hash = self.generate_entity_hash(platform, "Organization", value)
                organization_uri = self.base_namespace[id_hash]
                self.add_triple_with_metadata(
                    organization_uri, 
                    RDF.type, 
                    self.namespaces["schema"]["Organization"], 
                    {"extraction_method": extraction_method, "confidence": 1.0})
                self.add_triple_with_metadata(
                    organization_uri, 
                    self.namespaces["schema"]["name"], 
                    Literal(value, datatype=XSD.string), 
                    {"extraction_method": extraction_method, "confidence": 1.0})
                url_value = f"https://huggingface.co/{value}" if platform == "hugging_face" else value
                self.add_triple_with_metadata(
                    organization_uri, 
                    self.namespaces["schema"]["url"], 
                    URIRef(url_value), 
                    {"extraction_method": extraction_method, "confidence": 1.0})
                objects.append(organization_uri)
            elif "DefinedTerm" in range_value:
                if platform == Platform.HUGGING_FACE.value and (":" in value or len(value) <= 2):
                    objects.append(Literal(value, datatype=XSD.string))
                elif platform == Platform.OPEN_ML.value and isinstance(value, list):
                    unique_keywords = {str(item).strip() for item in value if str(item).strip()}
                    for keyword in unique_keywords:
                        id_hash = self.generate_entity_hash(platform, "DefinedTerm", keyword.lower())
                        defined_term_uri = self.base_namespace[id_hash]
                        
                        self.add_triple_with_metadata(
                            defined_term_uri,
                            RDF.type,
                            self.namespaces["schema"]["DefinedTerm"],
                            {"extraction_method": "System", "confidence": 1.0}
                        )
                        self.add_triple_with_metadata(
                            defined_term_uri,
                            self.namespaces["schema"]["name"],
                            Literal(keyword),
                            {"extraction_method": "System", "confidence": 1.0}
                        )
                        
                        objects.append(defined_term_uri)
                else:
                    id_hash = self.generate_entity_hash(platform, "DefinedTerm", value.lower().strip())
                    defined_term_uri = self.base_namespace[id_hash]
                    self.add_triple_with_metadata(
                        defined_term_uri,
                        RDF.type,
                        self.namespaces["schema"]["DefinedTerm"],
                        {"extraction_method": "System", "confidence": 1.0})
                    self.add_triple_with_metadata(
                        defined_term_uri,
                        self.namespaces["schema"]["name"],
                        Literal(value, datatype=XSD.string),
                        {"extraction_method": "System", "confidence": 1.0})
                    objects.append(defined_term_uri)
            elif "fair4ml:MLModel" in range_value:
                id_hash = self.generate_entity_hash(platform, "MLModel", value)
                ml_model_uri = self.base_namespace[id_hash]
                self.add_triple_with_metadata(
                    ml_model_uri, 
                    RDF.type, 
                    self.namespaces["fair4ml"]["MLModel"], 
                    {"extraction_method": extraction_method, "confidence": 1.0})
                self.add_triple_with_metadata(
                    ml_model_uri, 
                    self.namespaces["schema"]["name"], 
                    Literal(value, datatype=XSD.string), 
                    {"extraction_method": extraction_method, "confidence": 1.0})
                self.add_triple_with_metadata(
                    ml_model_uri, 
                    self.namespaces["schema"]["url"], 
                    URIRef("https://huggingface.co/"+value), 
                    {"extraction_method": extraction_method, "confidence": 1.0})
                objects.append(ml_model_uri)
        return objects
    
    def dataframe_to_graph_arXiv_schema(
        self,
        df: pd.DataFrame,
        identifier_column: Optional[str] = None,
        platform: str = None,
    ) -> Graph:
        """
        Function to convert a DataFrame to a Knowledge Graph using the arXiv schema.
        """
        if df.empty:
            return self.graph, self.metadata_graph

        if identifier_column and identifier_column not in df.columns:
            raise ValueError(
                f"Identifier column '{identifier_column}' not found in DataFrame"
            )

        for idx, row in df.iterrows():
            entity_id = row[identifier_column].split("v")[0].strip()
            id_hash = self.generate_entity_hash(platform, "ScholarlyArticle", entity_id)
            scholarly_article_uri = self.base_namespace[id_hash]

            # Add entity type with metadata
            self.add_triple_with_metadata(
                scholarly_article_uri,
                RDF.type,
                self.namespaces["schema"]["ScholarlyArticle"],
                row["extraction_metadata"])
            
            if row["title"] is not None:
                self.add_triple_with_metadata(
                    scholarly_article_uri,
                    self.namespaces["schema"]["name"],
                    Literal(row["title"], datatype=XSD.string),
                    row["extraction_metadata"])
            
            
            self.add_triple_with_metadata(
                scholarly_article_uri,
                self.namespaces["schema"]["url"],
                Literal("https://arxiv.org/abs/"+entity_id, datatype=XSD.string),
                row["extraction_metadata"])
            
            if row["summary"] is not None:
                self.add_triple_with_metadata(
                    scholarly_article_uri,
                    self.namespaces["schema"]["abstract"],
                    Literal(row["summary"], datatype=XSD.string),
                    row["extraction_metadata"])
            
            if row["doi"] is not None:
                self.add_triple_with_metadata(
                    scholarly_article_uri,
                    self.namespaces["schema"]["sameAs"],
                    URIRef(row["doi"]),
                    row["extraction_metadata"])
            
            if row["published"] is not None:
                self.add_triple_with_metadata(
                    scholarly_article_uri,
                    self.namespaces["schema"]["datePublished"],
                    Literal(row["published"], datatype=XSD.date),
                    row["extraction_metadata"])
            
            if row["categories"] is not None:
                for category in row["categories"]:
                    self.add_triple_with_metadata(
                        scholarly_article_uri,
                        self.namespaces["schema"]["keywords"],
                        Literal(category, datatype=XSD.string),
                        row["extraction_metadata"])
            
            if row["authors"] is not None:
                for author in row["authors"]:
                    self.add_triple_with_metadata(
                        scholarly_article_uri,
                        self.namespaces["schema"]["author"],
                        Literal(author, datatype=XSD.string),
                        row["extraction_metadata"])
            
            
        return self.graph, self.metadata_graph

    def dataframe_to_graph_keywords(
        self,
        df: pd.DataFrame,
        identifier_column: Optional[str] = None,
        platform: str = None,
    ) -> Graph:
        if df.empty:
            return self.graph, self.metadata_graph

        if identifier_column and identifier_column not in df.columns:
            raise ValueError(
                f"Identifier column '{identifier_column}' not found in DataFrame"
            )

        for idx, row in df.iterrows():
            entity_id = row[identifier_column].strip().lower()
            id_hash = self.generate_entity_hash(platform, "DefinedTerm", entity_id)
            defined_term_uri = self.base_namespace[id_hash]
            
            self.add_triple_with_metadata(
                defined_term_uri,
                RDF.type,
                self.namespaces["schema"]["DefinedTerm"],
                {"extraction_method": ExtractionMethod.ETL.value, "confidence": 1.0})
            
            if row["tag_name"] is not None:
                self.add_triple_with_metadata(
                    defined_term_uri,
                    self.namespaces["schema"]["name"],
                    Literal(row["tag_name"], datatype=XSD.string),
                    {"extraction_method": ExtractionMethod.ETL.value, "confidence": 1.0})
            
            if row["description"] is not None:
                self.add_triple_with_metadata(
                    defined_term_uri,
                    self.namespaces["schema"]["description"],
                    Literal(row["description"], datatype=XSD.string),
                    {"extraction_method": ExtractionMethod.ETL.value, "confidence": 1.0})
                
        
        return self.graph, self.metadata_graph
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

    def replace_blank_nodes_with_type(
        self, graph: Graph, row: pd.Series, platform: str
    ) -> None:
        """
        Replace blank nodes in the graph with the correct values.

        Args:
            graph (Graph): The RDF graph to replace blank nodes in
            row (pd.Series): The row containing the datasetId and other metadata
            platform (str): The platform prefix to use in the URIs
        """

        blank_nodes = self.identify_blank_nodes_with_type(graph)

        # blank_nodes is a dictionary with the type of the blank node as the key and the properties as the value
        if blank_nodes.get("https://schema.org/Dataset", []) != []:
            self.replace_node(
                old_id=blank_nodes.get("https://schema.org/Dataset", [])[0]["node_id"],
                new_id=row["datasetId"],
                graph=graph,
                platform=platform,
                type="https://schema.org/Dataset",
            )

        if blank_nodes.get("https://schema.org/Organization", []) != []:
            name = blank_nodes.get("https://schema.org/Organization", [])[0][
                "properties"
            ]["https://schema.org/name"]
            self.replace_node(
                old_id=blank_nodes.get("https://schema.org/Organization", [])[0][
                    "node_id"
                ],
                new_id=name,
                graph=graph,
                platform=platform,
                type="https://schema.org/Organization",
            )

        if blank_nodes.get("https://schema.org/Person", []) != []:
            name = blank_nodes.get("https://schema.org/Person", [])[0]["properties"][
                "https://schema.org/name"
            ]
            self.replace_node(
                old_id=blank_nodes.get("https://schema.org/Person", [])[0]["node_id"],
                new_id=name,
                graph=graph,
                platform=platform,
                type="https://schema.org/Person",
            )

    def identify_blank_nodes_with_type(
        self, graph: Graph
    ) -> Dict[str, List[Dict[str, Any]]]:
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

            blank_nodes[type_str].append(
                {"type": type_str, "node_id": node.n3(), "properties": properties}
            )

        return blank_nodes

    def replace_blank_nodes_with_no_type(
        self, graph: Graph, row: pd.Series, platform: str
    ) -> None:
        """
        Identify blank nodes that don't have a type and create a new node with an unique ID.

        Args:
            graph (Graph): The RDF graph to replace blank nodes in
            row (pd.Series): The row containing the datasetId and other metadata
            platform (str): The platform prefix to use in the URIs
        """

        blank_nodes = self.identify_blank_nodes_with_no_type(graph)

        for blank_node in blank_nodes:
            new_id = (
                blank_node["parent_id"]
                + "/"
                + blank_node["relation_type"].split("/")[-1]
            )
            self.replace_node(
                old_id=blank_node,
                new_id=new_id,
                graph=graph,
                platform=platform,
                type="https://schema.org/Dataset",
            )

    def replace_default_nodes(
        self, temp_graph: Graph, row: pd.Series, platform: str
    ) -> None:
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
            Original ID: https://test_example.org/default/split
            New ID: https://test_example.org/dataset_name/default/split
        """
        # Find all Field nodes using SPARQL query
        field_types = [
            URIRef("http://mlcommons.org/croissant/Field"),
            # We will ignore the following types for now, they don't add meningfull information
            # URIRef("http://mlcommons.org/croissant/FileSet"),
            # URIRef("http://mlcommons.org/croissant/File"),
            # URIRef("http://mlcommons.org/croissant/FileObject"),
            # URIRef("http://mlcommons.org/croissant/FileObjectSet"),
            URIRef("http://mlcommons.org/croissant/RecordSet"),
        ]

        for field_type in field_types:
            # Get all nodes of type Field
            for field_node in temp_graph.subjects(RDF.type, field_type):
                if isinstance(field_node, URIRef):
                    # Extract the original ID path component
                    original_type = str(field_type).split("/")[-1]
                    field_node_id = str(field_node).split("mlentory_graph/")[-1]+"/"+row["datasetId"]
                    
                    # print("----------------------------")
                    # print("Platform:", platform)                    
                    # print("Original type:", original_type)
                    # print("Field node ID:", field_node_id)
                    # print("----------------------------")

                    # Create new ID with dataset prefix
                    hash = self.generate_entity_hash(platform, original_type, field_node_id)
                    new_uri = self.base_namespace[hash]

                    # Replace all occurrences in the graph
                    self.replace_node(
                        old_id=field_node,
                        new_id=new_uri,
                        graph=temp_graph,
                        node_type=field_type,
                    )
                    # Add a name property to not forget the name of the field
                    temp_graph.add((new_uri, self.namespaces["schema"]["name"], Literal(str(field_node).split("mlentory_graph/")[-1], datatype=XSD.string)))

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

    def delete_remaining_blank_nodes(self, graph: Graph) -> None:
        """
        Delete all triples related to a blank node

        Args:
            graph (Graph): The RDF graph to delete blank nodes from
        """
        # Collect triples to remove
        triples_to_remove = [
            triple for triple in graph
            if isinstance(triple[0], BNode) or isinstance(triple[1], BNode) or isinstance(triple[2], BNode)
        ]
        
        # Remove collected triples
        for triple in triples_to_remove:
            graph.remove(triple)

    def delete_unwanted_nodes(self, graph: Graph) -> None:
        """
        Delete all triples related to unwanted entity types (FileSet, File, FileObject, FileObjectSet).

        This method removes all triples where either the subject or object is an entity of the
        specified unwanted types from the Croissant schema.

        Args:
            graph (Graph): The RDF graph to delete nodes from

        Example:
            >>> kg_handler.delete_unwanted_nodes(graph)
        """
        unwanted_types = [
            URIRef("http://mlcommons.org/croissant/FileSet"),
            URIRef("http://mlcommons.org/croissant/File"),
            URIRef("http://mlcommons.org/croissant/FileObject"),
            URIRef("http://mlcommons.org/croissant/FileObjectSet"),
            URIRef("http://mlcommons.org/croissant/RecordSet"),
            URIRef("http://mlcommons.org/croissant/Field"),
        ]

        # First, find all nodes of unwanted types
        unwanted_nodes = set()
        for type_uri in unwanted_types:
            unwanted_nodes.update(graph.subjects(RDF.type, type_uri))

        # Collect all triples that contain unwanted nodes
        triples_to_remove = [
            triple for triple in graph
            if (triple[0] in unwanted_nodes) or (triple[2] in unwanted_nodes)
        ]

        # Remove collected triples
        for triple in triples_to_remove:
            graph.remove(triple)

    def reset_graphs(self) -> None:
        """
        Reset the internal graphs while preserving namespace and schema configurations.

        This method clears both the main graph and metadata graph while maintaining
        the initialized namespaces and FAIR4ML schema.

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

    