from typing import List, Optional, Union, Dict, Tuple, Any
from datetime import datetime
import pandas as pd
from rdflib import Graph, Literal, URIRef, BNode
from rdflib.namespace import RDF, XSD

from .GraphBuilderBase import GraphBuilderBase
from ..utils.enums import Platform

class GraphBuilderFAIR4ML(GraphBuilderBase):
    """
    A Knowledge Graph Handler specifically for the FAIR4ML schema.

    Inherits common functionalities from BaseKnowledgeGraphHandler and implements
    methods to convert DataFrames based on the FAIR4ML schema definition.

    Args:
        base_namespace (str): The base URI namespace for the knowledge graph entities.
        FAIR4ML_schema_data (pd.DataFrame): DataFrame containing the FAIR4ML schema with columns
            'Source', 'Property', and 'Range'.

    Raises:
        ValueError: If the base_namespace is not a valid URI string or if FAIR4ML_schema_data
            has an invalid format or is missing.
    """
    def __init__(
        self,
        base_namespace: str,
        FAIR4ML_schema_data: pd.DataFrame,
    ) -> None:
        """
        Initialize the FAIR4MLKnowledgeGraphHandler.

        Args:
            base_namespace (str): The base URI namespace.
            FAIR4ML_schema_data (pd.DataFrame): DataFrame with FAIR4ML schema.

        Raises:
            ValueError: If FAIR4ML_schema_data is not provided or invalid.
        """
        super().__init__(base_namespace)
        if FAIR4ML_schema_data is None or not isinstance(FAIR4ML_schema_data, pd.DataFrame):
            raise ValueError("FAIR4ML_schema_data must be provided as a pandas DataFrame.")
        # TODO: Add more specific validation for schema columns if needed
        self.FAIR4ML_schema_data = FAIR4ML_schema_data

    def hf_dataframe_to_graph(
        self,
        df: pd.DataFrame,
        identifier_column: Optional[str] = None,
        platform: str = None,
    ) -> Tuple[Graph, Graph]:
        """
        Convert a DataFrame to a Knowledge Graph using the FAIR4ML schema.

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
            return self.graph, self.metadata_graph

        if identifier_column and identifier_column not in df.columns:
            raise ValueError(
                f"Identifier column '{identifier_column}' not found in DataFrame"
            )

        for idx, row in df.iterrows():
            entity_id = (
                row[identifier_column][0]["data"] if identifier_column else str(idx)
            )

            id_hash = self.generate_entity_hash(platform, "MLModel", entity_id)
            entity_uri = self.base_namespace[id_hash]

            # Add entity type with metadata
            self.add_triple_with_metadata(
                entity_uri,
                RDF.type,
                self.namespaces["fair4ml"]["ML_Model"],
                {"extraction_method": "System", "confidence": 1.0},
            )

            # Go through the properties of the model
            for column in df.columns:
                if column != identifier_column:
                    try:
                        predicate = self.get_predicate_uri(column)
                    except ValueError as e:
                        print(f"Warning: Skipping column '{column}' due to error: {e}")
                        continue # Skip to next column if predicate URI error

                    values = row[column]

                    # Check if the values are a list
                    if isinstance(values, list):
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
                    else:
                         # Handle cases where the value might not be a list of dicts (e.g., direct values)
                         # This part might need adjustment based on expected data structure variations
                        if values is not None: # Ensure value is not None
                            rdf_objects = self.generate_objects_for_FAIR4ML_schema(
                                column, values, platform # Assuming direct value here
                            )
                            for rdf_object in rdf_objects:
                                # Assuming default metadata if not provided in this structure
                                self.add_triple_with_metadata(
                                    entity_uri,
                                    predicate,
                                    rdf_object,
                                    {"extraction_method": "Unknown", "confidence": 0.0}, # Placeholder metadata
                                    None # No extraction time provided
                                )


        return self.graph, self.metadata_graph

    def generate_objects_for_FAIR4ML_schema(
        self, predicate: str, value: Any, platform: str
    ) -> List[Union[Literal, URIRef]]:
        """
        Generates appropriate RDF objects based on the predicate type and input value.

        Args:
            predicate (str): The predicate name (e.g., "schema.org:dateCreated").
            value (Any): The value to convert to RDF format.
            platform (str): The platform name for the entities in the DataFrame.

        Returns:
            List[Union[Literal, URIRef]]: A list of RDF literals or URI references.

        Raises:
            ValueError: If the value cannot be properly converted for the given predicate.

        Example:
            >>> predicate = "schema.org:dateCreated"
            >>> value = '2023-01-01T12:00:00Z'
            >>> objects = self.generate_objects_for_FAIR4ML_schema(predicate, value,"HF")
            >>> print(objects)
            [Literal('2023-01-01T12:00:00+00:00', datatype=XSD.dateTime)]
        """
        if value is None:
            return []

        # Ensure value is treated consistently, even if single item passed
        values_list = [value] if not isinstance(value, list) else value
        objects = []

        try:
            # Find the range where the predicate contains the Property value
            predicate_info = self.FAIR4ML_schema_data.loc[
                self.FAIR4ML_schema_data["Property"].apply(lambda x: x in predicate)
            ]

            if predicate_info.empty:
                 # Default case: treat as string if predicate not found in schema
                 for v in values_list:
                     if isinstance(v, str) and v.strip() in ["", "None", "No context to answer the question", "Information not found"]:
                         objects.append(Literal("Information not found", datatype=XSD.string))
                     elif v is not None:
                         objects.append(Literal(str(v), datatype=XSD.string))
                 return objects

            range_value = predicate_info["Real_Range"].values[0]

        except Exception as e:
            print(f"Error looking up predicate '{predicate}' in schema: {e}. Treating as string.")
            # Default to string if schema lookup fails
            for v in values_list:
                 if isinstance(v, str) and v.strip() in ["", "None", "No context to answer the question", "Information not found"]:
                     objects.append(Literal("Information not found", datatype=XSD.string))
                 elif v is not None:
                     objects.append(Literal(str(v), datatype=XSD.string))
            return objects


        for item_value in values_list:

            if isinstance(item_value, str):
                # Handle common non-informative strings
                if item_value.strip() in ["", "None", "No context to answer the question", "Information not found"]:
                    objects.append(Literal("Information not found", datatype=XSD.string))
                    continue # Skip further processing for this value
            elif item_value is None:
                 objects.append(Literal("Information not found", datatype=XSD.string))
                 continue

            item_value_str = str(item_value) # Use string representation for processing

            # If the predicate is in the schema, use the range value Handle different range types
            try:
                if "Text" in range_value or "String" in range_value:
                    objects.append(Literal(item_value_str, datatype=XSD.string))

                elif "Date" in range_value or "DateTime" in range_value:
                    try:
                        # Try to parse the value as a datetime
                        dt = datetime.fromisoformat(item_value_str.replace("Z", "+00:00"))
                        formatted_dt = dt.isoformat()
                        objects.append(Literal(formatted_dt, datatype=XSD.dateTime))
                    except ValueError:
                        # If parsing fails, treat it as a regular string
                        print(f"Warning: Could not parse '{item_value_str}' as date/dateTime for predicate '{predicate}'. Treating as string.")
                        objects.append(Literal(item_value_str, datatype=XSD.string))

                elif "Dataset" in range_value:
                    id_hash = self.generate_entity_hash(platform, "Dataset", item_value_str)
                    dataset_uri = self.base_namespace[id_hash]
                    self.add_triple_with_metadata(
                        dataset_uri,
                        RDF.type,
                        self.namespaces["fair4ml"]["Dataset"],
                        {"extraction_method": "System", "confidence": 1.0})
                    if( len(item_value_str) < 100):
                        self.add_triple_with_metadata(
                            dataset_uri,
                            self.namespaces["schema"]["name"],
                            Literal(item_value_str, datatype=XSD.string),
                            {"extraction_method": "System", "confidence": 1.0})
                        if platform == Platform.HUGGING_FACE.value:
                             self.add_triple_with_metadata(
                                dataset_uri,
                                self.namespaces["schema"]["url"],
                                Literal("https://huggingface.co/datasets/"+item_value_str, datatype=XSD.anyURI),
                                {"extraction_method": "System", "confidence": 1.0})
                    else:
                        self.add_triple_with_metadata(
                            dataset_uri,
                            self.namespaces["schema"]["description"],
                            Literal(item_value_str, datatype=XSD.string),
                            {"extraction_method": "System", "confidence": 1.0})
                        self.add_triple_with_metadata(
                            dataset_uri,
                            self.namespaces["schema"]["name"],
                            Literal("Extracted model info: "+item_value_str[:50]+"...", datatype=XSD.string),
                            {"extraction_method": "System", "confidence": 1.0})

                    objects.append(dataset_uri)

                elif "ScholarlyArticle" in range_value:
                    article_id = item_value_str.split("/")[-1].split("v")[0].strip()
                    if not article_id:
                        print(f"Warning: Could not extract valid arXiv ID from '{item_value_str}' for predicate '{predicate}'. Treating as string.")
                        objects.append(Literal(item_value_str, datatype=XSD.string))
                        continue

                    id_hash = self.generate_entity_hash(platform, "ScholarlyArticle", article_id)
                    scholarly_article_uri = self.base_namespace[id_hash]
                    self.add_triple_with_metadata(
                        scholarly_article_uri,
                        RDF.type,
                        self.namespaces["schema"]["ScholarlyArticle"],
                        {"extraction_method": "System", "confidence": 1.0})
                    self.add_triple_with_metadata(
                        scholarly_article_uri,
                        self.namespaces["schema"]["url"],
                        Literal("https://arxiv.org/abs/"+article_id, datatype=XSD.anyURI),
                        {"extraction_method": "System", "confidence": 1.0})
                    objects.append(scholarly_article_uri)

                elif "Boolean" in range_value:
                    objects.append(Literal(bool(item_value), datatype=XSD.boolean))

                elif "URL" in range_value:
                     try:
                        # Attempt to create a URIRef, fallback to Literal if invalid URI
                        objects.append(URIRef(item_value_str))
                     except Exception:
                        print(f"Warning: Value '{item_value_str}' is not a valid URI for predicate '{predicate}'. Treating as string.")
                        objects.append(Literal(item_value_str, datatype=XSD.string))

                elif "Person" in range_value:
                    id_hash = self.generate_entity_hash(platform, "Person", item_value_str)
                    person_uri = self.base_namespace[id_hash]

                    self.add_triple_with_metadata(
                        person_uri,
                        RDF.type,
                        self.namespaces["schema"]["Person"],
                        {"extraction_method": "System", "confidence": 1.0})
                    self.add_triple_with_metadata(
                        person_uri,
                        self.namespaces["schema"]["name"],
                        Literal(item_value_str, datatype=XSD.string),
                        {"extraction_method": "System", "confidence": 1.0})
                    if platform == Platform.HUGGING_FACE.value:
                        self.add_triple_with_metadata(
                            person_uri,
                            self.namespaces["schema"]["url"],
                            Literal("https://huggingface.co/"+item_value_str, datatype=XSD.anyURI),
                            {"extraction_method": "System", "confidence": 1.0})
                    objects.append(person_uri)

                elif "Organization" in range_value:
                    id_hash = self.generate_entity_hash(platform, "Organization", item_value_str)
                    organization_uri = self.base_namespace[id_hash]
                    self.add_triple_with_metadata(
                        organization_uri,
                        RDF.type,
                        self.namespaces["schema"]["Organization"],
                        {"extraction_method": "System", "confidence": 1.0})
                    self.add_triple_with_metadata(
                        organization_uri,
                        self.namespaces["schema"]["name"],
                        Literal(item_value_str, datatype=XSD.string),
                        {"extraction_method": "System", "confidence": 1.0})
                    if platform == Platform.HUGGING_FACE.value:
                        self.add_triple_with_metadata(
                            organization_uri,
                            self.namespaces["schema"]["url"],
                            URIRef("https://huggingface.co/"+item_value_str),
                            {"extraction_method": "System", "confidence": 1.0})
                    objects.append(organization_uri)

                elif "DefinedTerm" in range_value:
                    # Avoid creating DefinedTerms for HF specific codes like 'en: English'
                    if platform == Platform.HUGGING_FACE.value and (":" in item_value_str or len(item_value_str) <= 2):
                        objects.append(Literal(item_value_str, datatype=XSD.string))
                    else:
                        id_hash = self.generate_entity_hash(platform, "DefinedTerm", item_value_str.lower().strip())
                        defined_term_uri = self.base_namespace[id_hash]
                        self.add_triple_with_metadata(
                            defined_term_uri,
                            RDF.type,
                            self.namespaces["schema"]["DefinedTerm"],
                            {"extraction_method": "System", "confidence": 1.0})
                        self.add_triple_with_metadata(
                            defined_term_uri,
                            self.namespaces["schema"]["name"],
                            Literal(item_value_str, datatype=XSD.string),
                            {"extraction_method": "System", "confidence": 1.0})
                        objects.append(defined_term_uri)

                elif "fair4ml:MLModel" in range_value:
                    id_hash = self.generate_entity_hash(platform, "MLModel", item_value_str)
                    ml_model_uri = self.base_namespace[id_hash]
                    self.add_triple_with_metadata(
                        ml_model_uri,
                        RDF.type,
                        self.namespaces["fair4ml"]["MLModel"],
                        {"extraction_method": "System", "confidence": 1.0})
                    self.add_triple_with_metadata(
                        ml_model_uri,
                        self.namespaces["schema"]["name"],
                        Literal(item_value_str, datatype=XSD.string),
                        {"extraction_method": "System", "confidence": 1.0})
                    if platform == Platform.HUGGING_FACE.value:
                        self.add_triple_with_metadata(
                            ml_model_uri,
                            self.namespaces["schema"]["url"],
                            URIRef("https://huggingface.co/"+item_value_str),
                            {"extraction_method": "System", "confidence": 1.0})
                    objects.append(ml_model_uri)
                else:
                    # Fallback for unhandled range types - treat as string
                    print(f"Warning: Unhandled range type '{range_value}' for predicate '{predicate}'. Treating value '{item_value_str}' as string.")
                    objects.append(Literal(item_value_str, datatype=XSD.string))
            except Exception as e:
                 print(f"Error processing value '{item_value_str}' for predicate '{predicate}' with range '{range_value}': {e}. Treating as string.")
                 objects.append(Literal(item_value_str, datatype=XSD.string))

        return objects 