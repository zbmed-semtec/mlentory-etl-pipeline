import pprint
import rdflib
import json
import os
import hashlib
import pprint
import pandas as pd
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, XSD, FOAF
from rdflib.util import from_n3
from pandas import Timestamp
from tqdm import tqdm
from datetime import datetime
from typing import Callable, List, Dict, Set, Tuple
import pprint

from mlentory_load.dbHandler import SQLHandler, RDFHandler, IndexHandler
from mlentory_load.core.Entities import HFModel


class GraphHandler:
    """
    Handler for graph operations and version control across databases.

    This class manages:
    - Graph construction and updates
    - Version control of triples
    - Synchronization between databases
    - Metadata tracking

    Attributes:
        SQLHandler (SQLHandler): Handler for SQL operations
        RDFHandler (RDFHandler): Handler for RDF operations
        IndexHandler (IndexHandler): Handler for search index
        kg_files_directory (str): Directory for knowledge graph files
        platform (str): Platform identifier (e.g., "hugging_face")
        graph_identifier (str): URI for main graph
        deprecated_graph_identifier (str): URI for deprecated triples
    """

    def __init__(
        self,
        SQLHandler: SQLHandler,
        RDFHandler: RDFHandler,
        IndexHandler: IndexHandler,
        kg_files_directory: str = "./../kg_files",
        platform: str = "hugging_face",
        graph_identifier: str = "http://example.com/data_1",
        deprecated_graph_identifier: str = "http://example.com/data_2",
    ):
        """
        Initialize GraphHandler with handlers and configuration.

        Args:
            SQLHandler (SQLHandler): SQL database handler
            RDFHandler (RDFHandler): RDF store handler
            IndexHandler (IndexHandler): Search index handler
            kg_files_directory (str): Path to graph files
            platform (str): Platform identifier
            graph_identifier (str): Main graph URI
            deprecated_graph_identifier (str): Deprecated graph URI
            new_triplets (list): List of new triplets identify in the new data
            old_triplets (list): List of old triplets identify in the stored data
            curr_update_date (datetime): Date of the current update
            models_to_index (list): List of models to be indexed
            id_to_model_entity (dict): Dictionary to map the id of the model to the model entity
            df (pd.DataFrame): Dataframe with the new data to be processed
            kg (Graph): Knowledge graph with the new data to be processed
        """
        self.df_to_transform = None
        self.SQLHandler = SQLHandler
        self.RDFHandler = RDFHandler
        self.IndexHandler = IndexHandler
        self.kg_files_directory = kg_files_directory
        self.platform = platform
        self.graph_identifier = graph_identifier
        self.deprecated_graph_identifier = deprecated_graph_identifier
        self.new_triplets = []
        self.old_triplets = []
        self.models_to_index = []
        self.id_to_model_entity = {}
        self.curr_update_date = None
        self.df = None
        self.kg = None
        self.extraction_metadata = None
        self.entities_in_kg = {}

    def set_df(self, df: pd.DataFrame):
        """
        Load DataFrame for processing.

        Args:
            df (pd.DataFrame): Data to be processed
        """
        self.df = df

    def set_kg(self, kg: Graph):
        """
        Set the KG to be loaded.
        """
        self.kg = kg

    def set_extraction_metadata(self, extraction_metadata: Graph):
        """
        Set the extraction metadata to be loaded.
        """
        self.extraction_metadata = extraction_metadata

    def update_graph(self):
        """
        Update graphs across all databases.

        This method:
        1. Updates metadata graph
        2. Updates current graph
        3. Updates search indices
        """
        # This graph updates the metadata of the triplets and identifies which triplets are new and which ones are not longer valid
        if self.kg is not None:
            self.update_metadata_graph_with_kg()
        else:
            self.update_metadata_graph()

        # This update uses the new_triplets and the old_triplets list to update the current version of the graph.
        self.update_current_graph()

        # This updates the index with the new models and triplets
        if self.kg is not None:
            self.update_indexes_with_kg()
        else:
            self.update_indexes()

    def update_metadata_graph_with_kg(self):
        """
        Update the metadata graph with information from a knowledge graph containing metadata nodes.

        This method:
        1. Processes each StatementMetadata node in the graph
        2. Extracts subject, predicate, object and metadata information
        3. Uses process_triplet to handle each triplet's metadata
        """
        # Define the URIs we'll need
        META_NS = self.graph_identifier + "/meta/"
        STATEMENT_METADATA = URIRef(str(META_NS) + "StatementMetadata")

        max_extraction_time = None

        triplets_metadata = {}

        print("Processing extraction metadata...")
        for triplet in tqdm(self.extraction_metadata, desc="Creating triples dictionaries"):
            if triplet[0] not in triplets_metadata:
                triplets_metadata[triplet[0]] = {triplet[1]: triplet[2]}
            else:
                triplets_metadata[triplet[0]][triplet[1]] = triplet[2]

        # Get all nodes of type StatementMetadata
        for metadata_node in tqdm(triplets_metadata, desc="Processing extraction metadata nodes"):

            # print("METADATA NODE\n", metadata_node)

            metadata_node_dict = triplets_metadata[metadata_node]

            # Extract subject, predicate, object
            subject = metadata_node_dict[URIRef(META_NS + "subject")]
            predicate = metadata_node_dict[URIRef(META_NS + "predicate")]
            object_value = metadata_node_dict[URIRef(META_NS + "object")]

            # Extract metadata information
            confidence = float(metadata_node_dict[URIRef(META_NS + "confidence")])
            extraction_method = str(
                metadata_node_dict[URIRef(META_NS + "extractionMethod")]
            )
            extraction_time = str(
                metadata_node_dict[URIRef(META_NS + "extractionTime")]
            ).split(".")[0]

            # Convert datetime format from 2024-02-13T21:24:05+00:00 to our expected format
            extraction_time = datetime.strptime(
                extraction_time, "%Y-%m-%dT%H:%M:%S"
            ).strftime("%Y-%m-%d_%H-%M-%S")

            if max_extraction_time is None:
                max_extraction_time = extraction_time
            else:
                max_extraction_time = max(max_extraction_time, extraction_time)

            # Create extraction info dictionary
            extraction_info = {
                "confidence": confidence,
                "extraction_method": extraction_method,
                "extraction_time": extraction_time,
            }

            # Process the triplet with its metadata
            self.process_triplet(
                subject=subject,
                predicate=predicate,
                object=object_value,
                extraction_info=extraction_info,
            )

        # Update the dates of models that haven't changed
        if self.curr_update_date is None:
            # Use the latest extraction time as current update date
            self.curr_update_date = max_extraction_time

        self.update_triplet_ranges_for_unchanged_models(self.curr_update_date)
        self.curr_update_date = None

    # Construct all the triplets in the input dataframe
    def update_metadata_graph(self):
        """
        Update metadata graph with new information.

        This method:
        1. Processes each model
        2. Updates triplet metadata
        3. Handles deprecation of old data
        """

        for index, row in tqdm(
            self.df.iterrows(), total=len(self.df), desc="Updating metadata graph"
        ):
            # For each row we first create an m4ml:MLModel instance and their respective triplets
            model_uri = self.process_model(row)
            self.models_to_index.append(model_uri)

            # Deprecate all the triplets that were not created or updated for the current model
            self.deprecate_old_triplets(model_uri)

        # Update the dates of the models that have not been updated
        self.update_triplet_ranges_for_unchanged_models(self.curr_update_date)
        self.curr_update_date = None

    def process_model(self, row):
        """
        Process a single model row into graph triplets.

        Args:
            row (pd.Series): Model data to process

        Returns:
            URIRef: URI reference for the processed model
        """
        model_uri = self.text_to_uri_term(str(row["schema.org:name"][0]["data"]))

        self.process_triplet(
            subject=model_uri,
            predicate=RDF.type,
            object=URIRef("fair4ml:MLModel"),
            extraction_info=row["schema.org:name"][0],
        )

        if self.curr_update_date == None:
            self.curr_update_date = datetime.strptime(
                row["schema.org:name"][0]["extraction_time"], "%Y-%m-%d_%H-%M-%S"
            )

        # Go through all the columns and add the triplets
        for column in tqdm(self.df.columns, desc=f"Processing properties for model {model_uri}"):
            # if column == "schema.org:name":
            #     continue
            # Handle the cases where a new entity has to be created
            if column in [
                "fair4ml:mlTask",
                "fair4ml:sharedBy",
                "fair4ml:testedOn",
                "fair4ml:evaluatedOn",
                "fair4ml:trainedOn",
                "codemeta:referencePublication",
            ]:
                # Go through the different sources that can create information about the entity
                if type(row[column]) != list and pd.isna(row[column]):
                    continue
                for source in row[column]:
                    if source["data"] == None:
                        self.process_triplet(
                            subject=model_uri,
                            predicate=URIRef(column),
                            object=Literal("None"),
                            extraction_info=source,
                        )
                    elif type(source["data"]) == str:
                        data = source["data"]
                        self.process_triplet(
                            subject=model_uri,
                            predicate=URIRef(column),
                            object=self.text_to_uri_term(data.replace(" ", "_")),
                            extraction_info=source,
                        )
                    elif type(source["data"]) == list:
                        for entity in source["data"]:
                            self.process_triplet(
                                subject=model_uri,
                                predicate=URIRef(column),
                                object=self.text_to_uri_term(entity.replace(" ", "_")),
                                extraction_info=source,
                            )
            # Handle dates
            if column in [
                "schema.org:datePublished",
                "schema.org:dateCreated",
                "schema.org:dateModified",
            ]:
                self.process_triplet(
                    subject=model_uri,
                    predicate=URIRef(column),
                    object=Literal(row[column][0]["data"], datatype=XSD.date),
                    extraction_info=row[column][0],
                )
            # Handle text values
            if column in [
                "schema.org:storageRequirements",
                "schema.org:name",
                "schema.org:author",
                "schema.org:discussionUrl",
                "schema.org:identifier",
                "schema.org:url",
                "schema.org:releaseNotes",
                "schema.org:license",
                "codemeta:readme",
                "codemeta:issueTracker",
                "fair4ml:intendedUse",
            ]:
                if type(row[column]) != list and pd.isna(row[column]):
                    continue
                for source in row[column]:
                    if source["data"] == None:
                        self.process_triplet(
                            subject=model_uri,
                            predicate=URIRef(column),
                            object=Literal("", datatype=XSD.string),
                            extraction_info=source,
                        )
                    elif type(source["data"]) == str:
                        data = source["data"]
                        self.process_triplet(
                            subject=model_uri,
                            predicate=URIRef(column),
                            object=Literal(row[column][0]["data"], datatype=XSD.string),
                            extraction_info=source,
                        )
                    elif type(source["data"]) == list:
                        for entity in source["data"]:
                            self.process_triplet(
                                subject=model_uri,
                                predicate=URIRef(column),
                                object=Literal(
                                    row[column][0]["data"], datatype=XSD.string
                                ),
                                extraction_info=source,
                            )

        return model_uri

    def _get_or_create_triplet_id(self, subject: URIRef, predicate: URIRef, object: URIRef) -> Tuple[int, bool]:
        """
        Get or create a triplet ID in the database.

        Args:
            subject (URIRef): The subject of the triplet
            predicate (URIRef): The predicate of the triplet
            object (URIRef): The object of the triplet

        Returns:
            tuple[int, bool]: A tuple containing the triplet ID and a boolean indicating if it's new

        Raises:
            SQLError: If there's an error executing the SQL query
        """
        subject_json = str(subject.n3())
        predicate_json = str(predicate.n3())
        object_json = str(object.n3())
        object_hash = hashlib.md5(object_json.encode()).hexdigest()

        triplet_id_df = self.SQLHandler.query(
            f"""SELECT id FROM "Triplet" WHERE subject = '{subject_json}'
                                                   AND predicate = '{predicate_json}' 
                                                   AND md5(object) = '{object_hash}'"""
        )

        if triplet_id_df.empty:
            triplet_id = self.SQLHandler.insert(
                "Triplet",
                {
                    "subject": subject_json,
                    "predicate": predicate_json,
                    "object": object_json,
                },
            )
            return triplet_id, True
        return triplet_id_df.iloc[0]["id"], False

    def _get_or_create_extraction_info_id(self, extraction_info: dict) -> int:
        """
        Get or create an extraction info ID in the database.

        Args:
            extraction_info (dict): Dictionary containing extraction method and confidence

        Returns:
            int: The extraction info ID

        Raises:
            SQLError: If there's an error executing the SQL query
        """
        extraction_info_id_df = self.SQLHandler.query(
            f"""SELECT id FROM "Triplet_Extraction_Info" WHERE 
                                                method_description = '{extraction_info["extraction_method"]}' 
                                                AND extraction_confidence = {extraction_info["confidence"]}"""
        )

        if extraction_info_id_df.empty:
            return self.SQLHandler.insert(
                "Triplet_Extraction_Info",
                {
                    "method_description": extraction_info["extraction_method"],
                    "extraction_confidence": extraction_info["confidence"],
                },
            )
        return extraction_info_id_df.iloc[0]["id"]

    def _manage_version_range(
        self, triplet_id: int, extraction_info_id: int, extraction_time: datetime
    ) -> None:
        """
        Manage version range for a triplet, either creating a new one or updating existing.

        Args:
            triplet_id (int): The ID of the triplet
            extraction_info_id (int): The ID of the extraction info
            extraction_time (datetime): The extraction timestamp

        Raises:
            SQLError: If there's an error executing the SQL query
        """
        version_range_df = self.SQLHandler.query(
            f"""SELECT vr.id, vr.use_start, vr.use_end FROM "Version_Range" vr WHERE
                                                    triplet_id = '{triplet_id}'
                                                    AND extraction_info_id = '{extraction_info_id}'
                                                    AND deprecated = {False}"""
        )

        if version_range_df.empty:
            self.SQLHandler.insert(
                "Version_Range",
                {
                    "triplet_id": str(triplet_id),
                    "extraction_info_id": str(extraction_info_id),
                    "use_start": extraction_time,
                    "use_end": extraction_time,
                    "deprecated": False,
                },
            )
        else:
            version_range_id = version_range_df.iloc[0]["id"]
            self.SQLHandler.update(
                "Version_Range",
                {"use_end": extraction_time},
                f"id = '{version_range_id}'",
            )

    def process_triplet(self, subject, predicate, object, extraction_info):
        """
        Process a triplet by managing its storage and version control in the database.

        Args:
            subject: The subject of the triplet
            predicate: The predicate of the triplet
            object: The object of the triplet
            extraction_info (dict): Dictionary containing extraction metadata

        Raises:
            SQLError: If there's an error executing any SQL query
            ValueError: If extraction_info lacks required fields
        """
        triplet_id, is_new_triplet = self._get_or_create_triplet_id(subject, predicate, object)
        extraction_info_id = self._get_or_create_extraction_info_id(extraction_info)
        
        extraction_time = datetime.strptime(
            extraction_info["extraction_time"], "%Y-%m-%d_%H-%M-%S"
        )
        
        self._manage_version_range(triplet_id, extraction_info_id, extraction_time)

        if is_new_triplet:
            self.new_triplets.append((subject, predicate, object))

    def deprecate_old_triplets(self, model_uri):

        model_uri_json = str(model_uri.n3())

        old_triplets_df = self.SQLHandler.query(
            f"""SELECT t.id,t.subject,t.predicate,t.object, vr.deprecated, vr.use_end
                                                  FROM "Triplet" t 
                                                      JOIN "Version_Range" vr 
                                                      ON t.id = vr.triplet_id
                                                      WHERE t.subject = '{model_uri_json}'
                                                      AND vr.deprecated = {False}
                                                      AND vr.use_end < '{self.curr_update_date}'
                                                    """
        )

        if not old_triplets_df.empty:
            for index, old_triplet in old_triplets_df.iterrows():
                self.old_triplets.append(
                    (
                        self.n3_to_term(old_triplet["subject"]),
                        self.n3_to_term(old_triplet["predicate"]),
                        self.n3_to_term(old_triplet["object"]),
                    )
                )

        update_query = f"""
            UPDATE "Version_Range" vr
            SET deprecated = {True}, use_end = '{self.curr_update_date}'
            FROM "Triplet" t
                JOIN "Version_Range" on t.id = triplet_id
            WHERE t.subject = '{model_uri_json}'
            AND vr.use_end < '{self.curr_update_date}'
            AND vr.deprecated = {False}
            AND vr.triplet_id = t.id
        """
        self.SQLHandler.execute_sql(update_query)

    def update_triplet_ranges_for_unchanged_models(self, curr_date: str) -> None:
        """
        Update all triplet ranges that were not modified in the last update
        to have the same end date as the current date.

        Args:
            curr_date (str): Current date in format 'YYYY-MM-DD_HH-MM-SS'
        """
        # Convert the date string from 'YYYY-MM-DD_HH-MM-SS' to PostgreSQL timestamp format
        formatted_date = datetime.strptime(curr_date, "%Y-%m-%d_%H-%M-%S").strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        update_query = f"""
            UPDATE "Version_Range"
            SET use_end = '{formatted_date}'
            WHERE use_end != '{formatted_date}'
            AND deprecated = {False}
        """

        self.SQLHandler.execute_sql(update_query)

    def update_current_graph(self):
        """
        Update the current graph with new triplets.

        This method:
        1. Creates new triplets graph
        2. Creates deprecated triplets graph
        3. Updates the RDF store
        """
        print("Updating current graph...")
        new_triplets_graph = rdflib.Graph(identifier=self.graph_identifier)
        new_triplets_graph.bind("fair4ml", URIRef("http://fair4ml.com/"))
        new_triplets_graph.bind("codemeta", URIRef("http://codemeta.com/"))
        new_triplets_graph.bind("schema", URIRef("https://schema.org/"))
        new_triplets_graph.bind("mlentory", URIRef("https://mlentory.com/"))
        new_triplets_graph.bind("prov", URIRef("http://www.w3.org/ns/prov#"))

        old_triplets_graph = rdflib.Graph(identifier=self.deprecated_graph_identifier)
        old_triplets_graph.bind("fair4ml", URIRef("http://fair4ml.com/"))
        old_triplets_graph.bind("codemeta", URIRef("http://codemeta.com/"))
        old_triplets_graph.bind("schema", URIRef("https://schema.org/"))
        old_triplets_graph.bind("mlentory", URIRef("https://mlentory.com/"))
        old_triplets_graph.bind("prov", URIRef("http://www.w3.org/ns/prov#"))

        print("Adding new triplets...")
        for new_triplet in tqdm(self.new_triplets, desc="Processing new triplets"):
            new_triplets_graph.add(new_triplet)

        print("Processing deprecated triplets...")
        for old_triplet in tqdm(self.old_triplets, desc="Processing deprecated triplets"):
            old_triplets_graph.add(old_triplet)

        current_date = datetime.now().strftime("%Y-%m-%d")
        path_new_triplets_graph = os.path.join(
            self.kg_files_directory, f"new_triplets_graph_{current_date}.ttl"
        )
        print(f"Serializing new triplets to {path_new_triplets_graph}")
        new_triplets_graph.serialize(
            destination=path_new_triplets_graph, format="turtle"
        )

        print("Loading new triplets into RDF store...")
        self.RDFHandler.load_graph(
            ttl_file_path=path_new_triplets_graph,
            graph_identifier=self.graph_identifier,
        )

        if len(old_triplets_graph) > 0:
            path_old_triplets_graph = os.path.join(
                self.kg_files_directory, f"old_triplets_graph_{current_date}.ttl"
            )
            print(f"Serializing deprecated triplets to {path_old_triplets_graph}")
            old_triplets_graph.serialize(
                destination=path_old_triplets_graph, format="turtle"
            )

            print("Updating RDF store with deprecated triplets...")
            self.RDFHandler.delete_graph(
                ttl_file_path=path_old_triplets_graph,
                graph_identifier=self.graph_identifier,
                deprecated_graph_identifier=self.deprecated_graph_identifier,
            )

    def update_indexes_with_kg(self):
        """
        Update search indices with new and modified models.
        """
        new_models = []

        # Get all the nodes in the KG that are of type MLModel
        entities_in_kg = {}
        print("Updating search indices...")
        for triplet in tqdm(self.kg, desc="Processing current KG triplets"):
            entity_uri = str(triplet[0].n3())
            if entity_uri not in entities_in_kg:
                entities_in_kg[entity_uri] = {triplet[1].n3(): [str(triplet[2])]}
            else:
                if triplet[1].n3() not in entities_in_kg[entity_uri]:
                    entities_in_kg[entity_uri][triplet[1].n3()] = [str(triplet[2])]
                else:
                    entities_in_kg[entity_uri][triplet[1].n3()].append(str(triplet[2]))

        print("Updating search indices...")
        for entity_uri, entity_dict in tqdm(entities_in_kg.items(), desc="Processing entities"):
            # Check if the entity is a model from the entity_dict
            # pprint.pprint(entity_dict["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"])

            if "ML_Model" in entity_dict["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"][0]:
                index_model_entity = (
                    self.IndexHandler.create_hf_dataset_index_entity_with_dict(
                        entity_dict, entity_uri
                    )
                )
                # Check if model already exists in elasticsearch
                search_result = self.IndexHandler.search(
                    self.IndexHandler.hf_index,
                    {"query": {"match_phrase": {"db_identifier": str(entity_uri)}}},
                )

                if not search_result:
                    # Only index if model doesn't exist
                    new_models.append(index_model_entity)
                else:
                    # If model already exists, update the index
                    self.IndexHandler.update_document(
                        index_model_entity.meta.index,
                        search_result[0]["_id"],
                        index_model_entity.to_dict(),
                    )

        if len(new_models) > 0:
            print(f"Adding {len(new_models)} new models to search index...")
            self.IndexHandler.add_documents(new_models)

    def update_indexes(self):
        """Update search indices with new and modified models."""
        new_models = []
        for row_num, row in tqdm(
            self.df.iterrows(), total=len(self.df), desc="Updating indexes"
        ):

            # For each row we first create an m4ml:MLModel instance and their respective triplets
            model_uri = self.models_to_index[row_num]

            index_model_entity = self.IndexHandler.create_hf_model_index_entity(
                row, model_uri
            )

            # Check if model already exists in elasticsearch
            search_result = self.IndexHandler.search(
                self.IndexHandler.hf_index,
                {"query": {"match_phrase": {"db_identifier": str(model_uri.n3())}}},
            )

            # print("SEARCH RESULT: ", search_result)

            if not search_result:
                # Only index if model doesn't exist
                new_models.append(index_model_entity)
            else:
                # If model already exists, update the index
                self.IndexHandler.update_document(
                    index_model_entity.meta.index,
                    search_result[0]["_id"],
                    index_model_entity.to_dict(),
                )

        # Index the model entities
        if len(new_models) > 0:
            self.IndexHandler.add_documents(new_models)

        self.models_to_index = []

    def get_current_graph(self) -> Graph:
        """
        Retrieve current version of the RDF graph.

        Returns:
            Graph: Current active graph with all valid triplets
        """
        current_graph = rdflib.Graph(identifier=self.graph_identifier)
        current_graph.bind("fair4ml", URIRef("http://fair4ml.com/"))
        current_graph.bind("codemeta", URIRef("http://codemeta.com/"))
        current_graph.bind("schema", URIRef("https://schema.org/"))
        current_graph.bind("mlentory", URIRef("https://mlentory.com/"))
        current_graph.bind("prov", URIRef("http://www.w3.org/ns/prov#"))

        query = f"""
        CONSTRUCT {{ ?s ?p ?o }}
        WHERE {{
            GRAPH <{self.graph_identifier}> {{
                ?s ?p ?o
            }}
        }}
        """

        result_graph = self.RDFHandler.query(self.RDFHandler.sparql_endpoint, query)
        current_graph += result_graph

        return current_graph

    def n3_to_term(self, n3):
        return from_n3(n3.encode("unicode_escape").decode("unicode_escape"))

    def text_to_uri_term(self, text):
        return URIRef(f"mlentory:/{self.platform}/{text}")
