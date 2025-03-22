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

from .GraphHandler import GraphHandler


class GraphHandlerForDF(GraphHandler):
    """
    GraphHandlerForDF is a class that handles the loading of a graph from a file.
    """
    def __init__(
        self,
        SQLHandler,
        RDFHandler,
        IndexHandler,
        kg_files_directory,
        platform,
        graph_identifier,
        deprecated_graph_identifier,
    ):

        super().__init__(
            SQLHandler,
            RDFHandler,
            IndexHandler,
            kg_files_directory,
            platform,
            graph_identifier,
            deprecated_graph_identifier,
        )
        
        self.models_to_index = []
        self.id_to_model_entity = {}
        self.curr_update_date = None
        self.df_to_transform = None
        self.df = None
        self.extraction_metadata = None
        self.entities_in_kg = {}
    
    def set_df(self, df: pd.DataFrame):
        """
        Load DataFrame for processing.

        Args:
            df (pd.DataFrame): Data to be processed
        """
        self.df = df
    
    def update_graph(self):
        """
        Update graphs across all databases.
         This method:
        1. Updates extraction metadata graph
        2. Updates current graph
        3. Updates search indices
        """
        self.update_extraction_metadata_graph()
        self.update_current_graph()
        self.update_indexes()
    
    # Construct all the triplets in the input dataframe
    def update_extraction_metadata_graph(self):
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
        for column in tqdm(
            self.df.columns, desc=f"Processing properties for model {model_uri}"
        ):
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