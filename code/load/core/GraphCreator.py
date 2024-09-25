import rdflib
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, XSD, FOAF
from rdflib.util import from_n3
import uuid
import json
import pandas as pd
import json
import os
from datetime import datetime
from typing import Callable, List, Dict, Set

if "app_test" in os.getcwd():
    from load.core.dbHandler.MySQLHandler import MySQLHandler
    from load.core.dbHandler.VirtuosoHandler import VirtuosoHandler
else:
    from core.dbHandler.MySQLHandler import MySQLHandler
    from core.dbHandler.VirtuosoHandler import VirtuosoHandler


class GraphCreator:
    def __init__(
        self,
        mySQLHandler: MySQLHandler,
        virtuosoHandler: VirtuosoHandler,
        kg_files_directory: str = "./../kg_files",
    ):
        self.df_to_transform = None
        self.mySQLHandler = mySQLHandler
        self.virtuosoHandler = virtuosoHandler

        self.new_triplets_graph = rdflib.Graph(identifier="http://example.com/data_1")
        self.new_triplets_graph.bind("fair4ml", URIRef("http://fair4ml.com/"))
        self.new_triplets_graph.bind("codemeta", URIRef("http://codemeta.com/"))
        self.new_triplets_graph.bind("schema", URIRef("https://schema.org/"))
        self.new_triplets_graph.bind("mlentory", URIRef("https://mlentory.com/"))
        self.new_triplets_graph.bind("prov", URIRef("http://www.w3.org/ns/prov#"))

        self.old_triplets_graph = rdflib.Graph(identifier="http://example.com/data_2")
        self.old_triplets_graph.bind("fair4ml", URIRef("http://fair4ml.com/"))
        self.old_triplets_graph.bind("codemeta", URIRef("http://codemeta.com/"))
        self.old_triplets_graph.bind("schema", URIRef("https://schema.org/"))
        self.old_triplets_graph.bind("mlentory", URIRef("https://mlentory.com/"))
        self.old_triplets_graph.bind("prov", URIRef("http://www.w3.org/ns/prov#"))

        self.kg_files_directory = kg_files_directory

        # self.last_update_date = None
        self.curr_update_date = None

    def load_df(self, df):
        self.df = df

    def create_rdf_graph(self):
        self.update_sql_db()

        # current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        path_new_triplets_graph = os.path.join(
            self.kg_files_directory, f"new_triplets_graph_{current_date}.ttl"
        )
        self.new_triplets_graph.serialize(
            destination=path_new_triplets_graph, format="turtle"
        )

        self.virtuosoHandler.load_graph(ttl_file_path=path_new_triplets_graph)

        if len(self.old_triplets_graph) > 0:
            path_old_triplets_graph = os.path.join(
                self.kg_files_directory, f"old_triplets_graph_{current_date}.ttl"
            )
            self.old_triplets_graph.serialize(
                destination=path_old_triplets_graph, format="turtle"
            )

            self.virtuosoHandler.delete_graph(ttl_file_path=path_old_triplets_graph)

        # path_old_triplets_graph = os.path.join(self.kg_files_directory, "old_triplets_graph.ttl")
        # self.old_triplets_graph.serialize(destination=path_old_triplets_graph, format='turtle')

    def update_sql_db(self):
        for index, row in self.df.iterrows():
            # For each row we first create an m4ml:MLModel instance
            model_uri = URIRef(
                f"mlentory:/hugging_face/{str(row['schema.org:name'][0]['data'])}"
            )

            self.create_triplet_in_SQL(
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
            for column in self.df.columns:
                if column == "schema.org:name":
                    continue
                # Handle the cases where a new entity has to be created
                if column in [
                    "fair4ml:mlTask",
                    "fair4ml:sharedBy",
                    "fair4ml:testedOn",
                    "fair4ml:trainedOn",
                    "codemeta:referencePublication",
                ]:
                    # Go through the different sources that can create information about the entity
                    if type(row[column]) != list and pd.isna(row[column]):
                        continue
                    for source in row[column]:
                        if source["data"] == None:
                            self.create_triplet_in_SQL(
                                subject=model_uri,
                                predicate=URIRef(column),
                                object=Literal("None"),
                                extraction_info=source,
                            )
                        elif type(source["data"]) == str:
                            data = source["data"]
                            self.create_triplet_in_SQL(
                                subject=model_uri,
                                predicate=URIRef(column),
                                object=URIRef(
                                    f"mlentory:hugging_face/{data.replace(' ','_')}"
                                ),
                                extraction_info=source,
                            )
                        elif type(source["data"]) == list:
                            for entity in source["data"]:
                                self.create_triplet_in_SQL(
                                    subject=model_uri,
                                    predicate=URIRef(column),
                                    object=URIRef(
                                        f"mlentory:hugging_face/{entity.replace(' ','_')}"
                                    ),
                                    extraction_info=source,
                                )
                if column in [
                    "schema.org:datePublished",
                    "dateCreated",
                    "dateModified",
                ]:
                    # print("ROWWWWWWW",row[column])
                    self.create_triplet_in_SQL(
                        subject=model_uri,
                        predicate=URIRef(column),
                        object=Literal(row[column][0]["data"], datatype=XSD.date),
                        extraction_info=row[column][0],
                    )
                if column in ["storageRequirements", "name"]:
                    self.create_triplet_in_SQL(
                        subject=model_uri,
                        predicate=URIRef(column),
                        object=Literal(row[column]["data"], datatype=XSD.string),
                    )

            # Deprecate all the triplets that were not created or updated for the current model
            self.deprecate_old_triplets(model_uri)

        # Update the dates of the models that have not been updated
        self.update_triplet_ranges_for_unchanged_models(self.curr_update_date)
        self.curr_update_date = None

    def create_triplet_in_SQL(self, subject, predicate, object, extraction_info):
        subject_json = str(subject.n3())
        predicate_json = str(predicate.n3())
        object_json = str(object.n3())

        triplet_id = -1
        triplet_id_df = self.mySQLHandler.query(
            f"""SELECT id FROM Triplet WHERE subject = '{subject_json}'
                                                                                     AND predicate = '{predicate_json}' 
                                                                                     AND object = '{object_json}'"""
        )
        extraction_info_id = -1
        extraction_info_id_df = self.mySQLHandler.query(
            f"""SELECT id FROM Triplet_Extraction_Info WHERE 
                                                                    method_description = '{extraction_info["extraction_method"]}' 
                                                                    AND extraction_confidence = {extraction_info["confidence"]}"""
        )

        if triplet_id_df.empty:
            # We have to create a new triplet
            triplet_id = self.mySQLHandler.insert(
                "Triplet",
                {
                    "subject": subject_json,
                    "predicate": predicate_json,
                    "object": object_json,
                },
            )

            self.new_triplets_graph.add((subject, predicate, object))
        else:
            triplet_id = triplet_id_df.iloc[0]["id"]

        if extraction_info_id_df.empty:
            # We have to create a new extraction info
            extraction_info_id = self.mySQLHandler.insert(
                "Triplet_Extraction_Info",
                {
                    "method_description": extraction_info["extraction_method"],
                    "extraction_confidence": extraction_info["confidence"],
                },
            )
        else:
            extraction_info_id = extraction_info_id_df.iloc[0]["id"]

        # We already have the triplet and the extraction info
        # We need to check if there is already a version range for this triplet
        version_range_df = self.mySQLHandler.query(
            f"""SELECT id,start,end FROM Version_Range WHERE
                                                                        triplet_id = '{triplet_id}'
                                                                        AND extraction_info_id = '{extraction_info_id}'
                                                                        AND deprecated = {False}"""
        )
        version_range_id = -1
        extraction_time = datetime.strptime(
            extraction_info["extraction_time"], "%Y-%m-%d_%H-%M-%S"
        )

        if version_range_df.empty:
            # We have to create a new version range
            version_range_id = self.mySQLHandler.insert(
                "Version_Range",
                {
                    "triplet_id": str(triplet_id),
                    "extraction_info_id": str(extraction_info_id),
                    "start": extraction_time,
                    "end": extraction_time,
                    "deprecated": False,
                },
            )
        else:
            version_range_id = version_range_df.iloc[0]["id"]
            self.mySQLHandler.update(
                "Version_Range", {"end": extraction_time}, f"id = '{version_range_id}'"
            )

    def deprecate_old_triplets(self, model_uri):

        model_uri_json = str(model_uri.n3())

        old_triplets_df = self.mySQLHandler.query(
            f"""SELECT t.id,t.subject,t.predicate,t.object
                                                  FROM Triplet t 
                                                      JOIN Version_Range vr 
                                                      ON t.id = vr.triplet_id
                                                      WHERE t.subject = '{model_uri_json}'
                                                      AND vr.deprecated = 0
                                                      AND vr.end < '{self.curr_update_date}'
                                                    """
        )
        
        if not old_triplets_df.empty:
            for index, old_triplet in old_triplets_df.iterrows():
                self.old_triplets_graph.add(
                    (
                        self.n3_to_term(old_triplet["subject"]),
                        self.n3_to_term(old_triplet["predicate"]),
                        self.n3_to_term(old_triplet["object"]),
                    )
                )

        update_query = f"""
            UPDATE Version_Range vr
            JOIN Triplet t ON t.id = vr.triplet_id
            SET vr.deprecated = 1, vr.end = '{self.curr_update_date}'
            WHERE t.subject = '{model_uri_json}'
            AND vr.end < '{self.curr_update_date}'
            AND vr.deprecated = 0
        """
        self.mySQLHandler.execute_sql(update_query)

    def update_triplet_ranges_for_unchanged_models(self, curr_date: datetime) -> None:
        """
        The idea is to update all the triplet ranges that were not modified in the last update
        to have the same end date as the current date.
        """

        update_query = f"""
            UPDATE Version_Range vr
            SET vr.end = '{curr_date}'
            WHERE vr.end != '{curr_date}'
            AND vr.deprecated = 0
        """

        self.mySQLHandler.execute_sql(update_query)

    def n3_to_term(self, n3):
        return from_n3(n3.encode("unicode_escape").decode("unicode_escape"))
