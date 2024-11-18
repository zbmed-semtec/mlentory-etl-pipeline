import rdflib
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, XSD, FOAF
from rdflib.util import from_n3
import hashlib
import pandas as pd
from pandas import Timestamp
import json
import os
from datetime import datetime
from typing import Callable, List, Dict, Set

if "app_test" in os.getcwd():
    from load.core.dbHandler.SQLHandler import SQLHandler
    from load.core.dbHandler.RDFHandler import RDFHandler
    from load.core.Entities import HFModel
    from load.core.dbHandler.IndexHandler import IndexHandler
else:
    from core.dbHandler.SQLHandler import SQLHandler
    from core.dbHandler.RDFHandler import RDFHandler
    from core.Entities import HFModel
    from core.dbHandler.IndexHandler import IndexHandler


class GraphHandler:
    def __init__(
        self,
        SQLHandler: SQLHandler,
        RDFHandler: RDFHandler,
        IndexHandler: IndexHandler,
        kg_files_directory: str = "./../kg_files",
        platform: str = "hugging_face",
    ):
        self.df_to_transform = None
        self.SQLHandler = SQLHandler
        self.RDFHandler = RDFHandler
        self.IndexHandler = IndexHandler
        self.kg_files_directory = kg_files_directory
        self.platform = platform
        self.new_triplets = []
        self.old_triplets = []
        self.models_to_index = []
        self.id_to_model_entity = {}
        self.curr_update_date = None
        self.df = pd.DataFrame()

    def load_df(self, df: pd.DataFrame):
        self.df = df

    def update_graph(self):
        # This graph updates the metadata of the triplets and identifies which triplets are new and which ones are not longer valid
        self.update_metadata_graph()
        # This update uses the new_triplets and the old_triplets list to update the current version of the graph.
        self.update_current_graph()
        # This updates the index with the new models and triplets
        self.update_indexes()

    def update_indexes(self):
        new_models = []
        for row_num, row in self.df.iterrows():

            # For each row we first create an m4ml:MLModel instance and their respective triplets
            model_uri = self.models_to_index[row_num]

            index_model_entity = self.IndexHandler.create_hf_index_entity(
                row, model_uri
            )

            # Check if model already exists in elasticsearch
            search_result = self.IndexHandler.search(
                self.IndexHandler.hf_index,
                {"query": {"match_phrase": {"db_identifier": str(model_uri.n3())}}},
            )

            print("SEARCH RESULT: ", search_result)

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

    # Construct all the triplets in the input dataframe
    def update_metadata_graph(self):

        for index, row in self.df.iterrows():
            # For each row we first create an m4ml:MLModel instance and their respective triplets
            model_uri = self.process_model(row)
            self.models_to_index.append(model_uri)

            # Deprecate all the triplets that were not created or updated for the current model
            self.deprecate_old_triplets(model_uri)

        # Update the dates of the models that have not been updated
        self.update_triplet_ranges_for_unchanged_models(self.curr_update_date)
        self.curr_update_date = None

    def process_model(self, row):
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
        for column in self.df.columns:
            # if column == "schema.org:name":
            #     continue
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
            if column in [
                "schema.org:storageRequirements",
                "schema.org:name",
                "schema.org:releaseNotes",
                "codemeta:readme",
                "schema.org:license",
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

    # This function helps us identify if a triplet is new or old
    # In case the triplet is new, we add it to the graph
    # In case the triplet already exists, we update its metadata information
    def process_triplet(self, subject, predicate, object, extraction_info):
        subject_json = str(subject.n3())
        predicate_json = str(predicate.n3())
        object_json = str(object.n3())
        is_new_triplet = False

        triplet_id = -1

        object_hash = hashlib.md5(object_json.encode()).hexdigest()

        triplet_id_df = self.SQLHandler.query(
            f"""SELECT id FROM "Triplet" WHERE subject = '{subject_json}'
                                                                                     AND predicate = '{predicate_json}' 
                                                                                     AND md5(object) = '{object_hash}'"""
        )

        extraction_info_id = -1
        extraction_info_id_df = self.SQLHandler.query(
            f"""SELECT id FROM "Triplet_Extraction_Info" WHERE 
                                                                    method_description = '{extraction_info["extraction_method"]}' 
                                                                    AND extraction_confidence = {extraction_info["confidence"]}"""
        )

        if triplet_id_df.empty:
            # We have to create a new triplet
            triplet_id = self.SQLHandler.insert(
                "Triplet",
                {
                    "subject": subject_json,
                    "predicate": predicate_json,
                    "object": object_json,
                },
            )
            is_new_triplet = True
        else:
            triplet_id = triplet_id_df.iloc[0]["id"]

        if extraction_info_id_df.empty:
            # We have to create a new extraction info
            extraction_info_id = self.SQLHandler.insert(
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
        version_range_df = self.SQLHandler.query(
            f"""SELECT vr.id, vr.use_start, vr.use_end FROM "Version_Range" vr WHERE
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
            version_range_id = self.SQLHandler.insert(
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

    def update_triplet_ranges_for_unchanged_models(self, curr_date: datetime) -> None:
        """
        The idea is to update all the triplet ranges that were not modified in the last update
        to have the same end date as the current date.
        """

        update_query = f"""
            UPDATE "Version_Range"
            SET use_end = '{curr_date}'
            WHERE use_end != '{curr_date}'
            AND deprecated = {False}
        """

        self.SQLHandler.execute_sql(update_query)

    def update_current_graph(self):

        new_triplets_graph = rdflib.Graph(identifier="http://example.com/data_1")
        new_triplets_graph.bind("fair4ml", URIRef("http://fair4ml.com/"))
        new_triplets_graph.bind("codemeta", URIRef("http://codemeta.com/"))
        new_triplets_graph.bind("schema", URIRef("https://schema.org/"))
        new_triplets_graph.bind("mlentory", URIRef("https://mlentory.com/"))
        new_triplets_graph.bind("prov", URIRef("http://www.w3.org/ns/prov#"))

        old_triplets_graph = rdflib.Graph(identifier="http://example.com/data_2")
        old_triplets_graph.bind("fair4ml", URIRef("http://fair4ml.com/"))
        old_triplets_graph.bind("codemeta", URIRef("http://codemeta.com/"))
        old_triplets_graph.bind("schema", URIRef("https://schema.org/"))
        old_triplets_graph.bind("mlentory", URIRef("https://mlentory.com/"))
        old_triplets_graph.bind("prov", URIRef("http://www.w3.org/ns/prov#"))

        for new_triplet in self.new_triplets:
            new_triplets_graph.add(new_triplet)

        for old_triplet in self.old_triplets:
            old_triplets_graph.add(old_triplet)

        current_date = datetime.now().strftime("%Y-%m-%d")
        path_new_triplets_graph = os.path.join(
            self.kg_files_directory, f"new_triplets_graph_{current_date}.ttl"
        )
        new_triplets_graph.serialize(
            destination=path_new_triplets_graph, format="turtle"
        )

        self.RDFHandler.load_graph(ttl_file_path=path_new_triplets_graph)

        if len(old_triplets_graph) > 0:
            path_old_triplets_graph = os.path.join(
                self.kg_files_directory, f"old_triplets_graph_{current_date}.ttl"
            )
            old_triplets_graph.serialize(
                destination=path_old_triplets_graph, format="turtle"
            )

            self.RDFHandler.delete_graph(ttl_file_path=path_old_triplets_graph)

    def n3_to_term(self, n3):
        return from_n3(n3.encode("unicode_escape").decode("unicode_escape"))

    def text_to_uri_term(self, text):
        return URIRef(f"mlentory:/{self.platform}/{text}")
