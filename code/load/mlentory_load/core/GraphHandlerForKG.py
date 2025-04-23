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


class GraphHandlerForKG(GraphHandler):
    """
    GraphHandlerForKG is a class that handles the loading of a graph from a file.
    """

    def __init__(
        self,
        SQLHandler,
        RDFHandler,
        IndexHandler,
        kg_files_directory: str = "./../kg_files",
        platform: str = "hugging_face",
        graph_identifier: str = "http://mlentory.zbmed.de/mlentory_graph",
        deprecated_graph_identifier: str = "http://mlentory.zbmed.de/deprecated_mlentory_graph",
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
        self.kg = None
        self.extraction_metadata = None
        self.entities_in_kg = {}

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
        1. Updates extraction metadata graph
        2. Updates current graph
        3. Updates search indices
        """
        self.update_extraction_metadata_graph_with_kg()
        self.update_current_graph()
        self.update_indexes_with_kg()

    def update_extraction_metadata_graph_with_kg(self):
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
        for triplet in tqdm(
            self.extraction_metadata, desc="Creating triples dictionaries"
        ):
            if triplet[0] not in triplets_metadata:
                triplets_metadata[triplet[0]] = {triplet[1]: triplet[2]}
            else:
                triplets_metadata[triplet[0]][triplet[1]] = triplet[2]

        batch_size = 100
        batch_triplets = []

        # Get all nodes of type StatementMetadata
        for metadata_node in tqdm(
            triplets_metadata, desc="Processing extraction metadata nodes"
        ):

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
            batch_triplets.append((subject, predicate, object_value, extraction_info))
            if len(batch_triplets) == batch_size:
                self.process_triplet_batch(batch_triplets)
                batch_triplets = []

        if len(batch_triplets) > 0:
            self.process_triplet_batch(batch_triplets)

        # Update the dates of models that haven't changed
        if self.curr_update_date is None:
            # Use the latest extraction time as current update date
            self.curr_update_date = max_extraction_time

        self.update_triplet_ranges_for_unchanged_models(self.curr_update_date)
        self.curr_update_date = None

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
        for entity_uri, entity_dict in tqdm(
            entities_in_kg.items(), desc="Processing entities"
        ):
            # Check if the entity is a model from the entity_dict

            if (
                "ML_Model"
                in entity_dict["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"][0]
            ):
                index_model_entity = (
                    self.IndexHandler.create_hf_dataset_index_entity_with_dict(
                        entity_dict, entity_uri
                    )
                )
                
                search_result = None
                
                #Check if index exists
                # if not self.IndexHandler.index_exists(self.IndexHandler.hf_index):
                #     self.IndexHandler.create_index(self.IndexHandler.hf_index)
                # else:
                #     # Check if model already exists in elasticsearch
                #     search_result = self.IndexHandler.search(
                #         self.IndexHandler.hf_index,
                #         {"query": {"match_phrase": {"db_identifier": str(entity_uri)}}},
                #     )
                
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