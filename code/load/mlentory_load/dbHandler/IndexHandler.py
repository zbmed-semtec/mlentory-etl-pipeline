import os
import pprint
import numpy as np

np.float_ = np.float64

import pandas as pd
from typing import Any, Dict, List
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q, tokenizer
from elasticsearch.helpers import bulk
from elasticsearch_dsl import (
    Document,
    InnerDoc,
    Text,
    Integer,
    Keyword,
    Date,
    Nested,
    Boolean,
    analyzer,
    Index,
)
import inspect

from mlentory_load.core.Entities import HFModel, OpenMLRun


class IndexHandler:
    """
    Handler for Elasticsearch operations.

    This class provides functionality to:
    - Manage Elasticsearch indices
    - Handle document operations
    - Process model metadata
    - Manage search operations

    Attributes:
        es: Elasticsearch client instance
        hf_index (str): Name of HuggingFace models index
    """

    def __init__(self, es_host: str = "localhost", es_port: int = 9200):
        """
        Initialize IndexHandler with connection parameters.

        Args:
            es_host (str): Elasticsearch host address
            es_port (int): Elasticsearch port number
        """

        self.es = Elasticsearch(
            [{"host": es_host, "port": es_port, "scheme": "http"}],
            basic_auth=("elastic", "changeme"),
        )

    def initialize_HF_index(self, index_name: str = "hf_models"):
        """
        Initialize the HuggingFace models index.

        Args:
            index_name (str): Name for the index
        """
        self.hf_index = index_name
        if not self.es.indices.exists(index=index_name):
            HFModel.init(index=index_name, using=self.es)
        # else:
        #     self.es.indices.close(index=index_name)
        #     HFModel.init(index=index_name, using=self.es)
        #     self.es.indices.open(index=index_name)

    def initialize_OpenML_index(self, index_name: str = "openml_models"):
        """
        Initialize the OpenML run index.
        Args:
            index_name (str): Name for the index
        """
        self.openml_index = index_name
        if not self.es.indices.exists(index=index_name):
            OpenMLRun.init(index=index_name, using=self.es)

    def create_hf_model_index_entity(self, row: pd.Series, model_uri: str):
        """
        Create an Elasticsearch document for a HuggingFace model.

        Args:
            row (pd.Series): Model metadata
            model_uri (str): URI identifying the model

        Returns:
            HFModel: Document ready for indexing
        """
        model_uri_json = str(model_uri.n3())
        index_model_entity = HFModel()

        index_model_entity.meta.index = self.hf_index

        index_model_entity.db_identifier = model_uri_json
        if "schema.org:name" in row.keys():
            index_model_entity.name = self.handle_raw_data(row["schema.org:name"])[0]
        else:
            index_model_entity.name = ""
        if "schema.org:releaseNotes" in row.keys():
            index_model_entity.releaseNotes = self.handle_raw_data(
                row["schema.org:releaseNotes"]
            )[0]
        else:
            index_model_entity.releaseNotes = ""
        if "fair4ml:mlTask" in row.keys():
            index_model_entity.mlTask = self.handle_raw_data(row["fair4ml:mlTask"])
        else:
            index_model_entity.mlTask = []
        if "schema.org:author" in row.keys():
            index_model_entity.author = self.handle_raw_data(row["schema.org:author"])
        else:
            index_model_entity.author = []
        
        

        # print(index_model_entity.to_dict())
        return index_model_entity

    def create_openml_index_entity_with_dict(self, info:Dict, uri:str):
        index_run_entity =  OpenMLRun()

        index_run_entity.meta.index = self.openml_index

        index_run_entity.db_identifier = uri
        index_run_entity.name = ""
        index_run_entity.license = ""
        index_run_entity.mlTask = []
        index_run_entity.sharedBy = ""
        index_run_entity.modelCategory = []
        index_run_entity.trainedOn = set()
        index_run_entity.keywords = []

        for key, value in info.items():
            if "identifier" in key:
                index_run_entity.name = (
                    value[0].split("/")[-2] + "/" + value[0].split("/")[-1]
                )
            elif "name" in key:
                index_run_entity.name = value[0]
            elif "sharedBy" in key:
                index_run_entity.sharedBy = value[0]
            elif "license" in key:
                index_run_entity.license = value[0].lower()
            elif "mlTask" in key:
                value = [v.lower() for v in value]
                index_run_entity.mlTask.extend(value)
            elif "modelCategory" in key:
                value = [v.lower() for v in value]
                index_run_entity.modelCategory.extend(value)
            elif "trainedOn" in key:
                index_run_entity.trainedOn.update(value)
            elif "keywords" in key:
                index_run_entity.keywords.extend(value)

        index_run_entity.trainedOn = list(index_run_entity.trainedOn)

        return index_run_entity

    def create_hf_dataset_index_entity_with_dict(self, info: Dict, dataset_uri: str):
        index_model_entity = HFModel()

        index_model_entity.meta.index = self.hf_index

        index_model_entity.db_identifier = dataset_uri
        index_model_entity.name = ""
        index_model_entity.sharedBy = ""
        index_model_entity.releaseNotes = ""
        index_model_entity.license = ""
        index_model_entity.mlTask = []
        index_model_entity.keywords = []
        index_model_entity.relatedDatasets = set()
        index_model_entity.baseModels = []

        for key, value in info.items():
            if "identifier" in key:
                index_model_entity.name = (
                    value[0].split("/")[-2] + "/" + value[0].split("/")[-1]
                )
            elif "releaseNotes" in key:
                index_model_entity.releaseNotes = value[0]
            elif "sharedBy" in key:
                index_model_entity.sharedBy = value[0]
            elif "license" in key:
                index_model_entity.license = value[0].lower()
            elif "mlTask" in key:
                value = [v.lower() for v in value]
                index_model_entity.mlTask.extend(value)
            elif "trainedOn" in key:
                index_model_entity.relatedDatasets.update(value)
            elif "testedOn" in key:
                index_model_entity.relatedDatasets.update(value)
            elif "keywords" in key:
                index_model_entity.keywords.extend(value)
            elif "fineTunedFrom" in key:
                index_model_entity.baseModels.extend(value)
        
        index_model_entity.relatedDatasets = list(index_model_entity.relatedDatasets)
        
        return index_model_entity

    def handle_raw_data(self, raw_data: Any):
        """
        Process raw data into indexable format.

        Args:
            raw_data: Raw data to process

        Returns:
            list: Processed data ready for indexing
        """
        if type(raw_data) != list and pd.isna(raw_data):
            return ""
        formatted_data = []
        for source in raw_data:
            if source["data"] == None:
                formatted_data.append("")
            elif type(source["data"]) == str:
                data = source["data"]
                formatted_data.append(data)
            elif type(source["data"]) == list:
                for data in source["data"]:
                    formatted_data.append(data)

        return formatted_data

    def add_document(self, index_name: str, document: Dict):
        """
        Add a single document to an index.

        Args:
            index_name (str): Target index name
            document (Dict): Document to index
        """
        try:
            self.es.index(index=index_name, body=document)
            print("Document added successfully.")
        except Exception as e:
            print(f"Error adding document: {str(e)}")

    def add_documents(self, documents: List[Dict]):
        """
        Add multiple documents to an index.

        Args:
            documents (List[Dict]): Documents to index
        """
        try:
            bulk(self.es, [doc.upsert() for doc in documents])
            print("Documents added successfully.")
        except Exception as e:
            print(f"Error adding documents: {str(e)}")

    def update_document(self, index_name: str, document_id: str, document: Dict):
        """
        Update an existing document.

        Args:
            index_name (str): Target index name
            document_id (str): ID of document to update
            document (Dict): New document data
        """
        try:
            self.es.update(index=index_name, id=document_id, body={"doc": document})
            print("Document updated successfully.")
        except Exception as e:
            print(f"Error updating document: {str(e)}")

    def search(self, index_name: str, query: Dict):
        """
        Execute a search query.

        Args:
            index_name (str): Index to search
            query (Dict): Search query

        Returns:
            List[Dict]: Search results
        """
        try:
            result = self.es.search(index=index_name, body=query)
            return result["hits"]["hits"]
        except Exception as e:
            print(f"Error searching index: {str(e)}")
            return []

    def delete_index(self, index_name: str):
        """
        Delete an index.

        Args:
            index_name (str): Name of index to delete
        """
        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
            print(f"Index '{index_name}' deleted successfully.")
        else:
            print(f"Index '{index_name}' does not exist.")

    def clean_indices(self):
        """Delete all indices in the Elasticsearch instance."""
        indices_to_delete = list(self.es.indices.get_alias(index="*").keys())
        if len(indices_to_delete) > 0:
            self.es.indices.delete(index=indices_to_delete, ignore=[400, 404])