import pytest
import sys
import pandas as pd
from typing import List, Tuple
from unittest.mock import Mock
from pandas import Timestamp

sys.path.append(".")
from load.core.dbHandler.IndexHandler import IndexHandler
from load.core.GraphHandler import GraphHandler


class TestIndexHandler:
    """
    Test class for IndexHandler
    """

    @classmethod
    def setup_class(self):
        self.m4ml_example_dataframe = pd.read_json(
            "./tests/Test_files/load_files/hf_transformed_fair4ml_example.json"
        )
        self.graph_handler = GraphHandler(
            SQLHandler=Mock(),
            RDFHandler=Mock(),
            IndexHandler=Mock(),
            kg_files_directory="./tests/Test_files/load_files/kg_files",
        )

    @pytest.fixture
    def elasticsearch_handler(self) -> IndexHandler:

        elasticsearch_handler = IndexHandler(
            es_host="elastic",
            es_port=9200,
        )

        elasticsearch_handler.initialize_HF_index(index_name="test_hf_models")

        yield elasticsearch_handler

        elasticsearch_handler.clean_indices()
        elasticsearch_handler.es.close()

    def test_index_one_model(self, elasticsearch_handler):
        """
        Test the index_one_model method
        """
        row = self.m4ml_example_dataframe.iloc[0]

        model_uri = self.graph_handler.text_to_uri_term(
            row["schema.org:name"][0]["data"]
        )

        index_model_entity = elasticsearch_handler.index_hf_model(row, model_uri)

        # self.add_document(index_name="hf_models", document=index_model_entity)
        index_model_entity.save(using=elasticsearch_handler.es, index="hf_models")

        elasticsearch_handler.es.indices.refresh(index="hf_models")

        # Check if the document was added to the index
        response = elasticsearch_handler.es.search(
            index="hf_models", body={"query": {"match_all": {}}}
        )

        assert response["hits"]["total"]["value"] == 1

    def test_index_multiple_models(self, elasticsearch_handler):
        """
        Test the index_multiple_models method
        """
        index_model_entities = []
        for _, row in self.m4ml_example_dataframe.iterrows():
            model_uri = self.graph_handler.text_to_uri_term(
                row["schema.org:name"][0]["data"]
            )

            index_model_entity = elasticsearch_handler.index_hf_model(row, model_uri)
            index_model_entities.append(index_model_entity)
            # self.add_document(index_name="hf_models", document=index_model_entity)
            # index_model_entity.save(using=elasticsearch_handler.es , index="hf_models")

        elasticsearch_handler.add_documents(documents=index_model_entities)

        elasticsearch_handler.es.indices.refresh(index="hf_models")

        response = elasticsearch_handler.es.search(
            index="hf_models", body={"query": {"match_all": {}}}
        )

        print(response["hits"]["total"]["value"])

        assert response["hits"]["total"]["value"] == len(self.m4ml_example_dataframe)
