import numpy as np

np.float_ = np.float64
import pytest
import sys
import os
import json
import pandas as pd
from typing import List, Tuple
from unittest.mock import Mock
from pandas import Timestamp
from rdflib import Graph, URIRef, Literal, BNode
from datetime import datetime

sys.path.append(".")
from load.core.GraphHandler import GraphHandler
from load.core.dbHandler.SQLHandler import SQLHandler
from load.core.dbHandler.RDFHandler import RDFHandler
from load.core.dbHandler.IndexHandler import IndexHandler


class TestGraphHandler:
    """
    Test class for GraphHandler
    """

    @classmethod
    def setup_class(self):
        self.m4ml_example_dataframe = pd.read_json(
            "./tests/Test_files/load_files/hf_transformed_fair4ml_example.json"
        )

    @pytest.fixture
    def setup_sql_handler(self) -> SQLHandler:
        sql_handler = SQLHandler(
            host="postgres",
            user="test_user",
            password="test_password",
            database="test_DB",
        )
        sql_handler.connect()
        sql_handler.delete_all_tables()

        yield sql_handler
        # disconnect and close the connection
        sql_handler.disconnect()

    @pytest.fixture
    def setup_virtuoso_handler(self) -> RDFHandler:
        kg_files_directory = "./tests/Test_files/load_files/virtuoso_data/kg_files"

        rdfHandler = RDFHandler(
            container_name="virtuoso",
            kg_files_directory=kg_files_directory,
            _user="dba",
            _password="my_strong_password",
            sparql_endpoint="http://virtuoso:8890/sparql",
        )

        rdfHandler.reset_db()

        return rdfHandler

    @pytest.fixture
    def setup_elasticsearch_handler(self) -> IndexHandler:
        elasticsearch_handler = IndexHandler(
            es_host="elastic",
            es_port=9200,
        )

        elasticsearch_handler.initialize_HF_index(index_name="test_hf_models")

        yield elasticsearch_handler

        elasticsearch_handler.clean_indices()
        elasticsearch_handler.es.close()

    @pytest.fixture
    def setup_mock_graph_handler(self) -> GraphHandler:
        mock_SQLHandler = Mock(spec=SQLHandler)
        mock_RDFHandler = Mock(spec=RDFHandler)
        mock_IndexHandler = Mock(spec=IndexHandler)
        graph_handler = GraphHandler(
            mock_SQLHandler,
            mock_RDFHandler,
            mock_IndexHandler,
            kg_files_directory="./tests/Test_files/load_files/virtuoso_data/kg_files",
        )
        graph_handler.load_df(self.m4ml_example_dataframe)
        return graph_handler

    @pytest.fixture
    def setup_graph_handler(
        self, setup_sql_handler, setup_virtuoso_handler, setup_elasticsearch_handler
    ) -> GraphHandler:
        # Initializing the database handlers
        graph_handler = GraphHandler(
            setup_sql_handler,
            setup_virtuoso_handler,
            setup_elasticsearch_handler,
            kg_files_directory=setup_virtuoso_handler.kg_files_directory,
        )
        return graph_handler

    def create_graph(self, source_file_path: str, graph_handler: GraphHandler):
        df_example = pd.read_json(source_file_path)
        graph_handler.load_df(df_example)
        graph_handler.update_graph()

    def assert_sql_db_state(
        self,
        expected_triplets: int,
        expected_models: int,
        expected_ranges: int,
        expected_extraction_info: int,
        graph_handler: GraphHandler,
        expected_deprecated: int = 0,
        print_df: bool = False,
    ):
        triplets_df = graph_handler.SQLHandler.query('SELECT * FROM "Triplet"')
        ranges_df = graph_handler.SQLHandler.query('SELECT * FROM "Version_Range"')
        extraction_info_df = graph_handler.SQLHandler.query(
            'SELECT * FROM "Triplet_Extraction_Info"'
        )

        if print_df:
            print("TRIPlETS\n", triplets_df)
            print("RANGES\n", ranges_df)
            print("EXTRACTION_INFO\n", extraction_info_df)

        assert len(extraction_info_df) == expected_extraction_info
        assert len(triplets_df) == expected_triplets
        assert len(ranges_df) == expected_ranges
        assert len(triplets_df["subject"].unique()) == expected_models
        assert len(ranges_df[ranges_df["deprecated"] == True]) == expected_deprecated

    def assert_virtuoso_db_state(
        self,
        expected_triplets: int,
        expected_models: int,
        graph_handler: GraphHandler,
        print_graph=False,
    ):
        result_graph = graph_handler.RDFHandler.query(
            "http://virtuoso:8890/sparql",
            """CONSTRUCT { ?s ?p ?o } WHERE {GRAPH <http://example.com/data_1> {?s ?p ?o}}""",
        )
        # Check there are the number of expected models
        # print the result_graph
        if print_graph:
            for i, (s, p, o) in enumerate(result_graph):
                print(f"{i}: {s} {p} {o}")

        assert len(result_graph) == expected_triplets

        result_count = result_graph.query(
            """SELECT (COUNT(DISTINCT ?s) AS ?count) WHERE{?s ?p ?o}"""
        )

        for triple in result_count:
            assert triple.asdict()["count"]._value == expected_models
        # assert result_count[0]["count"] == expected_models
        # assert

    def assert_elasticsearch_state(
        self,
        expected_models: int,
        graph_handler: GraphHandler,
        print_info=False,
    ):
        graph_handler.IndexHandler.es.indices.refresh(index="test_hf_models")
        result = graph_handler.IndexHandler.es.search(
            index="test_hf_models",
            body={"query": {"match_all": {}}},
        )
        print("Check Elasticsearch: ", result, "\n")
        result_count = result["hits"]["total"]["value"]
        assert result_count == expected_models

    def assert_dbs_states(
        self,
        expected_triplets: int,
        expected_models: int,
        expected_ranges: int,
        expected_extraction_info: int,
        graph_handler: GraphHandler,
        expected_deprecated: int = 0,
        print_df: bool = False,
    ):
        self.assert_sql_db_state(
            expected_triplets,
            expected_models,
            expected_ranges,
            expected_extraction_info,
            graph_handler,
            expected_deprecated,
            print_df,
        )
        self.assert_virtuoso_db_state(
            expected_triplets - expected_deprecated, expected_models, graph_handler
        )

        self.assert_elasticsearch_state(expected_models, graph_handler)

    def test_one_new_triplet_creation(self, setup_graph_handler: GraphHandler):
        graph_handler = setup_graph_handler
        extraction_info = {
            "extraction_method": "Parsed_from_HF_dataset",
            "confidence": 1.0,
            "extraction_time": "2024-08-15_09-08-26",
        }
        subject = URIRef("subject")
        predicate = URIRef("predicate")
        object = URIRef("object")
        graph_handler.process_triplet(subject, predicate, object, extraction_info)

        # Ensure that the triplet was created
        new_triplet_df = graph_handler.SQLHandler.query(
            f"""
                                                          SELECT * FROM \"Triplet\" 
                                                          WHERE subject = '{str(subject.n3())}' 
                                                          AND predicate = '{str(predicate.n3())}' 
                                                          AND object = '{str(object.n3())}'
                                                          """
        )
        assert len(new_triplet_df) == 1
        # print(new_triplet_df)
        new_triplet_id = new_triplet_df.iloc[0]["id"]

        # Check if a new extraction_info was created
        new_extraction_info_df = graph_handler.SQLHandler.query(
            "SELECT * FROM \"Triplet_Extraction_Info\" WHERE method_description='Parsed_from_HF_dataset' AND extraction_confidence=1.0"
        )
        assert len(new_extraction_info_df) == 1
        new_extraction_info_id = new_extraction_info_df.iloc[0]["id"]

        # Check if a new version_range  was created
        new_version_range_df = graph_handler.SQLHandler.query(
            f"""SELECT * FROM \"Version_Range\" WHERE
                                                                        triplet_id = '{new_triplet_id}'
                                                                        AND extraction_info_id = '{new_extraction_info_id}'"""
        )
        assert len(new_version_range_df) == 1
        assert new_version_range_df.iloc[0]["use_start"] == Timestamp(
            "2024-08-15 09:08:26"
        )
        assert new_version_range_df.iloc[0]["use_end"] == Timestamp(
            "2024-08-15 09:08:26"
        )
        assert new_version_range_df.iloc[0]["deprecated"] == False
        # print(new_version_range_df)

    def test_small_graph_creation(self, setup_graph_handler: GraphHandler):
        graph_handler = setup_graph_handler
        # Read dataframe from json file
        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_1.json",
            graph_handler=graph_handler,
        )

        self.assert_dbs_states(
            expected_triplets=16,
            expected_models=2,
            expected_ranges=16,
            expected_extraction_info=2,
            expected_deprecated=0,
            graph_handler=graph_handler,
            print_df=False,
        )

    def test_small_graph_update_same_models(self, setup_graph_handler: GraphHandler):
        # The idea of this test is to evaluate if the graph is correctly updated when
        # the same models are loaded again but with changes in their properties.
        graph_handler = setup_graph_handler
        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_1.json",
            graph_handler=graph_handler,
        )

        self.assert_dbs_states(
            expected_triplets=16,
            expected_models=2,
            expected_ranges=16,
            expected_extraction_info=2,
            expected_deprecated=0,
            graph_handler=graph_handler,
            print_df=False,
        )

        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_2.json",
            graph_handler=graph_handler,
        )

        self.assert_dbs_states(
            expected_triplets=19,
            expected_models=2,
            expected_ranges=20,
            expected_extraction_info=3,
            expected_deprecated=3,
            graph_handler=graph_handler,
            print_df=False,
        )

    def test_small_graph_add_new_models(self, setup_graph_handler: GraphHandler):
        graph_handler = setup_graph_handler
        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_1.json",
            graph_handler=graph_handler,
        )
        self.assert_dbs_states(
            expected_triplets=16,
            expected_models=2,
            expected_ranges=16,
            expected_extraction_info=2,
            expected_deprecated=0,
            graph_handler=graph_handler,
            print_df=False,
        )

        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_3.json",
            graph_handler=graph_handler,
        )

        self.assert_dbs_states(
            expected_triplets=23,
            expected_models=3,
            expected_ranges=23,
            expected_extraction_info=2,
            expected_deprecated=0,
            graph_handler=graph_handler,
            print_df=False,
        )

    def test_small_graph_update_and_add_new_models(
        self, setup_graph_handler: GraphHandler
    ):
        graph_handler = setup_graph_handler
        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_1.json",
            graph_handler=graph_handler,
        )

        self.assert_dbs_states(
            expected_triplets=16,
            expected_models=2,
            expected_ranges=16,
            expected_extraction_info=2,
            expected_deprecated=0,
            graph_handler=graph_handler,
            print_df=False,
        )

        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_3.json",
            graph_handler=graph_handler,
        )

        self.assert_dbs_states(
            expected_triplets=23,
            expected_models=3,
            expected_ranges=23,
            expected_extraction_info=2,
            expected_deprecated=0,
            graph_handler=graph_handler,
            print_df=False,
        )

        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_2.json",
            graph_handler=graph_handler,
        )

        self.assert_dbs_states(
            expected_triplets=26,
            expected_models=3,
            expected_ranges=27,
            expected_extraction_info=3,
            expected_deprecated=0,
            graph_handler=graph_handler,
            print_df=False,
        )

    def test_triplet_deprecation(self, setup_graph_handler: GraphHandler):
        graph_handler = setup_graph_handler
        graph_handler.curr_update_date = datetime.strptime(
            "2024-07-16_09-14-40", "%Y-%m-%d_%H-%M-%S"
        )
        graph_handler.process_triplet(
            URIRef("http://example.com/model1"),
            URIRef("http://example.com/property1"),
            URIRef("http://example.com/object1"),
            extraction_info={
                "extraction_method": "Parsed_from_HF_dataset",
                "confidence": 1.0,
                "extraction_time": "2024-07-16_09-14-40",
            },
        )

        self.assert_sql_db_state(
            expected_triplets=1,
            expected_models=1,
            expected_ranges=1,
            expected_extraction_info=1,
            expected_deprecated=0,
            graph_handler=graph_handler,
        )

        graph_handler.curr_update_date = datetime.strptime(
            "2025-07-16_09-14-40", "%Y-%m-%d_%H-%M-%S"
        )

        graph_handler.process_triplet(
            URIRef("http://example.com/model1"),
            URIRef("http://example.com/property1"),
            URIRef("http://example.com/object2"),
            extraction_info={
                "extraction_method": "Parsed_from_HF_dataset",
                "confidence": 1.0,
                "extraction_time": "2025-07-16_09-14-40",
            },
        )
        graph_handler.deprecate_old_triplets(URIRef("http://example.com/model1"))

        self.assert_sql_db_state(
            expected_triplets=2,
            expected_models=1,
            expected_ranges=2,
            expected_extraction_info=1,
            expected_deprecated=1,
            graph_handler=graph_handler,
            print_df=False,
        )

    def test_small_graph_multiple_deprecations(self, setup_graph_handler: GraphHandler):
        graph_handler = setup_graph_handler
        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_1.json",
            graph_handler=graph_handler,
        )

        self.assert_dbs_states(
            expected_triplets=16,
            expected_models=2,
            expected_ranges=16,
            expected_extraction_info=2,
            expected_deprecated=0,
            graph_handler=graph_handler,
            print_df=False,
        )

        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_4.json",
            graph_handler=graph_handler,
        )

        self.assert_dbs_states(
            expected_triplets=16,
            expected_models=2,
            expected_ranges=16,
            expected_extraction_info=2,
            expected_deprecated=3,
            graph_handler=graph_handler,
            print_df=False,
        )

    def test_large_dataset(self, setup_graph_handler: GraphHandler):
        graph_handler = setup_graph_handler
        self.create_graph(
            "./tests/Test_files/load_files/hf_transformed_fair4ml_example.json",
            graph_handler,
        )

    # def test_malformed_input(self, setup_graph_handler: GraphHandler):
    #     graph_handler = setup_graph_handler
    #     with pytest.raises(ValueError):
    #         self.create_graph("./tests/Test_files/load_files/malformed_data.json", graph_handler)
