from multiprocessing import Process, Pool, set_start_method, get_context
import pytest
import sys
import os
import json
import pandas as pd
from typing import List, Tuple
from unittest.mock import Mock
from pandas import Timestamp
from rdflib import Graph, URIRef, Literal, BNode

sys.path.append(".")
from load.core.GraphCreator import GraphCreator
from load.core.dbHandler.MySQLHandler import MySQLHandler
from load.core.dbHandler.VirtuosoHandler import VirtuosoHandler


class TestGraphCreator:
    """
    Test class for GraphCreator
    """

    @classmethod
    def setup_class(self):
        self.m4ml_example_dataframe = pd.read_json(
            "./tests/Test_files/load_files/hf_transformed_fair4ml_example.json"
        )

    @pytest.fixture
    def setup_mysql_handler(self) -> MySQLHandler:
        my_sql_handler = MySQLHandler(
            host="mysql", user="test_user", password="test_pass", database="test_DB"
        )
        my_sql_handler.connect()
        my_sql_handler.reset_all_tables()

        yield my_sql_handler
        # disconnect and close the connection
        my_sql_handler.disconnect()

    @pytest.fixture
    def setup_virtuoso_handler(self) -> VirtuosoHandler:
        kg_files_directory = "./tests/Test_files/load_files/virtuoso_data/kg_files"

        virtuosoHandler = VirtuosoHandler(
            container_name="virtuoso",
            kg_files_directory=kg_files_directory,
            virtuoso_user="dba",
            virtuoso_password="my_strong_password",
            sparql_endpoint="http://virtuoso:8890/sparql",
        )

        virtuosoHandler.reset_db()

        return virtuosoHandler

    @pytest.fixture
    def setup_mock_graph_creator(self) -> GraphCreator:
        mock_mySQLHandler = Mock(spec=MySQLHandler)
        mock_virtuosoHandler = Mock(spec=VirtuosoHandler)
        graph_creator = GraphCreator(
            mock_mySQLHandler,
            mock_virtuosoHandler,
            kg_files_directory="./tests/Test_files/load_files/virtuoso_data/kg_files",
        )
        graph_creator.load_df(self.m4ml_example_dataframe)
        return graph_creator

    @pytest.fixture
    def setup_graph_creator(
        self, setup_mysql_handler, setup_virtuoso_handler
    ) -> GraphCreator:
        # Initializing the database handlers
        graph_creator = GraphCreator(
            setup_mysql_handler,
            setup_virtuoso_handler,
            kg_files_directory=setup_virtuoso_handler.kg_files_directory,
        )
        return graph_creator

    def create_graph(self, source_file_path: str, graph_creator: GraphCreator):
        df_example = pd.read_json(source_file_path)
        graph_creator.load_df(df_example)
        graph_creator.create_rdf_graph()

    def assert_sql_db_state(
        self,
        expected_triplets: int,
        expected_models: int,
        expected_ranges: int,
        expected_extraction_info: int,
        graph_creator: GraphCreator,
        expected_deprecated: int = 0,
        print_df: bool = False,
    ):
        triplets_df = graph_creator.mySQLHandler.query("SELECT * FROM Triplet")
        ranges_df = graph_creator.mySQLHandler.query("SELECT * FROM Version_Range")
        extraction_info_df = graph_creator.mySQLHandler.query(
            "SELECT * FROM Triplet_Extraction_Info"
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
        graph_creator: GraphCreator,
        print_graph=False,
    ):
        result_graph = graph_creator.virtuosoHandler.query(
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

    # def test_basic_creation(self, setup_mock_graph_creator: GraphCreator):
    #     graph_creator = setup_mock_graph_creator
    #     graph_creator.load_df(self.m4ml_example_dataframe)
    #     graph_creator.create_rdf_graph()

    def test_one_new_triplet_creation(self, setup_graph_creator: GraphCreator):
        graph_creator = setup_graph_creator
        extraction_info = {
            "extraction_method": "Parsed_from_HF_dataset",
            "confidence": 1.0,
            "extraction_time": "2024-08-15_09-08-26",
        }
        subject = URIRef("subject")
        predicate = URIRef("predicate")
        object = URIRef("object")
        graph_creator.create_triplet_in_SQL(subject, predicate, object, extraction_info)

        # Ensure that the triplet was created
        new_triplet_df = graph_creator.mySQLHandler.query(
            f"""
                                                          SELECT * FROM Triplet 
                                                          WHERE subject = '{str(subject.n3())}' 
                                                          AND predicate = '{str(predicate.n3())}' 
                                                          AND object = '{str(object.n3())}'
                                                          """
        )
        assert len(new_triplet_df) == 1
        # print(new_triplet_df)
        new_triplet_id = new_triplet_df.iloc[0]["id"]

        # Check if a new extraction_info was created
        new_extraction_info_df = graph_creator.mySQLHandler.query(
            "SELECT * FROM Triplet_Extraction_Info WHERE method_description='Parsed_from_HF_dataset' AND extraction_confidence=1.0"
        )
        assert len(new_extraction_info_df) == 1
        new_extraction_info_id = new_extraction_info_df.iloc[0]["id"]

        # Check if a new version_range  was created
        new_version_range_df = graph_creator.mySQLHandler.query(
            f"""SELECT * FROM Version_Range WHERE
                                                                        triplet_id = '{new_triplet_id}'
                                                                        AND extraction_info_id = '{new_extraction_info_id}'"""
        )
        assert len(new_version_range_df) == 1
        assert new_version_range_df.iloc[0]["start"] == Timestamp("2024-08-15 09:08:26")
        assert new_version_range_df.iloc[0]["end"] == Timestamp("2024-08-15 09:08:26")
        assert new_version_range_df.iloc[0]["deprecated"] == False
        # print(new_version_range_df)

    def test_small_graph_creation(self, setup_graph_creator: GraphCreator):
        graph_creator = setup_graph_creator
        # Read dataframe from json file
        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_1.json",
            graph_creator=graph_creator,
        )
        self.assert_sql_db_state(
            expected_triplets=14,
            expected_models=2,
            expected_ranges=14,
            expected_extraction_info=2,
            expected_deprecated=0,
            graph_creator=graph_creator,
            print_df=True,
        )

        self.assert_virtuoso_db_state(
            expected_triplets=14,
            expected_models=2,
            graph_creator=graph_creator,
            print_graph=True,
        )

    def test_small_graph_update_same_models(self, setup_graph_creator: GraphCreator):
        graph_creator = setup_graph_creator
        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_1.json",
            graph_creator=graph_creator,
        )
        self.assert_sql_db_state(
            expected_triplets=14,
            expected_models=2,
            expected_ranges=14,
            expected_extraction_info=2,
            expected_deprecated=0,
            graph_creator=graph_creator,
            print_df=False,
        )

        self.assert_virtuoso_db_state(
            expected_triplets=14,
            expected_models=2,
            graph_creator=graph_creator,
            print_graph=False,
        )

        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_2.json",
            graph_creator=graph_creator,
        )
        self.assert_sql_db_state(
            expected_triplets=17,
            expected_models=2,
            expected_ranges=18,
            expected_extraction_info=3,
            expected_deprecated=3,
            graph_creator=graph_creator,
            print_df=False,
        )

        self.assert_virtuoso_db_state(
            expected_triplets=14,
            expected_models=2,
            graph_creator=graph_creator,
            print_graph=False,
        )

    def test_small_graph_add_new_models(self, setup_graph_creator: GraphCreator):
        graph_creator = setup_graph_creator
        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_1.json",
            graph_creator=graph_creator,
        )
        self.assert_sql_db_state(
            expected_triplets=14,
            expected_models=2,
            expected_ranges=14,
            expected_extraction_info=2,
            expected_deprecated=0,
            graph_creator=graph_creator,
            print_df=False,
        )

        self.assert_virtuoso_db_state(
            expected_triplets=14,
            expected_models=2,
            graph_creator=graph_creator,
            print_graph=False,
        )

        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_3.json",
            graph_creator=graph_creator,
        )
        self.assert_sql_db_state(
            expected_triplets=20,
            expected_models=3,
            expected_ranges=20,
            expected_extraction_info=2,
            expected_deprecated=0,
            graph_creator=graph_creator,
            print_df=False,
        )

        self.assert_virtuoso_db_state(
            expected_triplets=20,
            expected_models=3,
            graph_creator=graph_creator,
            print_graph=False,
        )

    def test_small_graph_update_and_add_new_models(
        self, setup_graph_creator: GraphCreator
    ):
        graph_creator = setup_graph_creator
        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_1.json",
            graph_creator=graph_creator,
        )
        self.assert_sql_db_state(
            expected_triplets=14,
            expected_models=2,
            expected_ranges=14,
            expected_extraction_info=2,
            expected_deprecated=0,
            graph_creator=graph_creator,
            print_df=True,
        )

        self.assert_virtuoso_db_state(
            expected_triplets=14,
            expected_models=2,
            graph_creator=graph_creator,
            print_graph=True,
        )

        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_3.json",
            graph_creator=graph_creator,
        )

        self.assert_sql_db_state(
            expected_triplets=20,
            expected_models=3,
            expected_ranges=20,
            expected_extraction_info=2,
            expected_deprecated=0,
            graph_creator=graph_creator,
            print_df=False,
        )

        self.assert_virtuoso_db_state(
            expected_triplets=20,
            expected_models=3,
            graph_creator=graph_creator,
            print_graph=False,
        )

        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_2.json",
            graph_creator=graph_creator,
        )
        self.assert_sql_db_state(
            expected_triplets=23,
            expected_models=3,
            expected_ranges=24,
            expected_extraction_info=3,
            expected_deprecated=0,
            graph_creator=graph_creator,
        )

        self.assert_virtuoso_db_state(
            expected_triplets=23,
            expected_models=3,
            graph_creator=graph_creator,
            print_graph=True,
        )

    def test_triplet_deprecation(self, setup_graph_creator: GraphCreator):
        graph_creator = setup_graph_creator
        graph_creator.curr_update_date = "2024-07-16_09-14-40"
        graph_creator.create_triplet_in_SQL(
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
            graph_creator=graph_creator,
        )

        graph_creator.curr_update_date = "2025-07-16_09-14-40"
        graph_creator.create_triplet_in_SQL(
            URIRef("http://example.com/model1"),
            URIRef("http://example.com/property1"),
            URIRef("http://example.com/object2"),
            extraction_info={
                "extraction_method": "Parsed_from_HF_dataset",
                "confidence": 1.0,
                "extraction_time": "2025-07-16_09-14-40",
            },
        )
        graph_creator.deprecate_old_triplets(URIRef("http://example.com/model1"))

        self.assert_sql_db_state(
            expected_triplets=2,
            expected_models=1,
            expected_ranges=2,
            expected_extraction_info=1,
            expected_deprecated=1,
            graph_creator=graph_creator,
            print_df=False,
        )

    def test_small_graph_multiple_deprecations(self, setup_graph_creator: GraphCreator):
        graph_creator = setup_graph_creator
        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_1.json",
            graph_creator=graph_creator,
        )

        self.assert_sql_db_state(
            expected_triplets=14,
            expected_models=2,
            expected_ranges=14,
            expected_extraction_info=2,
            expected_deprecated=0,
            graph_creator=graph_creator,
            print_df=True,
        )

        self.assert_virtuoso_db_state(
            expected_triplets=14,
            expected_models=2,
            graph_creator=graph_creator,
            print_graph=True,
        )

        self.create_graph(
            source_file_path="./tests/Test_files/load_files/hf_transformed_fair4ml_example_small_4.json",
            graph_creator=graph_creator,
        )

        self.assert_sql_db_state(
            expected_triplets=15,
            expected_models=2,
            expected_ranges=15,
            expected_extraction_info=2,
            expected_deprecated=4,
            graph_creator=graph_creator,
            print_df=False,
        )

        self.assert_virtuoso_db_state(
            expected_triplets=11,
            expected_models=2,
            graph_creator=graph_creator,
            print_graph=True,
        )

    def test_large_dataset(self, setup_graph_creator: GraphCreator):
        graph_creator = setup_graph_creator
        self.create_graph(
            "./tests/Test_files/load_files/hf_transformed_fair4ml_example.json",
            graph_creator,
        )

    # def test_malformed_input(self, setup_graph_creator: GraphCreator):
    #     graph_creator = setup_graph_creator
    #     with pytest.raises(ValueError):
    #         self.create_graph("./tests/Test_files/load_files/malformed_data.json", graph_creator)
