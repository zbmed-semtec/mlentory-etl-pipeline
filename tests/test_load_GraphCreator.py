from multiprocessing import Process, Pool,set_start_method,get_context
import pytest
import sys
import os
import time
import pandas as pd
from typing import List, Tuple
from unittest.mock import Mock
from pandas import Timestamp


sys.path.append('.')
from load.core.GraphCreator import GraphCreator
from load.core.dbHandler.MySQLHandler import MySQLHandler
from load.core.dbHandler.VirtuosoHandler import VirtuosoHandler

class TestGraphCreator:
    """
    Test class for GraphCreator
    """
    
    @classmethod 
    def setup_class(self):
        self.m4ml_example_dataframe = pd.read_json("./tests/Test_files/hf_transformed_fair4ml_example.json")
    
    @pytest.fixture
    def setup_mysql_handler(self) -> MySQLHandler:
        
        my_sql_handler = MySQLHandler(host="mysql", 
                                    user="test_user",
                                    password="test_pass",
                                    database="test_DB")
        my_sql_handler.connect()
        yield my_sql_handler
        #disconnect and close the connection
        my_sql_handler.disconnect()
    
    
    @pytest.fixture
    def setup_virtuoso_handler(self,tmp_path) -> VirtuosoHandler:
        kg_files_directory = tmp_path / "kg_files"
        kg_files_directory.mkdir()
        
        return VirtuosoHandler(  container_name="code_virtuoso_1", 
                                            kg_files_directory=kg_files_directory,
                                            virtuoso_user="dba", 
                                            virtuoso_password="my_strong_password",
                                            sparql_endpoint="http://virtuoso:8890/sparql"
                                            )
    
    @pytest.fixture
    def setup_mock_graph_creator(self) -> GraphCreator:
        mock_mySQLHandler = Mock(spec=MySQLHandler)
        mock_virtuosoHandler = Mock(spec=VirtuosoHandler)
        graph_creator = GraphCreator(mock_mySQLHandler,mock_virtuosoHandler)
        graph_creator.load_df(self.m4ml_example_dataframe)
        return graph_creator
    
    @pytest.fixture
    def setup_graph_creator(self,setup_mysql_handler,setup_mock_graph_creator) -> GraphCreator:
        #Initializing the database handlers
        graph_creator = GraphCreator(setup_mysql_handler,setup_mock_graph_creator)
        return graph_creator
    
    def test_basic_creation(self, setup_mock_graph_creator: GraphCreator):
        graph_creator = setup_mock_graph_creator
        graph_creator.create_graph()
        graph_creator.store_graph("tests/Test_files/test_graph.ttl")
        #Check if the file exists
        assert os.path.isfile("tests/Test_files/test_graph.ttl")
    
    def test_one_new_triplet_creation(self, setup_graph_creator:GraphCreator):
        graph_creator = setup_graph_creator
        extraction_info = { "extraction_method":"Parsed_from_HF_dataset",
                            "confidence":1.0,
                            "extraction_time":"2024-08-15_09-08-26"}
        graph_creator.create_triplet("subject", "relation", "object",extraction_info)
        
        #Ensure that the triplet was created
        new_triplet_df = graph_creator.mySQLHandler.query("SELECT * FROM Triplet WHERE subject = 'subject' AND relation = 'relation' AND object = 'object'")
        assert len(new_triplet_df) == 1
        print(new_triplet_df)
        new_triplet_id = new_triplet_df.iloc[0]["id"]
        
        #Check if a new extraction_info was created
        new_extraction_info_df = graph_creator.mySQLHandler.query("SELECT * FROM Triplet_Extraction_Info WHERE method_description='Parsed_from_HF_dataset' AND extraction_confidence=1.0")
        assert len(new_extraction_info_df) == 1
        print(new_extraction_info_df)
        new_extraction_info_id = new_extraction_info_df.iloc[0]["id"]
        
        #Check if a new version_range  was created
        new_version_range_df = graph_creator.mySQLHandler.query(f"""SELECT * FROM Version_Range WHERE
                                                                        triplet_id = '{new_triplet_id}'
                                                                        AND extraction_info_id = '{new_extraction_info_id}'""")
        assert len(new_version_range_df) == 1
        assert  new_version_range_df.iloc[0]["start"] == Timestamp('2024-08-15 09:08:26')
        assert  new_version_range_df.iloc[0]["end"] == Timestamp('2024-08-15 09:08:26')
        print(new_version_range_df)
        
        
    
    
        
    
    