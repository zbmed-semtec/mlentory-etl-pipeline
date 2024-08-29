from multiprocessing import Process, Pool,set_start_method,get_context
import pytest
import sys
import os
import time
import pandas as pd
from typing import List, Tuple
from unittest.mock import Mock


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
    def setup_graph_creator(self) -> GraphCreator:
        mock_mySQLHandler = Mock(spec=MySQLHandler)
        mock_virtuosoHandler = Mock(spec=VirtuosoHandler)
        graph_creator = GraphCreator(mock_mySQLHandler,mock_virtuosoHandler)
        graph_creator.load_df(self.m4ml_example_dataframe)
        return graph_creator
    
    def test_basic_creation(self, setup_graph_creator: GraphCreator):
        graph_creator = setup_graph_creator
        graph_creator.create_graph()
        graph_creator.store_graph("tests/Test_files/test_graph.ttl")
        #Check if the file exists
        assert os.path.isfile("tests/Test_files/test_graph.ttl")
        
        
    
    