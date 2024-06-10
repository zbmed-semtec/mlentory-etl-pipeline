from multiprocessing import Process, Pool,set_start_method,get_context
import pytest
import sys
import os
import time
import pandas as pd
from typing import List, Tuple


sys.path.append('.')
from Transform.Core.GraphCreator import GraphCreator
from Transform.Core.FilesProcessor import FilesProcessor
from Transform.Core.QueueObserver import QueueObserver, MyQueueEventHandler
from Transform.Core.FieldProcessorHF import FieldProcessorHF

class TestGraphCreator:
    """
    Test class for GraphCreator
    """
    
    @classmethod 
    def setup_class(self):
        self.m4ml_example_dataframe = pd.read_csv("./Tests/Test_files/hf_transformed_example_file.tsv", sep="\t", usecols=lambda x: x != "Unnamed: 0")
    
    @pytest.fixture
    def setup_graph_creator(self) -> GraphCreator:
        graph_creator = GraphCreator()
        graph_creator.load_df(self.m4ml_example_dataframe)
        return graph_creator
    
    def test_basic_conversion(self, setup_graph_creator: GraphCreator):
        graph_creator = setup_graph_creator
        graph_creator.create_graph()
        # serialized_graph = graph_creator.graph.serialize(format="ttl")
        # # Print the serialized TTL data
        # print("\n This is the KG ",serialized_graph.decode("utf-8"))
        
        
    
    