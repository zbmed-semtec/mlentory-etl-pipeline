from multiprocessing import Process, Pool,set_start_method,get_context
import pytest
import sys
import os
import time
import pandas as pd
from typing import List, Tuple


sys.path.append('.')
from Transform.Core.FilesProcessor import FilesProcessor
from Transform.Core.QueueObserver import QueueObserver, MyQueueEventHandler
from Transform.Core.FieldProcessorHF import FieldProcessorHF

class TestFieldProcessorHF:
    """
    Test class for FieldProcessorHF
    """
    
    @classmethod 
    def setup_class(self):
        self.hf_example_file = pd.read_csv("./Tests/Test_files/hf_extracted_example_file.tsv", sep="\t", usecols=lambda x: x != "Unnamed: 0")
        
    @pytest.fixture
    def setup_field_processor(self) -> Tuple[QueueObserver, FilesProcessor, str]:
        """
        Setup a FilesProcessor instance with a QueueObserver and a temporary directory
        
        Args:
            request: pytest request object
            tmp_path: temporary directory path
        
        Returns:
            A tuple containing the QueueObserver, FilesProcessor, and temporary directory path
        """
        
        fields_processor_HF = FieldProcessorHF(path_to_config_data="./Config_Data")
        
        
        return fields_processor_HF
    
    def test_conversion(self, caplog, setup_field_processor: FieldProcessorHF,  logger) -> None:
        """
        Test that workers are created on complete batch.
        
        Args:
            caplog: pytest caplog fixture for capturing logs
            setup_field_processor: fixture for setting up the FieldProcessorHF instance
        """
        df = self.hf_example_file
        field_processor = setup_field_processor
        
        manager = get_context('spawn').Manager()
        model_list = manager.list()
        for index, row in df.iterrows():
            # print(row)
            m4ml_model_data = field_processor.process_row(row)
            model_list.append(m4ml_model_data)
            print("m4ml new row: \n",m4ml_model_data)
            
        models_m4ml_df = pd.DataFrame(list(model_list))
        
        print(models_m4ml_df.head())
        
        # Assert the each file got processed
        # assert self.check_files_got_processed(file_paths,file_processor)