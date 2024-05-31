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
    
    def create_files_for_batch_processing(self,  test_dir: str, wait_for_response: float, files_to_create: int, start_file_num: int, logger) -> List[str]:
        """
        Create files for batch processing and simulate file creation event
        
        This test can fail if the wait_for_response is not setup correctly, the events expected may not be logged and thus the test will fail.
        
        Args:
            caplog_workaround: A pytest fixture to capture and work with logs in multiprocessing instances.
            test_dir: The path to the temporary directory.
            wait_for_response: Time to wait after the period (seconds).
            files_to_create: number of files to create
            start_file_num: Starting number for file naming.
            logger: An object for logging messages.
        
        Returns:
            A list of file paths for the created objects
        """
        file_paths = []
        time.sleep(0.1)
        # Simulate multiple file creation event
        for file_num in range(files_to_create):
            file_path = f"new_file_{start_file_num+file_num}.tsv"
            file_paths.append(os.path.join(test_dir, file_path))
            self.hf_example_file.to_csv(file_paths[-1], sep="\t", index=False)
            # print()
        time.sleep(wait_for_response)
        
        return file_paths

    def  wait_for_next_batch_processing(self,  file_processor: FilesProcessor, logger, waiting_period: int, wait_for_response: float) -> None:
        """
        Wait for next batch processing
        
        Args:
            caplog_workaround: A pytest fixture to capture and work with logs in multiprocessing instances.
            file_processor: The FilesProcessor instance.
            logger: An object for logging messages.
            waiting_period: Waiting period for next batch processing.
            wait_for_response: Time to wait after the period (seconds).
        """
        time.sleep(0.1)
        for _ in range(waiting_period):
            file_processor.update_time_to_process()
        time.sleep(wait_for_response) 
    
    def check_files_got_processed(self, file_paths: List[str],file_processor:FilesProcessor):
        processed = True
        for file_path in file_paths:
            if file_path not in file_processor.processed_files:
                processed = False
                break
        return processed
    
    def count_finished_batches(self,caplog):
        cnt_batches = 0
        for record in caplog.records:
            if "FilesProcessor" in record.pathname:
                if "Finished processing batch" in record.msg:
                    cnt_batches += 1
        
        return cnt_batches
    
    @pytest.mark.fixture_data(1,2)
    def test_creates_workers_on_complete_batch(self, caplog, setup_field_processor: FieldProcessorHF,  logger) -> None:
        """
        Test that workers are created on complete batch.
        
        Args:
            pytest.mark.fixture_data(num_workers,next_batch_proc_time): Is a decorator to send data to the pytest fixtures.
                num_workers: The number of threads the file_processor will use.
                next_batch_proc_time: Waiting period for next batch processing.
            caplog: A pytest fixture to capture and work with logs.
            setup_file_processor: A tuple containing the QueueObserver, FilesProcessor, and test directory
            caplog_workaround: A pytest fixture to capture and work with logs in multiprocessing instances
            logger: An object for logging messages
        """
        df = self.hf_example_file
        field_processor = setup_field_processor
        
        # print(df.head())
        # Go through each row of the dataframe
        for index, row in df.iterrows():
            m4ml_model_data = field_processor.process_row(row)
            
        print(m4ml_model_data.head())
        
        # Assert the each file got processed
        # assert self.check_files_got_processed(file_paths,file_processor)