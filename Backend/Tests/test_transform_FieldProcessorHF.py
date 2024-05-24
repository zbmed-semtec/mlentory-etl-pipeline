import pytest
import sys
import os
import time
from typing import List, Tuple


sys.path.append('.')
from Transform.Core.FilesProcessor import FilesProcessor
from Transform.Core.QueueObserver import QueueObserver, MyQueueEventHandler
from Transform.Core.FieldProcessorHF import FieldProcessorHF

class TestFieldProcessorHF:
    """
    Test class for FieldProcessorHF
    """
    
    @pytest.fixture
    def setup_file_processor(self, request, tmp_path) -> Tuple[QueueObserver, FilesProcessor, str]:
        """
        Setup a FilesProcessor instance with a QueueObserver and a temporary directory
        
        Args:
            request: pytest request object
            tmp_path: temporary directory path
        
        Returns:
            A tuple containing the QueueObserver, FilesProcessor, and temporary directory path
        """
        
        marker = request.node.get_closest_marker("fixture_data")
        
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        #Create a "processed files" file on the test directory
        with open(test_dir / "Processed_files.txt", "w") as f:
            f.write("")
        
        processed_files_log_path = test_dir / "Processed_files.txt"
        
        fields_processor_HF = FieldProcessorHF(path_to_config_data="./Config_Data")
        
        if marker is None:
            file_processor = FilesProcessor(num_workers=2, 
                                            next_batch_proc_time=1,
                                            processed_files_log_path=processed_files_log_path,
                                            field_processor_HF=fields_processor_HF)
        else:
            file_processor = FilesProcessor(num_workers=marker.args[0], 
                                            next_batch_proc_time=marker.args[1],
                                            processed_files_log_path=processed_files_log_path,
                                            field_processor_HF=fields_processor_HF)
        # Create a QueueObserver instance
        observer = QueueObserver(watch_dir=test_dir, files_processor=file_processor)
        
        observer.start()
        
        yield observer, file_processor, test_dir
        
        observer.stop()
        file_processor = None
        observer = None
        
    @pytest.fixture
    def setup_file_processor_with_files(self, request, tmp_path) -> Tuple[QueueObserver, FilesProcessor, str]:
        """
        Setup a FilesProcessor instance with a QueueObserver and a temporary directory
        
        Args:
            request: pytest request object
            tmp_path: temporary directory path
        
        Returns:
            A tuple containing the QueueObserver, FilesProcessor, and temporary directory path
        """
        
        marker = request.node.get_closest_marker("fixture_data")
        
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        #Dummy file that has already been processed
        file_already_processed_path = os.path.join(test_dir, "new_file_0.tsv")
        with open(file_already_processed_path, "w") as f:
            f.write("Col1\tCol2\n")
            f.write("1\t2\n")
            
        #Create a "processed files" file on the test directory
        with open(test_dir / "Processed_files.txt", "w") as f:
            f.write(file_already_processed_path)
        processed_files_log_path = test_dir / "Processed_files.txt"
        
        if marker is None:
            file_processor = FilesProcessor(num_workers=2, next_batch_proc_time=1,processed_files_log_path=processed_files_log_path)
        else:
            file_processor = FilesProcessor(num_workers=marker.args[0], next_batch_proc_time=marker.args[1],processed_files_log_path=processed_files_log_path) 
        
        # Create a QueueObserver instance
        observer = QueueObserver(watch_dir=test_dir, files_processor=file_processor)
        
        observer.start()
        
        yield observer, file_processor, test_dir
        
        observer.stop()
        file_processor = None
        observer = None
    
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
            with open(file_paths[-1], "w") as f:
                f.write("Col1\tCol2\n")
                f.write("1\t2\n")
                f.write("3\t4\n")
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
    
    @pytest.mark.fixture_data(2,2)
    def test_creates_workers_on_complete_batch(self, caplog, setup_file_processor: Tuple[QueueObserver, FilesProcessor, str],  logger) -> None:
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
        _, file_processor, test_dir = setup_file_processor
        
        file_paths = []
        
        # Create files for batch processing and simulate file creation event
        file_paths.extend(self.create_files_for_batch_processing( test_dir,
                                               wait_for_response=1,
                                               files_to_create=2,
                                               start_file_num=0,
                                               logger=logger))
                        
        # Assert that the "Finished processing batch" message is logged
        assert self.count_finished_batches(caplog) == 1
        
        # Assert the each file got processed
        assert self.check_files_got_processed(file_paths,file_processor)