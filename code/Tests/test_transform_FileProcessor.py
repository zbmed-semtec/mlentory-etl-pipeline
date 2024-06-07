import pytest
import sys
import os
import time
from typing import List, Tuple
from unittest.mock import Mock


sys.path.append('.')
from Transform.Core.FilesProcessor import FilesProcessor
from Transform.Core.QueueObserver import QueueObserver, MyQueueEventHandler
from Transform.Core.FieldProcessorHF import FieldProcessorHF
from Transform.Core.GraphCreator import GraphCreator


STOP_SIGNAL = "Stop Read"

        
# Modify this variable depending on the average time to process one file in the system.
AVRG_TIME_TO_PROCESS_FILE = 0.15

class TestFileProcessor:
    """
    Test class for FileProcessor
    """
    
    @pytest.fixture
    def setup_file_processor(self,mocker, request, tmp_path) -> Tuple[QueueObserver, FilesProcessor, str]:
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
        
        load_queue_dir = test_dir / "load_queue"
        load_queue_dir.mkdir()
        
        #Create a "processed files" file on the test directory
        with open(test_dir / "Processed_files.txt", "w") as f:
            f.write("")
        processed_files_log_path = test_dir / "Processed_files.txt"
        
        
        # fields_processor_HF = FieldProcessorHF(path_to_config_data="./Config_Data")
        mock_field_processor_hf = Mock(spec=FieldProcessorHF)
        
        if marker is None:
            file_processor = FilesProcessor(num_workers=2, 
                                            next_batch_proc_time=1,
                                            processed_files_log_path=processed_files_log_path,
                                            load_queue_path=load_queue_dir,
                                            field_processor_HF=mock_field_processor_hf)
        else:
            file_processor = FilesProcessor(num_workers=marker.args[0], 
                                            next_batch_proc_time=marker.args[1],
                                            processed_files_log_path=processed_files_log_path,
                                            load_queue_path=load_queue_dir,
                                            field_processor_HF=mock_field_processor_hf) 
        
        file_processor.processed_models = []
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
        
        load_queue_dir = test_dir / "load_queue"
        load_queue_dir.mkdir()
        
        #Dummy file that has already been processed
        file_already_processed_path = os.path.join(test_dir, "new_file_0.tsv")
        with open(file_already_processed_path, "w") as f:
            f.write("Col1\tCol2\n")
            f.write("1\t2\n")
            
        #Create a "processed files" file on the test directory
        with open(test_dir / "Processed_files.txt", "w") as f:
            f.write(file_already_processed_path)
            
        processed_files_log_path = test_dir / "Processed_files.txt"
        
        mock_field_processor_hf = Mock(spec=FieldProcessorHF)
        
        if marker is None:
            file_processor = FilesProcessor(num_workers=2, 
                                            next_batch_proc_time=1,
                                            processed_files_log_path=processed_files_log_path,
                                            load_queue_path=load_queue_dir,
                                            field_processor_HF=mock_field_processor_hf)
        else:
            file_processor = FilesProcessor(num_workers=marker.args[0], 
                                            next_batch_proc_time=marker.args[1],
                                            processed_files_log_path=processed_files_log_path,
                                            load_queue_path=load_queue_dir,
                                            field_processor_HF=mock_field_processor_hf)
        
        file_processor.processed_models = []
        
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
        print("PROCESSED FILES!",file_processor.processed_files)
        print("FILES CREATED!",file_paths)
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
        

    @pytest.mark.fixture_data(3,2)
    def test_add_file_adds_files(self, setup_file_processor: Tuple[QueueObserver, FilesProcessor, str]) -> None:
        """
        Test that adding a file adds it to the files_to_proc list
        
        
        Args:
            pytest.mark.fixture_data(num_workers,next_batch_proc_time): Is a decorator to send data to the pytest fixtures.
                num_workers: The number of threads the file_processor will use.
                next_batch_proc_time: Waiting period for next batch processing.
            setup_file_processor: A tuple containing the QueueObserver, FilesProcessor, and test directory.
        """
        _, file_processor, test_dir = setup_file_processor
        
        # Wait for 0.1 seconds
        time.sleep(0.1)
        # Simulate file creation event
        file_path = os.path.join(test_dir, "new_file.tsv")
        with open(file_path, "w") as f:
            f.write("Test content")
        # Simulate another file creation event
        file_path = os.path.join(test_dir, "new_file_2.tsv")
        with open(file_path, "w") as f:
            f.write("Test content")
        # Wait for 0.1 seconds
        time.sleep(0.1)
        
        # Assert that the files_to_proc list has 2 files
        assert len(file_processor.files_to_proc) == 2
        
    @pytest.mark.fixture_data(3,2)
    def test_add_file_same_name_fails(self, setup_file_processor: Tuple[QueueObserver, FilesProcessor, str]) -> None:
        """
        Test that adding a file adds it to the files_to_proc list
        
        Args:
            pytest.mark.fixture_data(num_workers,next_batch_proc_time): Is a decorator to send data to the pytest fixtures.
                num_workers: The number of threads the file_processor will use.
                next_batch_proc_time: Waiting period for next batch processing.
            setup_file_processor: A tuple containing the QueueObserver, FilesProcessor, and test directory.
        """
        _, file_processor, test_dir = setup_file_processor
        
        # Wait for 0.1 seconds
        time.sleep(0.1)
        # Simulate file creation event
        file_path = os.path.join(test_dir, "new_file.tsv")
        with open(file_path, "w") as f:
            f.write("Test content")
        
        # Simulate another file creation event
        file_path = os.path.join(test_dir, "new_file.tsv")
        with open(file_path, "w") as f:
            f.write("Test content")
        # Wait for 0.1 seconds
        time.sleep(0.1)
        
        # Assert that the files_to_proc list has 2 files
        assert len(file_processor.files_to_proc) == 1
    
    
    @pytest.mark.fixture_data(2,2)
    def test_add_file_after_processing_same_name_fails(self, caplog, setup_file_processor: Tuple[QueueObserver, FilesProcessor, str],  logger) -> None:
        """
        Test that adding a file after processing one with the same name fails.
        
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
                                               wait_for_response=AVRG_TIME_TO_PROCESS_FILE*2,
                                               files_to_create=2,
                                               start_file_num=0,
                                               logger=logger))
                        
        # Assert that the "Finished processing batch" message is logged
        assert self.count_finished_batches(caplog) == 1
        
        # Assert the each file got processed
        assert self.check_files_got_processed(file_paths,file_processor)
        assert len(file_processor.processed_files) == 2
        
        # Create files for batch processing with the same file names
        file_paths.extend(self.create_files_for_batch_processing( test_dir,
                                               wait_for_response=0.1,
                                               files_to_create=2,
                                               start_file_num=0,
                                               logger=logger))
        
        # Assert that there is only on "Finished processing batch" message
        assert self.count_finished_batches(caplog) == 1
     
    @pytest.mark.fixture_data(2,2)
    def test_not_add_file_already_processed(self, caplog, setup_file_processor_with_files: Tuple[QueueObserver, FilesProcessor, str],  logger) -> None:
        """
        Test that a file that has already been processed is not added to the files_to_proc list.
        
        Args:
            pytest.mark.fixture_data(num_workers,next_batch_proc_time): Is a decorator to send data to the pytest fixtures.
                num_workers: The number of threads the file_processor will use.
                next_batch_proc_time: Waiting period for next batch processing.
            caplog: A pytest fixture to capture and work with logs.
            setup_file_processor_with_files: A tuple containing the QueueObserver, FilesProcessor, and test directory. The files_to_proc list will be populated with 1 file.
            caplog_workaround: A pytest fixture to capture and work with logs in multiprocessing instances
            logger: An object for logging messages
        """
        _, file_processor, test_dir = setup_file_processor_with_files
        
        file_paths = []
        
        # Create files for batch processing and simulate file creation event
        file_paths.extend(self.create_files_for_batch_processing( test_dir,
                                               wait_for_response=AVRG_TIME_TO_PROCESS_FILE*2,
                                               files_to_create=3,
                                               start_file_num=0,
                                               logger=logger))
                        
        # Assert that the "Finished processing batch" message is logged
        assert self.count_finished_batches(caplog) == 1
        
        # Assert that each file not processed before, got processed.
        assert self.check_files_got_processed(file_paths[1:3],file_processor)
        
        assert len(file_processor.processed_files) == 3
        
        # Create files for batch processing with the same file names
        file_paths.extend(self.create_files_for_batch_processing( test_dir,
                                               wait_for_response=0.5,
                                               files_to_create=3,
                                               start_file_num=0,
                                               logger=logger))
        
        assert len(file_processor.processed_files) == 3
        
        # Assert that there is only on "Finished processing batch" message
        assert self.count_finished_batches(caplog) == 1   
        
    
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
        
    @pytest.mark.fixture_data(2,2)
    def test_creates_workers_on_multiple_batches(self, caplog, setup_file_processor: Tuple[QueueObserver, FilesProcessor, str],  logger) -> None:
        """
        Test that workers are created on multiple batches
        
        Args:
            pytest.mark.fixture_data(num_workers,next_batch_proc_time): Is a decorator to send data to the pytest fixtures.
                num_workers: The number of threads the file_processor will use.
                next_batch_proc_time: Waiting period for next batch processing.
            caplog: A pytest fixture to capture and work with logs.
            setup_file_processor: A tuple containing the QueueObserver, FilesProcessor, and test directory
            caplog_workaround: A pytest fixture to capture and work with logs in multiprocessing instances
            logger: An object for logging messages
        """
        # Unpack the setup_file_processor tuple
        _, file_processor, test_dir = setup_file_processor
        
        # Initialize an empty list to store file paths
        file_paths = []
        
        # Create files for batch processing and simulate file creation event
        file_paths.extend(self.create_files_for_batch_processing( test_dir,
                                                            wait_for_response=2,
                                                            files_to_create=6,
                                                            start_file_num=0,
                                                            logger=logger))
               
        # Assert that 3 batches were processed
        assert self.count_finished_batches(caplog) == 3
        
        # Assert the each file got processed
        assert self.check_files_got_processed(file_paths,file_processor)


    @pytest.mark.fixture_data(3,10)
    def test_creates_workers_after_waiting(self, caplog, setup_file_processor: Tuple[QueueObserver, FilesProcessor, str],  logger) -> None:
        """
        Test that workers are created after waiting
        
        Args:
            pytest.mark.fixture_data(num_workers,next_batch_proc_time): Is a decorator to send data to the pytest fixtures.
                num_workers: The number of threads the file_processor will use.
                next_batch_proc_time: Waiting period for next batch processing.
            caplog: A pytest fixture to capture and work with logs.
            setup_file_processor: A tuple containing the QueueObserver, FilesProcessor, and test directory
            caplog_workaround: A pytest fixture to capture and work with logs in multiprocessing instances
            logger: An object for logging messages
        """
        # Unpack the setup_file_processor tuple
        _, file_processor, test_dir = setup_file_processor
        
        file_paths = []
        
        #This test can fail if the wait_for_response is not setup correctly, 
        # the events expected may not be logged and thus the test will fail. 
        file_paths.extend(self.create_files_for_batch_processing(test_dir,
                                               wait_for_response=0,
                                               files_to_create=2,
                                               start_file_num=0,
                                               logger=logger))
        
        self.wait_for_next_batch_processing(file_processor, logger,
                                            waiting_period = 10,
                                            wait_for_response = 0.4)
        
        assert file_processor.curr_waiting_time == file_processor.next_batch_proc_time
        
        assert self.count_finished_batches(caplog) == 1
        
        # Assert the each file got processed
        assert self.check_files_got_processed(file_paths,file_processor)
    
    @pytest.mark.fixture_data(2,10)
    def test_creates_workers_after_batch_and_waiting(self, caplog, setup_file_processor: Tuple[QueueObserver, FilesProcessor, str],  logger) -> None:
        """
        Test that workers are created after batch and waiting
        
        Args:
            pytest.mark.fixture_data(num_workers,next_batch_proc_time): Is a decorator to send data to the pytest fixtures.
                num_workers: The number of threads the file_processor will use.
                next_batch_proc_time: Waiting period for next batch processing.
            caplog: A pytest fixture to capture and work with logs.
            setup_file_processor: A tuple containing the QueueObserver, FilesProcessor, and test directory
            caplog_workaround: A pytest fixture to capture and work with logs in multiprocessing instances
            logger: An object for logging messages
        """
        # Unpack the setup_file_processor tuple
        _, file_processor, test_dir = setup_file_processor
        
        # Initialize an empty list to store file paths
        file_paths = []
        
        # Create files for batch processing and simulate file creation event
        file_paths.extend(self.create_files_for_batch_processing( test_dir,
                                                            wait_for_response=2,
                                                            files_to_create=5,
                                                            start_file_num=0,
                                                            logger=logger))
        
        # Wait for next batch processing
        self.wait_for_next_batch_processing( file_processor, logger,
                                            waiting_period=10,
                                            wait_for_response=0.6)
        
        # Assert that the current waiting time has reset
        assert file_processor.curr_waiting_time == file_processor.next_batch_proc_time
        
        # Assert that 3 batches were processed
        assert self.count_finished_batches(caplog) == 3
        
        # Assert the each file got processed
        assert self.check_files_got_processed(file_paths,file_processor)


    @pytest.mark.fixture_data(2,10)
    # @pytest.mark.skip
    def test_no_workers_if_no_new_files(self, caplog, setup_file_processor: Tuple[QueueObserver, FilesProcessor, str],  logger) -> None:
        """
        Test that no workers are created if no new files are present
        
        Args:
            pytest.mark.fixture_data(num_workers,next_batch_proc_time): Is a decorator to send data to the pytest fixtures.
                num_workers: The number of threads the file_processor will use.
                next_batch_proc_time: Waiting period for next batch processing.
            caplog: A pytest fixture to capture and work with logs.
            setup_file_processor: A tuple containing the QueueObserver, FilesProcessor, and test directory
            caplog_workaround: A pytest fixture to capture and work with logs in multiprocessing instances
            logger: An object for logging messages
        """
        # Unpack the setup_file_processor tuple
        _, file_processor, test_dir = setup_file_processor
        
        # Wait for next batch processing (no files will be created)
        self.wait_for_next_batch_processing( file_processor, logger,
                                            waiting_period=5,
                                            wait_for_response=0)
        
        # Assert that the current waiting time has not changed
        assert file_processor.curr_waiting_time == file_processor.next_batch_proc_time
        
        # Wait for next batch processing again (no files will be created)
        self.wait_for_next_batch_processing( file_processor, logger,
                                            waiting_period=6,
                                            wait_for_response=0)
        
        # Assert that the current waiting time still has not changed
        assert file_processor.curr_waiting_time == file_processor.next_batch_proc_time
        
        # Assert that no batches were processed
        assert self.count_finished_batches(caplog) == 0
    
    @pytest.mark.fixture_data(2,10)
    # @pytest.mark.skip
    def test_waiting_time_goes_down(self, caplog, setup_file_processor: Tuple[QueueObserver, FilesProcessor, str],  logger) -> None:
        """
        Test that the waiting time goes down
        
        Args:
            pytest.mark.fixture_data(num_workers,next_batch_proc_time): Is a decorator to send data to the pytest fixtures.
                num_workers: The number of threads the file_processor will use.
                next_batch_proc_time: Waiting period for next batch processing.
            caplog: A pytest fixture to capture and work with logs.
            setup_file_processor: A tuple containing the QueueObserver, FilesProcessor, and test directory
            caplog_workaround: A pytest fixture to capture and work with logs in multiprocessing instances
            logger: An object for logging messages
        """
        
        # Unpack the setup_file_processor tuple
        _, file_processor, test_dir = setup_file_processor
        
        # Initialize an empty list to store file paths
        file_paths = []
        
        # Create files for batch processing and simulate file creation event
        file_paths.extend(self.create_files_for_batch_processing( test_dir,
                                                            wait_for_response=0.3,
                                                            files_to_create=1,
                                                            start_file_num=0,
                                                            logger=logger))
        
        # Wait for next batch processing
        waiting_period = 5
        for _ in range(waiting_period):
            file_processor.update_time_to_process()
        
        # Assert that the current waiting time is going down
        assert file_processor.curr_waiting_time == file_processor.next_batch_proc_time - 5

        
        self.wait_for_next_batch_processing( file_processor, logger,
                                            waiting_period=5,
                                            wait_for_response=0.4)
        
        # Assert that the current waiting time resets
        assert file_processor.curr_waiting_time == file_processor.next_batch_proc_time
        
        # Assert that 1 batch was processed
        assert self.count_finished_batches(caplog) == 1
        
        
        # Assert the each file got processed
        assert self.check_files_got_processed(file_paths,file_processor)