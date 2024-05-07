import pytest
import sys
import watchdog
import os
import time
import logging
from logging import handlers
from multiprocessing import Queue

sys.path.append('./../Transform')
from Core.QueueObserver import QueueObserver,MyQueueEventHandler
from Core.FilesProcessor import FilesProcessor

from unittest.mock import Mock,patch


class TestQueueObserver:
    
    def setup_method(self, method):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    def test_on_created_calls_processor(self,mocker, caplog, tmp_path):
        
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        mock_processor = Mock(FilesProcessor)
        
        # Mock MyEventHandler to inject mock_processor
        mocker.patch('Core.QueueObserver.MyQueueEventHandler', return_value=MyQueueEventHandler(mock_processor))

        # Create a QueueObserver instance
        observer = QueueObserver(watch_dir=test_dir, files_processor=mock_processor)

        mock_event = Mock(watchdog.events.FileSystemEvent)
        mock_event.src_path = "test_dir/file.tsv"
        
        # Simulate file creation event
        observer.event_handler.on_created(mock_event)

        # Assert add_file is called with the event path
        mock_processor.add_file.assert_called_once_with(mock_event.src_path)
        assert f"{mock_event.src_path} has been added to the processing queue" in caplog.text
    
    def test_on_created_called_for_tsv_file(self, caplog, tmp_path):
        
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        mock_processor = Mock(FilesProcessor)
        
        # Create a QueueObserver instance
        observer = QueueObserver(watch_dir=test_dir, files_processor = mock_processor)
        
        observer.start()
        
        time.sleep(0.1)
        
        # Simulate file creation event
        file_path = os.path.join(test_dir, "new_file.tsv")
        with open(file_path, "w") as f:
            f.write("Test content")

        time.sleep(0.1)
        
        observer.stop()
        
        # Assert add_file is not called
        mock_processor.add_file.assert_called()
        assert f"{file_path} has been added to the processing queue" in caplog.text
    
    def test_on_deleted_logs_message(self, mocker, caplog, tmp_path):
        
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        mock_processor = Mock(FilesProcessor)
        
        # Mock MyEventHandler to inject mock_processor
        mocker.patch('Core.QueueObserver.MyQueueEventHandler', return_value=MyQueueEventHandler(mock_processor))
        
        # Create a QueueObserver instance
        observer = QueueObserver(watch_dir=test_dir, files_processor = mock_processor)

        mock_event = Mock(watchdog.events.FileSystemEvent)
        mock_event.src_path = "test_dir/file.tsv"
        
        # Simulate file deletion event
        observer.event_handler.on_deleted(mock_event)

        # Assert the log message
        assert f"Someone deleted {mock_event.src_path}!" in caplog.text
    
    def test_no_call_for_non_tsv_file(self, caplog, tmp_path):
        
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        mock_processor = Mock(FilesProcessor)
        
        # Create a QueueObserver instance
        observer = QueueObserver(watch_dir=test_dir, files_processor = mock_processor)
        
        observer.start()
        
        time.sleep(0.1)
        
        # Simulate file creation event
        with open(os.path.join(test_dir, "new_file.txt"), "w") as f:
            f.write("Test content")

        time.sleep(0.1)
        
        observer.stop()
        
        assert caplog.text == ""
    
    def test_error_on_invalid_watch_dir(self, caplog):
        
        mock_processor = Mock(FilesProcessor)
        
        # Simulate non-existent directory with context manager
        with pytest.raises(FileNotFoundError):
            observer = QueueObserver(watch_dir="/non_existent_dir", files_processor=mock_processor)
            observer.start()

class TestFileProcessor:
    
    def setup_method(self):
        print("HOLA")
        # logger = logging.getLogger(__name__)
        # logger.setLevel(logging.INFO)
    
    @pytest.fixture
    def setup_file_processor(self,request,tmp_path):
        
        marker = request.node.get_closest_marker("fixture_data")
        
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        if marker is None:
            file_processor = FilesProcessor(num_workers = 2, next_batch_proc_time = 1)
        else:
            file_processor = FilesProcessor(num_workers = marker.args[0], next_batch_proc_time = marker.args[1]) 
        
        # Create a QueueObserver instance
        observer = QueueObserver(watch_dir=test_dir, files_processor = file_processor)
        
        observer.start()
        
        yield observer, file_processor, test_dir
        
        observer.stop()
        

    @pytest.mark.fixture_data(3,2)
    def test_add_file_adds_files(self,setup_file_processor):
        _, file_processor, test_dir = setup_file_processor
        
        time.sleep(0.1)
        # Simulate file creation event
        file_path = os.path.join(test_dir, "new_file.tsv")
        with open(file_path, "w") as f:
            f.write("Test content")
         # Simulate file creation event
        file_path = os.path.join(test_dir, "new_file_2.tsv")
        with open(file_path, "w") as f:
            f.write("Test content")
        time.sleep(0.1)
        
        assert len(file_processor.files_to_proc) == 2
    
    @pytest.mark.fixture_data(2,2)
    def test_creates_workers_on_complete_batch(self,caplog,setup_file_processor,caplog_workaround):
        _, _, test_dir = setup_file_processor
        
        file_paths = []
        
        with caplog_workaround():
            time.sleep(0.1)
            # Simulate multiple file creation event
            for file_num in range(2):
                file_path = f"new_file_{file_num}.tsv"
                file_paths.append(os.path.join(test_dir, file_path))
                with open(file_paths[-1], "w") as f:
                    f.write("Test content")
            time.sleep(0.5)
            
        
        assert "Finished processing batch\n" in caplog.text
        
        # Assert the log messages for each file
        for file_path in file_paths:
            assert f"Processing file: {file_path}" in caplog.text
    
    @pytest.mark.fixture_data(2,2)
    def test_creates_workers_on_complete_batch(self,caplog,setup_file_processor,caplog_workaround):
        _, _, test_dir = setup_file_processor
        
        file_paths = []
        
        with caplog_workaround():
            time.sleep(0.1)
            # Simulate multiple file creation event
            for file_num in range(2):
                file_path = f"new_file_{file_num}.tsv"
                file_paths.append(os.path.join(test_dir, file_path))
                with open(file_paths[-1], "w") as f:
                    f.write("Test content")
            
            time.sleep(0.6)
        
        # assert "Finished processing batch\n" in caplog.text
        
        # Assert the log messages for each file
        for file_path in file_paths:
            assert f"Processing file: {file_path}" in caplog.text
        
        
    @pytest.mark.fixture_data(3,2)
    def test_creates_workers_on_three_complete_batches(self,caplog,setup_file_processor,caplog_workaround):
        _, _, test_dir = setup_file_processor
        
        file_paths = []
        
        with caplog_workaround():
            time.sleep(0.1)
            # Simulate multiple file creation event
            for file_num in range(6):
                file_path = f"new_file_{file_num}.tsv"
                file_paths.append(os.path.join(test_dir, file_path))
                with open(file_paths[-1], "w") as f:
                    f.write("Test content")
            
            time.sleep(1)
        
        print("HEREEEE:\n",caplog.text)
        #Need to divide by 2 because to get the logs from parallel process in the test environment
        #We capture 2 times the information of the process that are not inside the other process 
        assert caplog.text.count("Finished processing batch\n") == 3
        
        # Assert the log messages for each file
        for file_path in file_paths:
            assert f"Processing file: {file_path}" in caplog.text    
        