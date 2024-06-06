import pytest
import sys
import watchdog
import os
import time
import logging

print(os.getcwd())
sys.path.append('.')
from Transform.Core.QueueObserver import QueueObserver,MyQueueEventHandler
from Transform.Core.FilesProcessor import FilesProcessor

from unittest.mock import Mock

class TestQueueObserver:
    
    def setup_method(self, method):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    def test_on_created_calls_processor(self,mocker, caplog, tmp_path):
        
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        mock_processor = Mock(FilesProcessor)
        
        # Mock MyEventHandler to inject mock_processor
        mocker.patch('Transform.Core.QueueObserver.MyQueueEventHandler', return_value=MyQueueEventHandler(mock_processor))

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
        mocker.patch('Transform.Core.QueueObserver.MyQueueEventHandler', return_value=MyQueueEventHandler(mock_processor))
        
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