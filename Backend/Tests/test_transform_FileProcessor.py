import pytest
import sys
import os
import time

sys.path.append('./../Transform')
from Core.QueueObserver import QueueObserver,MyQueueEventHandler
from Core.FilesProcessor import FilesProcessor


STOP_SIGNAL = "Stop Read"


class TestFileProcessor:
    
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
    
    def create_files_for_batch_processing(self,caplog_workaround,test_dir, wait_for_response,files_to_create,start_file_num,logger):
        file_paths = []
    
        with caplog_workaround():
            time.sleep(0.1)
            # Simulate multiple file creation event
            for file_num in range(files_to_create):
                file_path = f"new_file_{start_file_num+file_num}.tsv"
                file_paths.append(os.path.join(test_dir, file_path))
                with open(file_paths[-1], "w") as f:
                    f.write("Test content")
            time.sleep(wait_for_response)
            logger.info(STOP_SIGNAL)
        return file_paths

    def wait_for_next_batch_processing(self, caplog_workaround,file_processor, logger, waiting_period, wait_for_response):
        with caplog_workaround():
            time.sleep(0.1)
            for _ in range(waiting_period):
                file_processor.update_time_to_process()
            time.sleep(wait_for_response)
            logger.info(STOP_SIGNAL)
        

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
    def test_creates_workers_on_complete_batch(self,caplog,setup_file_processor,caplog_workaround,logger):
        _, _, test_dir = setup_file_processor
        
        file_paths = []
        
        #This tests can fail if the wait_for_response is not setup correctly, 
        # the events expected may not be logged and thus the test will fail. 
        file_paths.extend(self.create_files_for_batch_processing(caplog_workaround,test_dir,
                                               wait_for_response=0.4,
                                               files_to_create=2,
                                               start_file_num=0,
                                               logger=logger))
                        
       
        assert "Finished processing batch\n" in caplog.text
        
        # Assert the log messages for each file
        for file_path in file_paths:
            assert f"Processing file: {file_path}" in caplog.text
        
        
    @pytest.mark.fixture_data(2,2)
    def test_creates_workers_on_multiple_batches(self,caplog,setup_file_processor,caplog_workaround,logger):
        _, _, test_dir = setup_file_processor
        
        file_paths = []
        
        #This tests can fail if the wait_for_response is not setup correctly, 
        # the events expected may not be logged and thus the test will fail. 
        file_paths.extend(self.create_files_for_batch_processing(caplog_workaround,test_dir,
                                               wait_for_response=1,
                                               files_to_create=6,
                                               start_file_num=0,
                                               logger=logger))
        logger.info(STOP_SIGNAL)
        
        #Check on the logs registered in the conftest.py file how many times a batch was processed.
        cnt_batchs = 0
        for record in caplog.records:
            if "conftest" in record.pathname:
                if "Finished processing batch" in record.msg:
                    cnt_batchs +=1
            
        assert cnt_batchs == 3
        
        # Assert the log messages for each file
        for file_path in file_paths:
            assert f"Processing file: {file_path}" in caplog.text    
    
    
    @pytest.mark.fixture_data(3,10)
    def test_creates_workers_after_waiting(self,caplog,setup_file_processor,caplog_workaround,logger):
        _, file_processor, test_dir = setup_file_processor
        
        file_paths = []
        
        #This tests can fail if the wait_for_response is not setup correctly, 
        # the events expected may not be logged and thus the test will fail. 
        file_paths.extend(self.create_files_for_batch_processing(caplog_workaround,test_dir,
                                               wait_for_response=0,
                                               files_to_create=2,
                                               start_file_num=0,
                                               logger=logger))
        
        self.wait_for_next_batch_processing(caplog_workaround,file_processor, logger,
                                            waiting_period = 10,
                                            wait_for_response = 0.4)
        
        assert file_processor.curr_waiting_time == file_processor.next_batch_proc_time
        assert "Finished processing batch\n" in caplog.text
        # Assert the log messages for each file
        for file_path in file_paths:
            assert f"Processing file: {file_path}" in caplog.text 
    
    @pytest.mark.fixture_data(2,10)
    def test_creates_workers_after_batch_and_waiting(self,caplog,setup_file_processor,caplog_workaround,logger):
        _, file_processor, test_dir = setup_file_processor
        
        file_paths = []
        
        #This tests can fail if the wait_for_response is not setup correctly, 
        # the events expected may not be logged and thus the test will fail. 
        file_paths.extend(self.create_files_for_batch_processing(caplog_workaround,test_dir,
                                               wait_for_response=0.7,
                                               files_to_create=5,
                                               start_file_num=0,
                                               logger=logger))
        
        self.wait_for_next_batch_processing(caplog_workaround,file_processor, logger,
                                            waiting_period = 10,
                                            wait_for_response = 0.4)
        
        # Assert the current waiting time has reseted 
        assert file_processor.curr_waiting_time == file_processor.next_batch_proc_time
        
        #Check on the logs registered in the conftest.py file to count how many times a batch was processed.
        cnt_batchs = 0
        
        for record in caplog.records:
            if "conftest" in record.pathname:
                if "Finished processing batch" in record.msg:
                    cnt_batchs +=1  
        
        assert cnt_batchs == 3
        
        # Assert the log messages for each file
        for file_path in file_paths:
            assert f"Processing file: {file_path}" in caplog.text
        
        
    @pytest.mark.fixture_data(2,10)    
    def test_no_workers_if_no_new_files(self,caplog,setup_file_processor,caplog_workaround,logger): 
        _, file_processor, test_dir = setup_file_processor
        
        self.wait_for_next_batch_processing(caplog_workaround,file_processor, logger,
                                            waiting_period = 5,
                                            wait_for_response = 0)
        
        # Assert the current waiting time has not changed 
        assert file_processor.curr_waiting_time == file_processor.next_batch_proc_time
        
        self.wait_for_next_batch_processing(caplog_workaround,file_processor, logger,
                                            waiting_period = 6,
                                            wait_for_response = 0)
        # Assert the current waiting time still has not changed 
        assert file_processor.curr_waiting_time == file_processor.next_batch_proc_time
        
        # Assert the current the processing logs are clean
        cnt_batchs = 0
        
        for record in caplog.records:
            if "conftest" in record.pathname:
                if "Finished processing batch" in record.msg:
                    cnt_batchs +=1
        
        assert cnt_batchs == 0
    
    @pytest.mark.fixture_data(2,10)
    def test_waiting_time_goes_down(self,caplog,setup_file_processor,caplog_workaround,logger):
        _, file_processor, test_dir = setup_file_processor
        
        file_paths = []
        
        #This tests can fail if the wait_for_response is not setup correctly, 
        # the events expected may not be logged and thus the test will fail. 
        file_paths.extend(self.create_files_for_batch_processing(caplog_workaround,test_dir,
                                               wait_for_response=0,
                                               files_to_create=1,
                                               start_file_num=0,
                                               logger=logger))
        
        self.wait_for_next_batch_processing(caplog_workaround,file_processor, logger,
                                            waiting_period = 5,
                                            wait_for_response = 0)
           
        # Assert the current waiting time is going down
        assert file_processor.curr_waiting_time == file_processor.next_batch_proc_time-5
        
        self.wait_for_next_batch_processing(caplog_workaround,file_processor, logger,
                                            waiting_period = 5,
                                            wait_for_response = 0.4)
           
        # Assert the current waiting time resets
        assert file_processor.curr_waiting_time == file_processor.next_batch_proc_time
        
        #Check on the logs registered in the conftest.py file to count how many times a batch was processed.
        cnt_batchs = 0
        
        for record in caplog.records:
            if "conftest" in record.pathname:
                if "Finished processing batch" in record.msg:
                    cnt_batchs +=1  
        
        assert cnt_batchs == 1
        
        # Assert the log messages for each file
        for file_path in file_paths:
            assert f"Processing file: {file_path}" in caplog.text