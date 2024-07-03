from multiprocessing import Process, Pool,set_start_method,get_context
from typing import Callable, List, Dict
import traceback
import logging
from datetime import datetime
import time
import pandas as pd
import os

if("app_test" in os.getcwd()):
    from transform.core.FieldProcessorHF import FieldProcessorHF
    from transform.core.GraphCreator import GraphCreator
else:
    from core.FieldProcessorHF import FieldProcessorHF
    from core.GraphCreator import GraphCreator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FilesProcessor:
    """
    This class processes a list of files using a specified number of worker processes.

    Attributes:
        files_to_proc (List[str]): A list of filenames to be processed.
        num_workers (int): The number of worker processes to use for processing.
        next_batch_proc_time (int): Time (in seconds) to wait before processing the next batch of files (if no new files are added).
        curr_waiting_time (int): Time (in seconds) remaining before processing the next batch of files.
        processed_files_log_path (str): The path to a file that keeps track of the processed files
        field_processor_HF (FieldProcessorHF): An instance of the FieldProcessorHF class for processing the fields of the files that come from HuggingFace.
        self.processed_files (set): A set of files that have been processed.
        self.processed_files_in_last_batch (list): A list of files that were processed in the last batch.
    """

    def __init__(self, num_workers: int, 
                 next_batch_proc_time: int,
                 processed_files_log_path: str,
                 load_queue_path: str,
                 field_processor_HF: FieldProcessorHF,
                 graph_creator: GraphCreator):
        """
        Initializes a new FilesProcessor instance.

        Args:
            num_workers (int): The number of worker processes to use.
        """
        # set_start_method("spawn", force=True)
        manager = get_context('spawn').Manager()
        self.files_to_proc: List[str] = []
        self.num_workers: int = num_workers
        self.next_batch_proc_time: int = next_batch_proc_time
        self.curr_waiting_time: int = next_batch_proc_time
        self.processed_files_log_path: str = processed_files_log_path
        self.processed_files: Dict = manager.dict()
        self.processed_files_in_last_batch: List = manager.list()
        self.processed_models: List = manager.list()
        self.field_processor_HF: FieldProcessorHF = field_processor_HF
        self.load_queue_path: str = load_queue_path
        self.graph_creator: GraphCreator = graph_creator
        #Getting current processed files
        with open(self.processed_files_log_path, 'r') as file:
            for line in file:
                self.processed_files[line.strip()] = 1
        
        # print(self.processed_files)
        

    def process_batch(self) -> None:
        """
        Creates and starts worker processes to process the files in the queue.

        Raises:
            ValueError: If there are no files to process.
        """
        if not self.files_to_proc:
            raise ValueError("No files to process")

        workers: List[Process] = []
        files_in_proc = self.files_to_proc.copy()  # Avoid modifying original list during iteration
        self.files_to_proc = []

        start_time = time.perf_counter()

        for filename in files_in_proc:
            worker = Process(target=self.process_file, args=(filename,))
            # worker.daemon = True  # Not recommended for most use cases
            worker.start()
            workers.append(worker)

        for worker in workers:
            worker.join()

        for worker in workers:
            worker.terminate()  # Best practice to terminate even if join() finishes first

        for filename in self.processed_files_in_last_batch:
            self.processed_files[filename] = 1
            with open(self.processed_files_log_path, 'a') as file:
                file.write(filename + '\n')
        
        m4ml_models_df =  pd.DataFrame(list(self.processed_models))
        
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Get current date and time

        filename_tsv = f"{self.load_queue_path}/{now}_Transformed_HF_fair4ml_schema_Dataframe.tsv"  # Create new filename
        filename_json = f"{self.load_queue_path}/{now}_Transformed_HF_fair4ml_schema_Dataframe.json"  # Create new filename
        
        m4ml_models_df.to_csv(filename_tsv,sep="\t")
        m4ml_models_df.to_json(filename_json,orient="records",indent=4)
        
        end_time = time.perf_counter()-start_time
        
        print(end_time)
        
        logger.info("Finished processing batch\n")

    def process_file(self, filename: str) -> None:
        """
        Processes a single file.

        Args:
            filename (str): The name of the file to process.

        Raises:
            Exception: If an error occurs during processing.
        """
        try:
            logger.info(f"Processing file: {filename}")
            # print(f"Processing file: {filename}")
            if filename.endswith(".tsv"):
                df = pd.read_csv(filename, sep="\t", usecols=lambda x: x != 0)
            elif filename.endswith(".json"):
                df = pd.read_json(filename)
            else:
                raise ValueError("Unsupported file type")

            # Go through each row of the dataframe
            for index, row in df.iterrows():
                model_data = self.field_processor_HF.process_row(row)
                self.processed_models.append(model_data)
            
            print(model_data.head())
                
            self.processed_files_in_last_batch.append(filename)
            logger.info(f"Finished processing: {filename}")
            #When the file is being processed you need to keep in mind 
        except Exception as e:
            print(f"Error processing file: {traceback.format_exc()}")
            logger.exception(f"Error processing file: {traceback.format_exc()}")

    def add_file(self, filename: str) -> None:
        """
        Adds a filename to the processing queue.

        Args:
            filename (str): The name of the file to be added.
        """
        # print("Are we good? ",filename)
        # print(self.processed_files)
        if(filename not in self.processed_files):
            self.files_to_proc.append(filename)
            
            if len(self.files_to_proc) == self.num_workers:
                self.process_batch()

    def update_time_to_process(self) -> None:
        """
        Updates the time remaining before processing the next batch of files.

        Also triggers processing if the timer reaches zero or there are no jobs left.
        """
        self.curr_waiting_time -= 1
        # print(self.curr_waiting_time)
        if (len(self.files_to_proc) == 0) or (self.curr_waiting_time == 0):
            self.curr_waiting_time = self.next_batch_proc_time  # Reset timer
            if self.files_to_proc:
                self.process_batch()
