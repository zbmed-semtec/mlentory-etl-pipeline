from multiprocessing import Process, Pool
from typing import Callable, List
import logging
import time

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
    """

    def __init__(self, num_workers: int, next_batch_proc_time: int):
        """
        Initializes a new FilesProcessor instance.

        Args:
            num_workers (int): The number of worker processes to use.
        """
        self.files_to_proc: List[str] = []
        self.num_workers = num_workers
        self.next_batch_proc_time = next_batch_proc_time
        self.curr_waiting_time = 0

    def create_workers(self) -> None:
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

        # start_time = time.perf_counter()

        for filename in files_in_proc:
            worker = Process(target=self.process_file, args=(filename,))
            # worker.daemon = True  # Not recommended for most use cases
            worker.start()
            workers.append(worker)

        for worker in workers:
            worker.join()

        for worker in workers:
            worker.terminate()  # Best practice to terminate even if join() finishes first

        # elapsed_time = time.perf_counter() - start_time
        # print(f"{__file__} executed in {elapsed_time:0.2f} seconds.")
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
            time.sleep(0.3)
            logger.info(f"Processing file: {filename}")
        except Exception as e:
            logger.exception(f"Error processing file: {e}")

    def add_file(self, filename: str) -> None:
        """
        Adds a filename to the processing queue.

        Args:
            filename (str): The name of the file to be added.
        """
        self.files_to_proc.append(filename)

        if len(self.files_to_proc) == self.num_workers:
            self.create_workers()

    def update_time_to_process(self) -> None:
        """
        Updates the time remaining before processing the next batch of files.

        Also triggers processing if the timer reaches zero or there are no jobs left.
        """
        self.curr_waiting_time -= 1

        if (len(self.files_to_proc)== 0) or (self.curr_waiting_time == 0):
            self.curr_waiting_time = self.next_batch_proc_time  # Reset timer
            if self.files_to_proc:
                self.create_workers()
