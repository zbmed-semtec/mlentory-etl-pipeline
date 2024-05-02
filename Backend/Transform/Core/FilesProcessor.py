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
        num_jobs_left (int): The number of remaining files to be processed.
        next_batch_proc_time (float): Time (in seconds) to wait before processing the next batch of files (if no new files are added).
    """

    def __init__(self, num_workers: int):
        """
        Initializes a new FilesProcessor instance.

        Args:
            num_workers (int): The number of worker processes to use.
        """
        self.files_to_proc: List[str] = []
        self.num_workers = num_workers
        self.num_jobs_left = 0
        self.next_batch_proc_time = 30

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
        self.num_jobs_left = 0

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

        elapsed_time = time.perf_counter() - start_time
        print(f"{__file__} executed in {elapsed_time:0.2f} seconds.")

    def process_file(self, filename: str) -> None:
        """
        Processes a single file.

        Args:
            filename (str): The name of the file to process.

        Raises:
            Exception: If an error occurs during processing.
        """
        try:
            with open("./../Transform_Queue/logs.txt", "a") as f:
                time.sleep(5)
                f.write(f"{filename}: adfadfasdfadf\n")
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
        self.num_jobs_left += 1

        if self.num_jobs_left == self.num_workers:
            self.create_workers()

    def update_time_to_process(self) -> None:
        """
        Updates the time remaining before processing the next batch of files.

        Also triggers processing if the timer reaches zero or there are no jobs left.
        """
        self.next_batch_proc_time -= 1

        if (self.num_jobs_left == 0) or (self.next_batch_proc_time == 0):
            self.next_batch_proc_time = 30  # Reset timer
            if self.files_to_proc:
                self.create_workers()
