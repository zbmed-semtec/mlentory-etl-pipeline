from queue import Queue
from multiprocessing import Process
from typing import Callable, List

class FilesProcessor:
  def __init__(self, num_workers):
    self.queue = Queue()
    self.workers = []
    self.num_workers = num_workers
    self._create_workers()

  def _create_workers(self):
    for _ in range(self.num_workers):
      worker = Process(target=self._worker)
      worker.daemon = True
      worker.start()
      self.workers.append(worker)

  def _worker(self):
    while True:
      try:
        filename = self.queue.get()
        self.process_file(filename)  # Replace with your actual processing logic
        self.queue.task_done()
      except Exception as e:
        print(f"Error processing file: {e}")

  def process_file(self, filename):
    # Replace this with your actual file processing logic
    print(f"Processing file: {filename}")

  def add_file(self, filename):
    print("hello")
    self.queue.put(filename)

  def wait_for_completion(self):
    # Wait for all workers to finish processing tasks in the queue
    self.queue.join()

  def shutdown(self):
    # Stop all worker processes
    for worker in self.workers:
      worker.terminate()