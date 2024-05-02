from multiprocessing import Process,Pool
from typing import Callable, List
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FilesProcessor:
  def __init__(self, num_workers):
    self.files_to_proc = []
    self.num_workers = num_workers
    self.num_jobs_left = 0
    self.next_batch_proc_time = 30
    
    
  def create_workers(self):
    workers = []
    
    files_in_procc = self.files_to_proc
    
    self.files_to_proc = []
    
    self.num_jobs_left = 0
    
    # print(files_in_procc)
    
    s = time.perf_counter()
    
    for file_name in files_in_procc:
      
      worker = Process(target=self.process_file,args=(file_name,))
      # worker.daemon = True
      worker.start()
      workers.append(worker)
    
    for worker in workers:
      worker.join()
    
    for worker in workers:
      worker.terminate()
    
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")

  def process_file(self, filename):
    try:
      with open("./../Transform_Queue/logs.txt", "a") as f:
          time.sleep(5)
          f.write(f"{filename}: adfadfasdfadf\n")
          logger.info(f"Processing file: {filename}")
    except Exception as e:
      logger.exception(f"Error processing file: {e}")

  
  def add_file(self, filename):
    # result = self.pool.apply_async(self.process_file,args=(filename,),callback = self.log_result)
    self.files_to_proc.append(filename)
    self.num_jobs_left += 1
    if(self.num_jobs_left == self.num_workers):
      self.create_workers()
    
    # print(result.get())

  def update_time_to_proccess(self):
    # print(self.next_batch_proc_time)
    self.next_batch_proc_time -= 1
    
    if(self.num_jobs_left == 0):
      self.next_batch_proc_time = 30
    
    if self.next_batch_proc_time == 0:
      self.next_batch_proc_time = 30
      self.create_workers()
    