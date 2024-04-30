from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer
from queue import *
from Core.FilesProcessor import FilesProcessor


class MyEventHandler(PatternMatchingEventHandler):
  
    def __init__(self, files_processor):
      super().__init__()
      self.file_processor = files_processor
  
    """
    This class defines the logic to be executed when changes are made on the directory being watched.
    """
    def on_created(self,event):
      self.file_processor.add_file(event.src_path)      
      print(f"hey, {event.src_path} has been created!")
        
    def on_deleted(self,event):
        print(f"Someone deleted {event.src_path}!")
        

class QueueObserver:
  """
  This class observes a specified directory for file creation and deletion events.
  """

  def __init__(self, watch_dir):
    """
    Initializes the QueueObserver with the directory to watch.

    Args:
      watch_dir (str): The path to the directory to be monitored.
    """
    self.watch_dir = watch_dir
    self.files_processor = FilesProcessor(num_workers=4)
    self.event_handler = MyEventHandler(self.files_processor)
    self.observer = Observer()

  def start(self):
    """
    Starts monitoring the watch directory for events.
    """
    self.observer.schedule(self.event_handler, self.watch_dir, recursive=False)
    self.observer.start()

  def stop(self):
    """
    Stops monitoring the watch directory.
    """
    self.observer.stop()
    self.observer.join()  # Wait for all threads to finish