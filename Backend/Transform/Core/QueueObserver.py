from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from Core.FilesProcessor import FilesProcessor
import logging

logger = logging.getLogger("main_2")
logger.setLevel(logging.INFO)

class QueueObserver:
    """
    This class observes a specified directory for file creation and deletion events.

    Attributes:
        watch_dir (str): The path to the directory to be monitored.
        files_processor (FilesProcessor): An instance of the FilesProcessor class used for processing files.
        event_handler (MyEventHandler): An instance of the MyEventHandler class for handling file system events.
        observer (Observer): An instance of the Observer class for monitoring the directory.
    """

    def __init__(self, watch_dir: str, files_processor: FilesProcessor):
        """
        Initializes a new QueueObserver instance.

        Args:
            watch_dir (str): The path to the directory to be monitored.
        """
        self.watch_dir = watch_dir
        self.files_processor = files_processor
        self.event_handler = MyQueueEventHandler(self.files_processor)
        self.observer = Observer()

    def start(self) -> None:
        """
        Starts monitoring the watch directory for events.
        """
        self.observer.schedule(self.event_handler, self.watch_dir, recursive=False)
        self.observer.start()

    def stop(self) -> None:
        """
        Stops monitoring the watch directory.

        Waits for all threads to finish before returning.
        """
        self.observer.stop()
        self.observer.join()
        
class MyQueueEventHandler(PatternMatchingEventHandler):
    """
    This class defines the logic to be executed when changes are made on the directory being watched.

    Attributes:
        file_processor (FilesProcessor): An instance of the FilesProcessor class used for processing files.
    """

    def __init__(self, files_processor: FilesProcessor):
        """
        Initializes a new MyEventHandler instance.

        Args:
            files_processor (FilesProcessor): An instance of the FilesProcessor class.
        """
        super().__init__(patterns=["*.tsv"])
        self.file_processor = files_processor

    def on_created(self, event) -> None:
        """
        Handles file creation events.

        Args:
            event (watchdog.events.FileSystemEvent): The file system event object.
        """
        logger.info(f"{event.src_path} has been added to the processing queue")
        self.file_processor.add_file(event.src_path)

    def on_deleted(self, event) -> None:
        """
        Handles file deletion events.

        Args:
            event (watchdog.events.FileSystemEvent): The file system event object.
        """
        logger.info(f"Someone deleted {event.src_path}!")