from Core.QueueObserver import QueueObserver 

import time


if __name__ == "__main__":
    observer = QueueObserver("./../Transform_Queue/")
    observer.start()
    
    try:
        # Keep the script running to monitor changes
        while True:
            observer.files_processor.update_time_to_proccess()
            time.sleep(0.5)
    except KeyboardInterrupt:
        observer.stop()