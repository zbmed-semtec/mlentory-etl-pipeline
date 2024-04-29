from Core.QueueObserver import QueueObserver 

import time


if __name__ == "__main__":
    observer = QueueObserver("./../Transform_Queue/")
    observer.start()
   
    try:
        # Keep the script running to monitor changes
        while True:
                time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()