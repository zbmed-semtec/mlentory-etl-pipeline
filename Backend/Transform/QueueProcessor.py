from Core.QueueObserver import QueueObserver 
import logging
import time


if __name__ == "__main__":
    
    logging.basicConfig(filename='./Processing_Logs/transform.log', filemode='w', format='%(asctime)s %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    observer = QueueObserver("./../Transform_Queue/")
    observer.start()
    
    try:
        # Keep the script running to monitor changes
        while True:
            observer.files_processor.update_time_to_proccess()
            logger.info("LOL")
            time.sleep(0.5)
    except Exception as e:
        logger.exception("Exception occurred")
    except KeyboardInterrupt:
        logging.info("Server Shutdown")
        observer.stop()