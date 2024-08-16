from core.QueueObserver import QueueObserver
from core.FileProcessor import FileProcessor
from core.LoadProcessor import LoadProcessor
from core.dbHandler.VirtuosoHandler import VirtuosoHandler
from core.dbHandler.MySQLHandler import MySQLHandler
import argparse
import datetime
import logging
import time


def main():
    #Handling script arguments
    parser = argparse.ArgumentParser(description="Queue Observer Script")
    parser.add_argument("--folder", "-f", type=str, required=False, default="./../load_queue/",
                        help="Path to the folder to observe (default: ./../load_queue/)")
    args = parser.parse_args()

    #Setting up logging system
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'./loading_logs/load_{timestamp}.log'
    logging.basicConfig(filename=filename, filemode='w', format='%(asctime)s %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    try:
        #Initializing the database handlers
        mySQLHandler = MySQLHandler(host="mysql", 
                                    user="user",
                                    password="password123",
                                    database="MLentory_DB")
        
        virtuosoHandler = VirtuosoHandler(  container_name="code_virtuoso_1", 
                                            kg_files_directory="/../kg_files",
                                            virtuoso_user="dba", 
                                            virtuoso_password="my_strong_password",
                                            sparql_endpoint="http://virtuoso:8890/sparql"
                                            )
        #Initializing the load processor
        load_processor = LoadProcessor(mySQLHandler=mySQLHandler,virtuosoHandler=virtuosoHandler)
        
        file_processor = FileProcessor(processed_files_log_path="./loading_logs/Processed_files.txt",load_processor=load_processor)
        observer = QueueObserver(watch_dir=args.folder,file_processor=file_processor)
        observer.start()
        file_processor.process_file("./../load_queue/test copy 2.ttl")
        
        # Keep the script running to monitor changes
        # while True:
        #     time.sleep(0.5)
    except Exception as e:
        logger.exception("Exception occurred ", e)
    except KeyboardInterrupt:
        logger.info("Server Stop")

if __name__ == "__main__":
  main()