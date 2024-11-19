from core.QueueObserver import QueueObserver
from core.FileProcessor import FileProcessor
from core.LoadProcessor import LoadProcessor
from core.dbHandler.RDFHandler import RDFHandler
from core.dbHandler.SQLHandler import SQLHandler
from core.dbHandler.IndexHandler import IndexHandler
from core.GraphHandler import GraphHandler
import argparse
import datetime
import logging
import time


def main():
    # Handling script arguments
    parser = argparse.ArgumentParser(description="Queue Observer Script")
    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        required=False,
        default="./../load_queue/",
        help="Path to the folder to observe (default: ./../load_queue/)",
    )
    args = parser.parse_args()

    # Setting up logging system
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"./loading_logs/load_{timestamp}.log"
    logging.basicConfig(
        filename=filename,
        filemode="w",
        format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    #Try to connect three times to the databases
    failed_attempts = 0
    connected = False
    sqlHandler = None
    rdfHandler = None
    elasticsearchHandler = None
    graphHandler = None
    load_processor = None
    observer = None
    
    while(failed_attempts<3 and (not connected)):
        logger.info("\n Trying to connect to DBs \n")
        print("\n Trying to connect to DBs \n")
        try:
            # Initializing the database handlers
            sqlHandler = SQLHandler(
                host="postgres",
                user="user",
                password="password",
                database="history_DB",
            )
            sqlHandler.connect()

            rdfHandler = RDFHandler(
                container_name="virtuoso",
                kg_files_directory="/../kg_files",
                _user="dba",
                _password="my_strong_password",
                sparql_endpoint="http://virtuoso:8890/sparql",
            )

            elasticsearchHandler = IndexHandler(
                es_host="elastic",
                es_port=9200,
            )

            elasticsearchHandler.initialize_HF_index(index_name="hf_models")

            # Initializing the graph creator
            graphHandler = GraphHandler(
                SQLHandler=sqlHandler,
                RDFHandler=rdfHandler,
                IndexHandler=elasticsearchHandler,
                kg_files_directory="/../kg_files",
            )

            # Initializing the load processor
            load_processor = LoadProcessor(
                SQLHandler=sqlHandler,
                RDFHandler=rdfHandler,
                IndexHandler=elasticsearchHandler,
                GraphHandler=graphHandler,
                kg_files_directory="./../kg_files",
            )
            
            logger.info("\n\n Connected to DBs successfully")
            print("\n\n Connected to DBs successfully")
            print(load_processor)
            print(graphHandler)
            
            file_processor = FileProcessor(
                processed_files_log_path="./loading_logs/Processed_files.txt",
                load_processor=load_processor,
            )
            
            observer = QueueObserver(watch_dir=args.folder, file_processor=file_processor)
            observer.start()

            load_processor.clean_DBs()

            while True:
                time.sleep(0.5)
                
        except Exception as e:
            logger.exception("Exception occurred ", e)
            failed_attempts+=1
            time.sleep(10)
            if(observer != None):
                observer.stop()
    
    if(failed_attempts == 3):
        logger.info("\n\n Failed to connect to DBs")
        print("\n\n Failed to connect to DBs")
            
            
        
        
        


if __name__ == "__main__":
    main()
