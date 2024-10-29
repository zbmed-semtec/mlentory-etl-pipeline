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

def print_DB_states(graph_handler):
    triplets_df = graph_handler.SQLHandler.query('SELECT * FROM "Triplet"')
    ranges_df = graph_handler.SQLHandler.query('SELECT * FROM "Version_Range"')
    extraction_info_df = graph_handler.SQLHandler.query(
        'SELECT * FROM "Triplet_Extraction_Info"'
    )

    print("SQL TRIPlETS\n", triplets_df)
    print("SQL RANGES\n", ranges_df)
    print("SQL EXTRACTION INFO\n", extraction_info_df)
    
    result_graph = graph_handler.RDFHandler.query(
        "http://virtuoso:8890/sparql",
        """CONSTRUCT { ?s ?p ?o } WHERE {GRAPH <http://example.com/data_1> {?s ?p ?o}}""",
    )
    
    print("VIRTUOSO TRIPlETS\n")
    for i, (s, p, o) in enumerate(result_graph):
        print(f"{i}: {s} {p} {o}")

    result_count = result_graph.query(
        """SELECT (COUNT(DISTINCT ?s) AS ?count) WHERE{?s ?p ?o}"""
    )
    
    print("VIRTUOSO MODEL COUNT\n", result_count)
    
    graph_handler.IndexHandler.es.indices.refresh(index="test_hf_models")
    result = graph_handler.IndexHandler.es.search(
        index="test_hf_models",
        body={"query": {"match_all": {}}},
    )
    print("Check Elasticsearch: ", result, "\n")
    
    

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

    try:
        # Initializing the database handlers
        sqlHandler = SQLHandler(
            host="postgres",
            user="user",
            password="password",
            database="history_DB",
        )
        SQLHandler.connect()
        
        rdfHandler = RDFHandler(
            container_name="virtuoso",
            kg_files_directory="/../kg_files",
            virtuoso_user="dba",
            virtuoso_password="my_strong_password",
            sparql_endpoint="http://virtuoso:8890/sparql",
        )
        
        elasticsearchHandler = IndexHandler(
            es_host="elastic",
            es_port=9200,
        )

        elasticsearchHandler.initialize_HF_index(index_name="hf_models")
        
        # Initializing the graph creator
        GraphHandler = GraphHandler(
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
            GraphHandler=GraphHandler,
            kg_files_directory="./../kg_files",
        )

        file_processor = FileProcessor(
            processed_files_log_path="./loading_logs/Processed_files.txt",
            load_processor=load_processor,
        )
        observer = QueueObserver(watch_dir=args.folder, file_processor=file_processor)
        observer.start()
        file_processor.process_file("./../load_queue/test copy.json")
        
        print_DB_states(graph_handler=GraphHandler)

        # Keep the script running to monitor changes
        # while True:
        #     time.sleep(0.5)
    except Exception as e:
        logger.exception("Exception occurred ", e)
    except KeyboardInterrupt:
        logger.info("Server Stop")


if __name__ == "__main__":
    main()
