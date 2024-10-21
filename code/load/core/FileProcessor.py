from typing import Callable, List, Dict, Set
from rdflib.graph import ConjunctiveGraph as Graph
from rdflib.store import Store
from rdflib.plugin import get as plugin
from rdflib.term import URIRef, Literal
import logging
import time
import pandas as pd
import traceback
import os

if "app_test" in os.getcwd():
    from load.core.LoadProcessor import LoadProcessor
else:
    from core.LoadProcessor import LoadProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FileProcessor:
    """
    This class is responsible for receiving the data and loading it into the different databases.
    """

    def __init__(self, processed_files_log_path: str, load_processor: LoadProcessor):
        """
        Initializes a new FileProcessor instance.
        """

        self.processed_files_log_path = processed_files_log_path
        self.processed_files: Set = set()
        self.load_processor = load_processor

        # Getting current processed files
        with open(self.processed_files_log_path, "r") as file:
            for line in file:
                self.processed_files[line.strip()] = 1

    def process_file(self, filename: str) -> None:
        """
        Processes a file and loads its data into the different databases.

        Args:
        filename (str): The path to the file to be processed.
        """
        if filename not in self.processed_files:
            try:
                logger.info(f"Processing file: {filename}")
                # print(f"Processing file: {filename}")
                if filename.endswith(".tsv"):
                    df = pd.read_csv(filename, sep="\t", usecols=lambda x: x != 0)
                elif filename.endswith(".json"):
                    m4ml_models_df = pd.read_json(filename)
                elif filename.endswith(".ttl"):
                    g = Graph()
                    g.parse(filename, format="turtle")
                    turtle_df = pd.DataFrame(list(g.triples((None, None, None))))
                    print(turtle_df.head())
                else:
                    raise ValueError("Unsupported file type")

                self.load_processor.update_dbs_with_df(df=m4ml_models_df)

                # self.processed_files_in_last_batch.append(filename)
                logger.info(f"Finished processing: {filename}")
                # When the file is being processed you need to keep in mind
            except Exception as e:
                print(f"Error processing file: {traceback.format_exc()}")
                logger.exception(f"Error processing file: {traceback.format_exc()}")
