import os
import shutil
from rdflib.graph import Graph, ConjunctiveGraph
import logging
from datetime import datetime
from typing import Callable, List, Dict, Set
from pandas import DataFrame
from mlentory_load.dbHandler import SQLHandler, RDFHandler, IndexHandler
from mlentory_load.core.GraphHandler import GraphHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LoadProcessor:
    """
    This class is responsible for loading the data into the different databases.
    """

    def __init__(
        self,
        SQLHandler: SQLHandler,
        RDFHandler: RDFHandler,
        IndexHandler: IndexHandler,
        GraphHandler: GraphHandler,
        kg_files_directory: str,
    ):
        """
        Initializes a new LoadProcessor instance.
        """
        self.SQLHandler = SQLHandler
        self.SQLHandler.connect()
        self.RDFHandler = RDFHandler
        self.IndexHandler = IndexHandler
        self.GraphHandler = GraphHandler
        self.kg_files_directory = kg_files_directory

    def update_dbs_with_df(self, df):

        # The graph handler updates the SQL and RDF databases with the new data
        self.GraphHandler.load_df(df)
        self.GraphHandler.update_graph()

    def load_df(self, df: DataFrame, output_ttl_file_path: str = None):
        self.update_dbs_with_df(df)

        if output_ttl_file_path is not None:
            print("OUTPUT TTL FILE PATH\n", output_ttl_file_path)
            current_graph = self.GraphHandler.get_current_graph()
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_ttl_file_path = os.path.join(
                output_ttl_file_path, f"{current_date}_mlentory_graph.ttl"
            )
            current_graph.serialize(output_ttl_file_path, format="turtle")

    def print_DB_states(self):
        triplets_df = self.GraphHandler.SQLHandler.query('SELECT * FROM "Triplet"')
        ranges_df = self.GraphHandler.SQLHandler.query('SELECT * FROM "Version_Range"')
        extraction_info_df = self.GraphHandler.SQLHandler.query(
            'SELECT * FROM "Triplet_Extraction_Info"'
        )

        print("SQL TRIPlETS\n", triplets_df)
        print("SQL RANGES\n", ranges_df)
        print("SQL EXTRACTION INFO\n", extraction_info_df)

        result_graph = self.GraphHandler.get_current_graph()

        print("VIRTUOSO TRIPlETS\n")
        for i, (s, p, o) in enumerate(result_graph):
            print(f"{i}: {s} {p} {o}")

        result_count = result_graph.query(
            """SELECT (COUNT(DISTINCT ?s) AS ?count) WHERE{?s ?p ?o}"""
        )

        for triple in result_count:
            print("VIRTUOSO MODEL COUNT\n", triple.asdict()["count"]._value)

        self.GraphHandler.IndexHandler.es.indices.refresh(index="hf_models")
        result = self.GraphHandler.IndexHandler.es.search(
            index="hf_models",
            body={"query": {"match_all": {}}},
        )
        print("Check Elasticsearch: ", result, "\n")

    def clean_DBs(self):
        self.RDFHandler.reset_db()
        self.IndexHandler.clean_indices()
        self.SQLHandler.clean_all_tables()
