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
    Main processor for loading and synchronizing model metadata across databases.

    This class coordinates the loading of data into different database systems:
    - PostgreSQL for relational data
    - Virtuoso for RDF graphs
    - Elasticsearch for search indexing

    Attributes:
        SQLHandler (SQLHandler): Handler for PostgreSQL operations
        RDFHandler (RDFHandler): Handler for Virtuoso RDF store
        IndexHandler (IndexHandler): Handler for Elasticsearch
        GraphHandler (GraphHandler): Handler for graph operations
        kg_files_directory (str): Directory for knowledge graph files
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
        Initialize LoadProcessor with database handlers.

        Args:
            SQLHandler (SQLHandler): PostgreSQL handler
            RDFHandler (RDFHandler): RDF store handler
            IndexHandler (IndexHandler): Elasticsearch handler
            GraphHandler (GraphHandler): Graph operations handler
            kg_files_directory (str): Path to knowledge graph files
        """
        self.SQLHandler = SQLHandler
        self.SQLHandler.connect()
        self.RDFHandler = RDFHandler
        self.IndexHandler = IndexHandler
        self.GraphHandler = GraphHandler
        self.kg_files_directory = kg_files_directory

    def update_dbs_with_df(self, df: DataFrame):
        """
        Update all databases with new data from DataFrame.

        Args:
            df (DataFrame): Data to be loaded into databases
        """

        # The graph handler updates the SQL and RDF databases with the new data
        self.GraphHandler.set_df(df)
        self.GraphHandler.update_graph()

    def load_df(self, df: DataFrame, output_ttl_file_path: str = None):
        """
        Load DataFrame into databases and optionally save TTL file.

        Args:
            df (DataFrame): Data to be loaded
            output_ttl_file_path (str, optional): Path to save TTL output
        """
        self.update_dbs_with_df(df)

        if output_ttl_file_path is not None:
            print("OUTPUT TTL FILE PATH\n", output_ttl_file_path)
            current_graph = self.GraphHandler.get_current_graph()
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_ttl_file_path = os.path.join(
                output_ttl_file_path, f"{current_date}_mlentory_graph.ttl"
            )
            current_graph.serialize(output_ttl_file_path, format="turtle")

    def update_dbs_with_kg(self, kg: Graph, extraction_metadata: Graph):
        """
        Update all databases with new data from KG. The KG represents the metadata

        Args:
            kg (Graph): Knowledge graph to be loaded
            extraction_metadata (Graph): Extraction metadata of the KG
        """
        self.GraphHandler.set_kg(kg)
        self.GraphHandler.set_extraction_metadata(extraction_metadata)
        self.GraphHandler.update_graph()

    def print_DB_states(self):
        """Print current state of all databases for debugging."""
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
        """Clean all databases, removing existing data."""
        self.RDFHandler.reset_db()
        self.IndexHandler.clean_indices()
        self.SQLHandler.clean_all_tables()
