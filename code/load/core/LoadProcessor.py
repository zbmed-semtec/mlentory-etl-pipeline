import os
import shutil
import mysql.connector
from rdflib.graph import Graph, ConjunctiveGraph
import docker
import subprocess
import logging
from datetime import datetime
from typing import Callable, List, Dict, Set
from SPARQLWrapper import SPARQLWrapper, JSON, DIGEST, TURTLE

if "app_test" in os.getcwd():
    from code.load.core.dbHandler.SQLHandler import SQLHandler
    from code.load.core.dbHandler.RDFHandler import RDFHandler
    from code.load.core.GraphHandler import GraphHandler
else:
    from code.load.core.dbHandler.SQLHandler import SQLHandler
    from code.load.core.dbHandler.RDFHandler import RDFHandler
    from code.load.core.GraphHandler import GraphHandler

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
        GraphHandler: GraphHandler,
        kg_files_directory: str,
    ):
        """
        Initializes a new LoadProcessor instance.
        """
        self.SQLHandler = SQLHandler
        self.SQLHandler.connect()
        self.RDFHandler = RDFHandler
        self.GraphHandler = GraphHandler
        self.kg_files_directory = kg_files_directory

    def load_dataframe(self, df):
        
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        filename_ttl = f"{self.kg_files_directory}/{now}_Transformed_HF_fair4ml_schema_KG.ttl"  # Create new filename

        self.GraphHandler.load_df(df)
        self.GraphHandler.create_graph()
        self.GraphHandler.store_graph(filename_ttl)

        # extracted_graph = Graph()
        # extracted_graph.parse(ttl_file_path,format="turtle")

    def load_graph(self, df, kg_files_directory):

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        filename_ttl = f"{self.kg_files_directory}/{now}_Transformed_HF_fair4ml_schema_KG.ttl"  # Create new filename

        self.graph_creator.load_df(df)
        self.graph_creator.create_graph()
        self.graph_creator.store_graph(filename_ttl)

        extracted_graph = Graph()
        # extracted_graph.parse(ttl_file_path,format="turtle")

        # for subject, predicate, object in graph_to_load:
        #     print(f"subject: {subject}, predicate: {predicate}, object: {object}")

    def get_difference_graph(self, extracted_graph):
        # There are going to be new triplets never seen before
        # and triplets seen before that where not valid and now they are valid
        # I'm pondering on how to extract information on the version
        for subject, predicate, object in extracted_graph:
            print(f"subject: {subject}, predicate: {predicate}, object: {object}")
            result = self.SQLHandler.query(
                f"SELECT * FROM Triplet WHERE subject = '{subject}' AND relation = '{predicate}' AND object = '{object}'"
            )
            if result.empty:
                print(f"subject: {subject}, relation: {predicate}, object: {object}")
                print("Triplet not found in the database")
                # We need to delete the triplet from the graph
                extracted_graph.remove((subject, predicate, object))

    def query_virtuoso(self, sparql_endpoint, query, user, password):
        sparql = SPARQLWrapper(sparql_endpoint)
        sparql.setHTTPAuth(DIGEST)
        sparql.setCredentials(user, password)
        sparql.setQuery(query)
        # sparql.setReturnFormat(TURTLE)
        g = sparql.query()._convertRDF()

        self.print_sample_triples(g)
        return g

    def print_sample_triples(self, graph, num_triples=10):
        print(f"Printing {num_triples} sample triples:")
        for i, (s, p, o) in enumerate(graph):
            if i >= num_triples:
                break
            print(f"{s} {p} {o}")

    def load_graph_to_mysql(self, graph: Graph, table_name: str = "triples") -> None:
        if not self.connection:
            self.connect_to_mysql()

        cursor = self.connection.cursor()

        for subject, predicate, obj in graph:
            query = f"INSERT INTO {table_name} (subject, predicate, object) VALUES (%s, %s, %s)"
            values = (str(subject), str(predicate), str(obj))
            cursor.execute(query, values)

        self.connection.commit()
        cursor.close()

    # def create_graph_
