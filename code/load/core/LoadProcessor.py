import os
import shutil
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
    from code.load.core.dbHandler.IndexHandler import IndexHandler
    from code.load.core.GraphHandler import GraphHandler
    from code.load.core.Entities import HFModel
else:
    from core.dbHandler.SQLHandler import SQLHandler
    from core.dbHandler.RDFHandler import RDFHandler
    from core.dbHandler.IndexHandler import IndexHandler
    from core.GraphHandler import GraphHandler
    from core.Entities import HFModel

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
