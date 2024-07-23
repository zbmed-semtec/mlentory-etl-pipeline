import mysql.connector
from rdflib.graph import Graph
import docker
import subprocess
import logging
from typing import Callable, List, Dict,Set

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LoadProcessor:
    """
    This class is responsible for loading the data into the different databases.
    """
    
    def __init__(self, host, user, password, database, port):
        """
        Initializes a new LoadProcessor instance.
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection = None
    
    def connect_to_mysql(self):
        self.connection = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
    
    def load_graph_to_virtuoso(self,container_name,ttl_file_path,virtuoso_user, virtuoso_password):
        """
        Uploads a TTL file to a Virtuoso instance running in a Docker container.

        Args:
            ttl_file_path: Path to the TTL file on the host machine.
            container_name: Name of the Docker container.
            virtuoso_user: Virtuoso username.
            virtuoso_password: Virtuoso password.
        """

        client = docker.from_env()
        container = client.containers.get(container_name)

        # Copy TTL file to container
        container.exec_run(f"mkdir -p /tmp/data")
        container.put_archive(ttl_file_path, "/tmp/data")

        # Execute Virtuoso command to load data
        command = f"isql -S localhost:1111 {virtuoso_user} {virtuoso_password} <<EOF\nLOAD_RDF_FILE('/tmp/data/{ttl_file_path.split('/')[-1]}', <graph_uri>);\nEOF"
        result = container.exec_run(command)

        # Check for errors
        if result.exit_code != 0:
            print(f"Error uploading TTL file: {result.output}")
    
    def query_virtuoso(self,sparql_endpoint,query):
        g = Graph()
        g.open(sparql_endpoint)
        results = g.query(query)
        for row in results:
            print(row)
    
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