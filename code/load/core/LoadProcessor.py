import os
import shutil
import mysql.connector
from rdflib.graph import Graph,ConjunctiveGraph
import docker
import subprocess
import logging
from typing import Callable, List, Dict,Set
from SPARQLWrapper import SPARQLWrapper, JSON, DIGEST,TURTLE

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
    
    def load_graph_to_virtuoso(self,container_name,ttl_file_path,kg_files_directory,virtuoso_user, virtuoso_password):
        """
        Uploads a TTL file to a Virtuoso instance running in a Docker container.

        Args:
            ttl_file_path: Path to the TTL file on the host machine.
            kg_files_directory: Directory where the TTL file will be located.
            container_name: Name of the Docker container.
            virtuoso_user: Virtuoso username.
            virtuoso_password: Virtuoso password.
        """

        client = docker.from_env()
        container = client.containers.get(container_name)
        new_ttl_file_path = f"{kg_files_directory}/{ttl_file_path.split('/')[-1]}"

        shutil.move(ttl_file_path, new_ttl_file_path)
        
        sql_command = f""" exec=\"DELETE FROM DB.DBA.LOAD_LIST; 
                                ld_dir('/opt/virtuoso-opensource/database/kg_files',
                                '{ttl_file_path.split('/')[-1]}',
                                'http://example.com/data_1');DB.DBA.rdf_loader_run();\""""
                                
        command = """isql -S 1111 -U {virtuoso_user} -P {virtuoso_password} {sql_command}""".format(
            virtuoso_user=virtuoso_user, 
            virtuoso_password=virtuoso_password,
            sql_command=sql_command)
        
        print("\nCOMMANDDDDDDDD: ", command)
        result = container.exec_run(command)
        print("\nRESULTTTTTTTTT: ", result.output)

        # Check for errors
        # if result.exit_code != 0:
        
    
    def query_virtuoso(self,sparql_endpoint,query,user,password):
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