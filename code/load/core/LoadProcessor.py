import os
import shutil
import mysql.connector
from rdflib.graph import Graph
import docker
import subprocess
import logging
from typing import Callable, List, Dict,Set
from SPARQLWrapper import SPARQLWrapper, JSON, DIGEST

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
        

        # Execute Virtuoso command to load data
        # isql -S 1111 -U dba -P my_strong_password
        # curl -u dba:my_strong_password \
        #  -H "Content-Type: application/sparql-query" \
        #  -H "Accept: application/json" \
        #  -d 'SELECT * WHERE { ?s ?p ?o } LIMIT 10' \
        #  http://virtuoso:1111/sparql
        # 'EXEC=status()'
        # command = """isql -S 1111 -U {virtuoso_user} -P {virtuoso_password} 
        # 'EXEC=SPARQL SELECT DISTINCT ?g WHERE {{ GRAPH ?g {{?s ?p ?t}} }};'""".format(virtuoso_user=virtuoso_user, virtuoso_password=virtuoso_password)
        # isql -S 1111 -U dba -P my_strong_password exec="ld_dir('/opt/virtuoso-opensource/database/kg_files','2024-07-29_14-31-16_Transformed_HF_fair4ml_schema_KG.ttl','http://example.com/data2');DB.DBA.rdf_loader_run();"
        
        sql_command = f" exec=\"ld_dir('/opt/virtuoso-opensource/database/kg_files','{ttl_file_path.split('/')[-1]}', 'http://example.com/data_2');DB.DBA.rdf_loader_run();\""
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
        # g = Graph()
        # g.open(sparql_endpoint)
        # results = g.query(query)
        # print("Query results::::::::::::")
        # for row in results:
        #     print(row)
        sparql = SPARQLWrapper(sparql_endpoint)
        sparql.setHTTPAuth(DIGEST)
        sparql.setCredentials(user, password)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        for result in results["results"]["bindings"]:
            print(result)
        
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