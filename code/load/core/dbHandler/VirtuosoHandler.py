import docker
import shutil
from SPARQLWrapper import SPARQLWrapper, DIGEST, TURTLE
from rdflib import Graph

class VirtuosoHandler:
    def __init__(self, container_name, virtuoso_user, virtuoso_password, kg_files_directory, sparql_endpoint):
        self.container_name = container_name
        self.virtuoso_user = virtuoso_user
        self.virtuoso_password = virtuoso_password
        self.kg_files_directory = kg_files_directory
        self.sparql_endpoint = sparql_endpoint
        self.client = docker.from_env()

    def load_graph(self, ttl_file_path):
        """
        Uploads a TTL file to a Virtuoso instance running in a Docker container.

        Args:
            ttl_file_path: Path to the TTL file on the host machine.
            kg_files_directory: Directory where the TTL file will be located.
            container_name: Name of the Docker container.
            virtuoso_user: Virtuoso username.
            virtuoso_password: Virtuoso password.
        """
        container = self.client.containers.get(self.container_name)
        new_ttl_file_path = f"{self.kg_files_directory}/{ttl_file_path.split('/')[-1]}"

        shutil.move(ttl_file_path, new_ttl_file_path)
        
        sql_command = f""" exec=\"DELETE FROM DB.DBA.LOAD_LIST; 
                                ld_dir('/opt/virtuoso-opensource/database/kg_files',
                                '{ttl_file_path.split('/')[-1]}',
                                'http://example.com/data_1');DB.DBA.rdf_loader_run();\""""
                                
        command = f"""isql -S 1111 -U {self.virtuoso_user} -P {self.virtuoso_password} {sql_command}"""
        
        print("\nCOMMANDDDDDDDD: ", command)
        result = container.exec_run(command)
        print("\nRESULTTTTTTTTT: ", result.output)

    def query(self, sparql_endpoint, query):
        sparql = SPARQLWrapper(sparql_endpoint)
        sparql.setHTTPAuth(DIGEST)
        sparql.setCredentials(self.virtuoso_user, self.virtuoso_password)
        sparql.setQuery(query)
        sparql.setReturnFormat(TURTLE)
        g = sparql.query()._convertRDF()
        
        self.print_sample_triples(g)
        return g
    
    def print_sample_triples(self, graph, num_triples=10):
        print(f"Printing {num_triples} sample triples:")
        for i, (s, p, o) in enumerate(graph):
            if i >= num_triples:
                break
            print(f"{s} {p} {o}")