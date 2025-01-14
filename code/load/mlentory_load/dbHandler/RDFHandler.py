import docker
import shutil
from SPARQLWrapper import SPARQLWrapper, DIGEST, POST, GET, TURTLE, CSV, JSON, XML
from rdflib import Graph


class RDFHandler:
    """
    Handler for RDF triple store operations using Virtuoso.

    This class provides functionality to:
    - Manage RDF graphs in Virtuoso
    - Execute SPARQL queries
    - Handle Docker container operations
    - Process TTL files

    Attributes:
        container_name (str): Docker container name
        _user (str): Virtuoso username
        _password (str): Virtuoso password
        kg_files_directory (str): Directory for knowledge graph files
        sparql_endpoint (str): SPARQL endpoint URL
        client: Docker client instance
    """

    def __init__(
        self,
        container_name: str,
        _user: str,
        _password: str,
        kg_files_directory: str,
        sparql_endpoint: str,
    ):
        """
        Initialize RDFHandler with connection parameters.

        Args:
            container_name (str): Docker container name
            _user (str): Virtuoso username
            _password (str): Virtuoso password
            kg_files_directory (str): Directory for knowledge graph files
            sparql_endpoint (str): SPARQL endpoint URL
        """
        self.container_name = container_name
        self._user = _user
        self._password = _password
        self.kg_files_directory = kg_files_directory
        self.sparql_endpoint = sparql_endpoint
        self.client = docker.from_env()

    def reset_db(self):
        """Reset the RDF database to its initial state."""
        container = self.client.containers.get(self.container_name)

        sql_command = f""" exec=\"RDF_GLOBAL_RESET ();\""""
        command = f"""isql -S 1111 -U {self._user} -P {self._password} {sql_command}"""
        result = container.exec_run(command)
        print(result)

    def load_graph(self, ttl_file_path: str, graph_identifier: str):
        """
        Load a TTL file into the RDF store.

        Args:
            ttl_file_path (str): Path to TTL file
            graph_identifier (str): URI for the target graph
        """
        container = self.client.containers.get(self.container_name)
        new_ttl_file_path = f"{self.kg_files_directory}/{ttl_file_path.split('/')[-1]}"

        shutil.move(ttl_file_path, new_ttl_file_path)

        sql_command = f""" exec=\"DELETE FROM DB.DBA.LOAD_LIST; 
                                ld_dir('/opt/virtuoso-opensource/database/kg_files',
                                '{ttl_file_path.split('/')[-1]}',
                                '{graph_identifier}');
                                DB.DBA.rdf_loader_run();
                                checkpoint;\""""

        command = f"""isql -S 1111 -U {self._user} -P {self._password} {sql_command}"""
        result = container.exec_run(command)

    def upload_rdf_file(self, rdf_file_path, container_rdf_folder, graph_iri):
        """
        Uploads a RDF file containing a graph to a RDF database instance running in a Docker container.

        Args:
            rdf_file_path: Path to the RDF file on the host machine.
            kg_files_directory: Directory where the RDF file will be located.
            container_name: Name of the Docker container where the RDF database is running.
            container_rdf_folder: Directory where the RDF file will be located in the container.
            graph_iri: Identifier  of the graph to be loaded.
            _user:  username.
            _password:  password.
        """
        container = self.client.containers.get(self.container_name)
        new_rdf_file_path = f"{self.kg_files_directory}/{rdf_file_path.split('/')[-1]}"

        shutil.move(rdf_file_path, new_rdf_file_path)

        sql_command = f""" exec=\"DELETE FROM DB.DBA.LOAD_LIST; 
                                ld_dir('{container_rdf_folder}',
                                '{rdf_file_path.split('/')[-1]}',
                                '{graph_iri}');
                                DB.DBA.rdf_loader_run();
                                checkpoint;\""""

        command = f"""isql -S 1111 -U {self._user} -P {self._password} {sql_command}"""

        result = container.exec_run(command)

    def delete_graph(
        self,
        ttl_file_path: str,
        graph_identifier: str,
        deprecated_graph_identifier: str,
    ):
        """
        Delete triples from a graph based on TTL file.

        Args:
            ttl_file_path (str): Path to TTL file
            graph_identifier (str): URI of main graph
            deprecated_graph_identifier (str): URI of deprecated graph
        """
        container = self.client.containers.get(self.container_name)
        new_ttl_file_path = f"{self.kg_files_directory}/{ttl_file_path.split('/')[-1]}"

        shutil.move(ttl_file_path, new_ttl_file_path)
        sql_command = f""" exec=\"
                                ld_dir('/opt/virtuoso-opensource/database/kg_files',
                                '{ttl_file_path.split('/')[-1]}',
                                '{deprecated_graph_identifier}');
                                DB.DBA.rdf_loader_run();
                                log_enable(3,1);
                                Delete from rdf_quad a 
                                where exists (select * from rdf_quad b
                                where a.s = b.s and a.p = b.p and a.o = b.o 
                                and b.g = iri_to_id('{deprecated_graph_identifier}')
                                and a.g = iri_to_id('{graph_identifier}'));\"
                        """

        command = f"""isql -S 1111 -U {self._user} -P {self._password} {sql_command}"""
        result = container.exec_run(command)

    def delete_triple(
        self,
        sparql_endpoint: str,
        subject: str,
        predicate: str,
        object: str,
        graph_iri: str,
    ):
        """
        Delete a specific triple from the graph.

        Args:
            sparql_endpoint (str): SPARQL endpoint URL
            subject (str): Triple subject
            predicate (str): Triple predicate
            object (str): Triple object
            graph_iri (str): Graph URI
        """
        sparql = SPARQLWrapper(sparql_endpoint)
        sparql.setHTTPAuth(DIGEST)
        sparql.setCredentials(self._user, self._password)
        sparql.setMethod(POST)
        # query = "DELETE { ?s ?p ?o } WHERE {GRAPH <http://example.com/data_1> {?s ?p ?o}}".format(subject=subject, predicate=predicate, object=object, graph_iri=graph_iri)
        # query = f"DELETE {{ ?s ?p ?o }} WHERE {{GRAPH <{graph_iri}> {{{subject} {predicate} {object}}}}}"
        query = f"WITH <{graph_iri}> DELETE {{ {subject._value} {predicate._value} {object._value} }}"
        sparql.setQuery(query)
        # sparql.setReturnFormat(TURTLE)

        g = sparql.query()

        for row in g:
            print(row)

        return g

    def query(self, sparql_endpoint: str, query: str):
        """
        Execute a SPARQL query.

        Args:
            sparql_endpoint (str): SPARQL endpoint URL
            query (str): SPARQL query string

        Returns:
            Graph: Query results as RDF graph
        """
        sparql = SPARQLWrapper(sparql_endpoint)
        sparql.setHTTPAuth(DIGEST)
        sparql.setCredentials(self._user, self._password)
        sparql.setQuery(query)
        # sparql.setReturnFormat(TURTLE)

        g = sparql.query()._convertRDF()

        # self.print_sample_triples(g)
        return g

    def print_sample_triples(self, graph, num_triples=10):
        print(f"Printing {num_triples} sample triples:")
        for i, (s, p, o) in enumerate(graph):
            if i >= num_triples:
                break
            print(f"{s} {p} {o}")
