import os
import shutil
import uuid
from datetime import datetime
from tqdm import tqdm
from rdflib.graph import Graph, ConjunctiveGraph
import logging

from rdflib import Graph, URIRef, Literal, BNode, Namespace
from rdflib.namespace import RDF, XSD, FOAF, RDFS, SKOS

from datetime import datetime
from typing import Callable, List, Dict, Set, Tuple
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
        
        self.META_NS = self.GraphHandler.graph_identifier + "/meta/"
        self.STATEMENT_METADATA = URIRef(str(self.META_NS) + "StatementMetadata")

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
                output_ttl_file_path, f"{current_date}_mlentory_graph.nt"
            )
            current_graph.serialize(output_ttl_file_path, format="nt")

    def update_dbs_with_kg(self, kg: Graph, extraction_metadata: Graph, remote_db: bool = False, kg_chunks_size: int = 100, save_chunks: bool = False, chunks_output_dir: str = ""):
        """
        Update all databases with new data from KG. The KG represents the metadata

        Args:
            kg (Graph): Knowledge graph to be loaded
            extraction_metadata (Graph): Extraction metadata of the KG
            remote_db (bool): Whether to use remote databases
            kg_chunks_size (int): Number of models to process at a time
            save_chunks (bool, optional): Whether to save the chunks to disk. Defaults to False.
        """
        
        kg_chunks, extraction_metadata_chunks = self.create_chunks(kg, extraction_metadata, kg_chunks_size, save_chunks = save_chunks, save_path=chunks_output_dir)
        
        for kg_chunk, extraction_metadata_chunk in zip(kg_chunks, extraction_metadata_chunks):
            if remote_db:
                self.send_kg_to_remote_db(kg_chunk, extraction_metadata_chunk)
            else:
                self.GraphHandler.set_kg(kg_chunk)
                self.GraphHandler.set_extraction_metadata(extraction_metadata_chunk)
                self.GraphHandler.update_graph()


    def create_chunks(self, kg: Graph, extraction_metadata: Graph, kg_chunks_size: int, save_chunks: bool = False, save_path: str = None) -> Tuple[List[Graph], List[Graph]]:
        """
        Create chunks of the KG and extraction metadata.
        We need to make sure that the triplets in one kg chunk are all from the same entity, as well making sure that the extraction metadata triplets match the kg triplets.

        Args:
            kg (Graph): The knowledge graph to be chunked
            extraction_metadata (Graph): The extraction metadata graph to be chunked
            kg_chunks_size (int): The size of each chunk
            save_chunks (bool, optional): Whether to save the chunks to disk. Defaults to False.
            save_path (str, optional): Base path where to save the chunks. Required if save_chunks is True.

        Returns:
            tuple: A tuple containing two lists:
                - List of knowledge graph chunks
                - List of corresponding extraction metadata chunks

        Raises:
            ValueError: If save_chunks is True but save_path is None
        """
        
        triplets_metadata = {}
        entity_triplets = {}
        entity_metadata_subjects = {}
        
        
        for triplet in tqdm(
            kg, desc="Creating extraction triples dictionaries for chunking"
        ):
            if triplet[0] not in entity_triplets:
                entity_triplets[triplet[0]] = [triplet]
            else:
                entity_triplets[triplet[0]].append(triplet)
        
        for triplet in tqdm(
            extraction_metadata, desc="Creating extraction triples dictionaries for chunking"
        ):
            if triplet[0] not in triplets_metadata:
                triplets_metadata[triplet[0]] = [triplet]
            else:
                triplets_metadata[triplet[0]].append(triplet)
            
            if URIRef(self.META_NS + "subject") == triplet[1]:
                if triplet[2] not in entity_metadata_subjects:
                    entity_metadata_subjects[triplet[2]] = [triplet[0]]
                else:
                    entity_metadata_subjects[triplet[2]].append(triplet[0])
        
        
        kg_chunks = []
        extraction_metadata_chunks = []
        
        # Replicate the graph namespaces of kg and extraction_metadata graphs
        temp_kg_chunk = Graph()
        temp_metadata_chunk = Graph()
        
        # Copy namespaces from original graphs
        for prefix, namespace in kg.namespaces():
            temp_kg_chunk.bind(prefix, namespace)
            temp_metadata_chunk.bind(prefix, namespace)
        
        batch_triplet_cnt = 0
        
        for subject in entity_triplets:
            
            for triplet_of_entity in entity_triplets[subject]:
                batch_triplet_cnt+=1
                temp_kg_chunk.add(triplet_of_entity)
            
            for triplets_metadata_subject in entity_metadata_subjects[subject]:
                for metadata_triplet_of_entity in triplets_metadata[triplets_metadata_subject]:
                    temp_metadata_chunk.add(metadata_triplet_of_entity)
                
            
            if batch_triplet_cnt*6 >= kg_chunks_size:
                batch_triplet_cnt = 0
                kg_chunks.append(temp_kg_chunk)
                
                extraction_metadata_chunks.append(temp_metadata_chunk)
                # Create new graph chunks with copied namespaces
                temp_kg_chunk = Graph()
                temp_metadata_chunk = Graph()
                
                # Copy namespaces to new chunks
                for prefix, namespace in kg.namespaces():
                    temp_kg_chunk.bind(prefix, namespace)
                for prefix, namespace in extraction_metadata.namespaces():
                    temp_metadata_chunk.bind(prefix, namespace)  
        
        if batch_triplet_cnt>0:
            kg_chunks.append(temp_kg_chunk)
            extraction_metadata_chunks.append(temp_metadata_chunk)
        
        if save_chunks:
            if save_path is "":
                raise ValueError("save_path must be provided when save_chunks is True")
            
            # Create a unique directory for this chunk set
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
            chunk_dir = os.path.join(save_path, f"{timestamp}_{unique_id}")
            
            try:
                # Create the directory if it doesn't exist
                os.makedirs(chunk_dir, exist_ok=True)
                logger.info(f"Created chunk directory: {chunk_dir}")
                
                # Save each chunk pair with numbered filenames
                for i, (kg_chunk, metadata_chunk) in enumerate(zip(kg_chunks, extraction_metadata_chunks), 1):
                    # Save KG chunk
                    kg_filename = os.path.join(chunk_dir, f"kg_chunk_{i:03d}.ttl")
                    kg_chunk.serialize(kg_filename, format="turtle")
                    
                    # Save metadata chunk
                    metadata_filename = os.path.join(chunk_dir, f"metadata_chunk_{i:03d}.ttl")
                    metadata_chunk.serialize(metadata_filename, format="turtle")
                
                logger.info(f"Successfully saved {len(kg_chunks)} chunk pairs to {chunk_dir}")
                
            except Exception as e:
                logger.error(f"Failed to save chunks to {chunk_dir}: {str(e)}")
                raise
        
        return kg_chunks, extraction_metadata_chunks
                
    
    def send_kg_to_remote_db(self, kg_chunk: Graph, extraction_metadata_chunk: Graph):
        """
        Send the KG and extraction metadata to the remote database.
        """
        print("SENDING KG TO REMOTE DB\n")
        for s, p, o in kg_chunk:
            print(s, p, o)
            
        print("SENDING EXTRACTION METADATA TO REMOTE DB\n")
        for s, p, o in extraction_metadata_chunk:
            print(s, p, o)
    
    def print_DB_states(self):
        """Print current state of all databases for debugging."""
        triplets_df = self.GraphHandler.SQLHandler.query(
            'SELECT COUNT(*) FROM "Triplet"'
        )
        ranges_df = self.GraphHandler.SQLHandler.query(
            'SELECT COUNT(*) FROM "Version_Range"'
        )
        triplets_extraction_info_df = self.GraphHandler.SQLHandler.query(
            'SELECT COUNT(*) FROM "Triplet_Extraction_Info"'
        )
        print("SQL TRIPlETS\n", triplets_df)
        print("SQL RANGES\n", ranges_df)
        print("SQL EXTRACTION INFO\n", triplets_extraction_info_df)

        result_graph = self.GraphHandler.get_current_graph()

        # print("VIRTUOSO TRIPlETS\n")
        # for i, (s, p, o) in enumerate(result_graph):
        #     print(f"{i}: {s} {p} {o}")

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
        # print("Check Elasticsearch: ", result, "\n")

    def clean_DBs(self):
        """Clean all databases, removing existing data."""
        self.RDFHandler.reset_db()
        self.IndexHandler.clean_indices()
        self.SQLHandler.clean_all_tables()
