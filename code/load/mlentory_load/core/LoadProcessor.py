import os
import shutil
import uuid
import requests
import tempfile
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
        remote_api_base_url: str = None,
    ):
        """
        Initialize LoadProcessor with database handlers.

        Args:
            SQLHandler (SQLHandler): PostgreSQL handler
            RDFHandler (RDFHandler): RDF store handler
            IndexHandler (IndexHandler): Elasticsearch handler
            GraphHandler (GraphHandler): Graph operations handler
            kg_files_directory (str): Path to knowledge graph files
            remote_api_base_url (str, optional): Base URL for remote upload API
        """
        self.SQLHandler = SQLHandler
        self.RDFHandler = RDFHandler
        self.IndexHandler = IndexHandler
        self.GraphHandler = GraphHandler
        self.kg_files_directory = kg_files_directory
        self.remote_api_base_url = remote_api_base_url
        self.upload_timeout = 300  # Default 5 minutes
        
        if remote_api_base_url is None:
            self.SQLHandler.connect()
            # Initialize all indices to prevent conflicts between different data sources
            self.IndexHandler.initialize_HF_index(index_name="hf_models")
            self.IndexHandler.initialize_OpenML_index(index_name="openml_models")
            self.IndexHandler.initialize_AI4Life_index(index_name="ai4life_models")
        
        self.META_NS = self.GraphHandler.graph_identifier + "/meta/"
        self.STATEMENT_METADATA = URIRef(str(self.META_NS) + "StatementMetadata")

    def set_upload_timeout(self, timeout_seconds: int) -> None:
        """
        Set the timeout for HTTP uploads to remote database.
        
        Args:
            timeout_seconds (int): Timeout in seconds for HTTP uploads
        """
        self.upload_timeout = timeout_seconds
        logger.info(f"Upload timeout set to {timeout_seconds} seconds")

    def update_dbs_with_kg(self, kg: Graph, extraction_metadata: Graph,
                                            extraction_name: str = "hf_extraction",
                                            remote_db: bool = False, 
                                            kg_chunks_size: int = 100, 
                                            save_load_input: bool = False,
                                            load_input_dir: str = "",
                                            save_load_output: bool = False, 
                                            load_output_dir: str = "", 
                                            trigger_etl: bool = True):
        """
        Update all databases with new data from KG. The KG represents the metadata

        Args:
            kg (Graph): Knowledge graph to be loaded
            extraction_metadata (Graph): Extraction metadata of the KG
            extraction_name (str, optional): Name/type of extraction (e.g., "hf_extraction", "openml_extraction"). Defaults to "hf_extraction".
            remote_db (bool): Whether to use remote databases
            kg_chunks_size (int): Number of models to process at a time. If 0, treats entire KG as single chunk.
            save_load_output (bool, optional): Whether to save the chunks to disk. Defaults to False.
            load_output_dir (str, optional): Directory to store the chunks on disk. Defaults to "". 
                If this parameter is "", when save_load_output or remote_db are true an exception will be thrown.
            trigger_etl (bool, optional): Whether to automatically trigger ETL processing after remote upload. Defaults to True.
        """
        
        if (save_load_output or remote_db) and (load_output_dir == ""):
            raise ValueError("No output directory found")
        
        if save_load_input and load_input_dir == "":
            raise ValueError("No input directory found")
        
        # if save_load_input and load_input_dir != "":
        #     uploaself.RDFHandler
        
        if remote_db:
            # For remote uploads, we need to save chunks and get file paths
            kg_chunks, extraction_metadata_chunks, chunk_files = self.create_chunks_with_files(
                kg=kg, extraction_metadata=extraction_metadata, kg_chunks_size=kg_chunks_size, chunks_output_dir=load_output_dir
            )
            self.send_batch_to_remote_db(chunk_files, trigger_etl=trigger_etl, extraction_name=extraction_name)
        else:
            kg_chunks, extraction_metadata_chunks = self.create_chunks(
                kg, extraction_metadata, kg_chunks_size, 
                save_chunks=save_load_output, save_path=load_output_dir
            )
            
            for kg_chunk, extraction_metadata_chunk in zip(kg_chunks, extraction_metadata_chunks):
                self.GraphHandler.set_kg(kg_chunk)
                self.GraphHandler.set_extraction_metadata(extraction_metadata_chunk)
                self.GraphHandler.update_graph()

    def create_chunks(self, kg: Graph, extraction_metadata: Graph, kg_chunks_size: int, save_chunks: bool = False, save_path: str = None) -> Tuple[List[Graph], List[Graph]]:
        """
        Create chunks of the KG and extraction metadata.
        We need to make sure that the triplets in one kg chunk are all from the same entity, as well making sure that the extraction metadata triplets match the kg triplets.
        If kg_chunks_size is 0 only 1 chunk, with all the original data, will be created.

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
         
        kg_chunks = []
        extraction_metadata_chunks = []
        
        if kg_chunks_size == 0:
            kg_chunks.append(kg)
            extraction_metadata_chunks.append(extraction_metadata)
        
        if kg_chunks_size>0:
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
                extraction_metadata, desc="Creating extraction metadata triples dictionaries for chunking"
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
                    
                
                if batch_triplet_cnt*7 >= kg_chunks_size:
                    batch_triplet_cnt = 0
                    kg_chunks.append(temp_kg_chunk)
                    
                    extraction_metadata_chunks.append(temp_metadata_chunk)
                    # Create new graph chunks with copied namespaces
                    temp_kg_chunk = Graph()
                    temp_metadata_chunk = Graph()
                    
                    # Copy namespaces to new chunks
                    for prefix, namespace in kg.namespaces():
                        temp_kg_chunk.bind(prefix, namespace)
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
                    kg_filename = os.path.join(chunk_dir, f"kg_chunk_{i:03d}.nt")
                    kg_chunk.serialize(kg_filename, format="nt")
                    
                    # Save metadata chunk
                    metadata_filename = os.path.join(chunk_dir, f"metadata_chunk_{i:03d}.nt")
                    metadata_chunk.serialize(metadata_filename, format="nt")
                
                logger.info(f"Successfully saved {len(kg_chunks)} chunk pairs to {chunk_dir}")
                
            except Exception as e:
                logger.error(f"Failed to save chunks to {chunk_dir}: {str(e)}")
                raise
        
        return kg_chunks, extraction_metadata_chunks
        
    def create_chunks_with_files(self, kg: Graph, extraction_metadata: Graph, kg_chunks_size: int, chunks_output_dir: str) -> Tuple[List[Graph], List[Graph], Dict[str, str]]:
        """
        Create chunks and save them to files for remote upload.
        
        Args:
            kg (Graph): The knowledge graph to be chunked
            extraction_metadata (Graph): The extraction metadata graph to be chunked
            kg_chunks_size (int): The size of each chunk
            chunks_output_dir (str): Directory where to save the chunks
            
        Returns:
            tuple: A tuple containing:
                - List of knowledge graph chunks
                - List of corresponding extraction metadata chunks  
                - Dictionary with chunk file information including batch_id and chunk_dir
        """
        if not chunks_output_dir:
            raise ValueError("chunks_output_dir must be provided for remote uploads")
            
        # Store the time before creating chunks to identify the new directory
        before_time = datetime.now()
        
        # Create chunks and save them
        kg_chunks, extraction_metadata_chunks = self.create_chunks(
            kg, extraction_metadata, kg_chunks_size, 
            save_chunks=True, save_path=chunks_output_dir
        )
        
        # Find the chunk directory that was just created (after before_time)
        try:
            dirs = [d for d in os.listdir(chunks_output_dir) 
                   if os.path.isdir(os.path.join(chunks_output_dir, d))]
            if dirs:
                # Get the most recently created directory
                newest_dir = max(dirs, key=lambda d: os.path.getctime(os.path.join(chunks_output_dir, d)))
                chunk_dir_path = os.path.join(chunks_output_dir, newest_dir)
                
                # Use the directory name as batch_id (it already contains timestamp and unique id)
                batch_id = newest_dir
            else:
                raise ValueError("No chunk directory found")
        except Exception as e:
            logger.error(f"Failed to find chunk directory: {e}")
            raise
            
        chunk_files = {
            "batch_id": batch_id,
            "chunk_dir": chunk_dir_path,
            "num_chunks": len(kg_chunks)
        }
        
        return kg_chunks, extraction_metadata_chunks, chunk_files

    def send_batch_to_remote_db(self, chunk_files: Dict[str, str], trigger_etl: bool = True, extraction_name: str = "hf_extraction"):
        """
        Send a batch of chunk files to the remote database.
        
        Args:
            chunk_files (Dict): Dictionary containing batch_id, chunk_dir, and num_chunks
            trigger_etl (bool): Whether to automatically trigger ETL processing after upload. Defaults to True.
            extraction_name (str): Name/type of extraction for identification. Defaults to "hf_extraction".
        """
        batch_id = chunk_files["batch_id"]
        upload_date = "".join(batch_id.split("_")[0:-2]) 
        chunk_dir = chunk_files["chunk_dir"]
        num_chunks = chunk_files["num_chunks"]
        
        try:
            logger.info(f"Sending batch {batch_id} for {extraction_name} with {num_chunks} chunks from {chunk_dir}")
            
            # Upload each chunk pair
            for i in range(1, num_chunks + 1):
                kg_filename = os.path.join(chunk_dir, f"kg_chunk_{i:03d}.nt")
                metadata_filename = os.path.join(chunk_dir, f"metadata_chunk_{i:03d}.nt")
                
                if not os.path.exists(kg_filename) or not os.path.exists(metadata_filename):
                    logger.error(f"Chunk files not found: {kg_filename}, {metadata_filename}")
                    continue
                    
                self.send_chunk_files_to_remote_db(
                    batch_id, f"{upload_date}_{extraction_name}", i-1, num_chunks,
                    kg_filename, metadata_filename
                )
            
            # Finalize the batch
            finalize_url = f"{self.remote_api_base_url.rstrip('/')}/upload/batch/{batch_id}/finalize"
            #Check this timeout for big uploads
            response = requests.post(finalize_url, timeout=120)
            response.raise_for_status()
            
            logger.info(f"Successfully finalized batch {batch_id}. Response: {response.json()}")
            
            # Trigger ETL processing if requested
            if trigger_etl:
                self.trigger_remote_etl_processing(batch_id, extraction_name)
            
        except Exception as e:
            logger.error(f"Failed to send batch to remote database: {e}")
            raise

    def trigger_remote_etl_processing(self, batch_id: str, extraction_name: str):
        """
        Trigger ETL processing for a batch on the remote server.
        
        Args:
            batch_id (str): The batch identifier to process through ETL
            extraction_name (str): Name/type of extraction for identification
            
        Raises:
            requests.RequestException: If HTTP request fails
            Exception: If ETL processing trigger fails
        """
        try:
            etl_url = f"{self.remote_api_base_url.rstrip('/')}/upload/process-etl/{batch_id}"
            logger.info(f"Triggering ETL processing for {extraction_name} batch {batch_id} at: {etl_url}")
            
            # Trigger ETL processing with extended timeout for large datasets
            response = requests.post(etl_url, timeout=600)  # 10 minute timeout for ETL
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Successfully triggered ETL processing for {extraction_name} batch {batch_id}. Response: {result}")
            
        except requests.RequestException as e:
            logger.error(f"Failed to trigger ETL processing for {extraction_name} batch {batch_id}: {e}")
            # Don't re-raise here - ETL can be triggered manually later
            logger.warning(f"ETL processing for {extraction_name} batch {batch_id} can be triggered manually via: {etl_url}")
        except Exception as e:
            logger.error(f"Unexpected error triggering ETL for {extraction_name} batch {batch_id}: {e}")
            logger.warning(f"ETL processing for {extraction_name} batch {batch_id} can be triggered manually via the API")

    def trigger_auto_etl_processing(self):
        """
        Trigger automatic ETL processing for all complete batches on the remote server.
        
        Returns:
            dict: Response from the auto-processing endpoint
            
        Raises:
            requests.RequestException: If HTTP request fails
            Exception: If auto-processing trigger fails
        """
        try:
            auto_etl_url = f"{self.remote_api_base_url.rstrip('/')}/upload/auto-process-etl"
            logger.info(f"Triggering auto ETL processing at: {auto_etl_url}")
            
            # Trigger auto ETL processing with extended timeout
            response = requests.post(auto_etl_url, timeout=900)  # 15 minute timeout for auto-processing
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Successfully triggered auto ETL processing. Response: {result}")
            return result
            
        except requests.RequestException as e:
            logger.error(f"Failed to trigger auto ETL processing: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error triggering auto ETL processing: {e}")
            raise

    def get_batch_status(self, batch_id: str):
        """
        Get the status of a batch upload on the remote server.
        
        Args:
            batch_id (str): The batch identifier to check
            
        Returns:
            dict: Batch status information
            
        Raises:
            requests.RequestException: If HTTP request fails
            Exception: If status check fails
        """
        try:
            status_url = f"{self.remote_api_base_url.rstrip('/')}/upload/batch/{batch_id}/status"
            logger.info(f"Checking batch status for {batch_id} at: {status_url}")
            
            response = requests.get(status_url, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Batch {batch_id} status: {result}")
            return result
            
        except requests.RequestException as e:
            logger.error(f"Failed to get batch status for {batch_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting batch status for {batch_id}: {e}")
            raise

    def send_chunk_files_to_remote_db(self, batch_id: str, file_name: str, chunk_number: int, total_chunks: int, kg_file_path: str, metadata_file_path: str):
        """
        Send individual chunk files to the remote database.
        
        Args:
            batch_id (str): Batch identifier
            file_name (str): Base file name for the chunk
            chunk_number (int): Chunk sequence number (0-based)
            total_chunks (int): Total number of chunks in batch
            kg_file_path (str): Path to KG chunk file
            metadata_file_path (str): Path to metadata chunk file
        """
        try:
            # Prepare file uploads
            with open(kg_file_path, 'rb') as kg_f, open(metadata_file_path, 'rb') as meta_f:
                files = {
                    'chunk_kg_data': (os.path.basename(kg_file_path), kg_f, 'text/nt'),
                    'chunk_extraction_metadata': (os.path.basename(metadata_file_path), meta_f, 'text/nt')
                }
                
                data = {
                    'batch_id': batch_id,
                    'file_name': file_name,
                    'chunk_number': chunk_number,
                    'total_chunks': total_chunks
                }
                
                # Send HTTP POST request to upload endpoint
                upload_url = f"{self.remote_api_base_url.rstrip('/')}/upload/chunk"
                logger.info(f"Uploading chunk {chunk_number + 1}/{total_chunks} to: {upload_url}")
                
                response = requests.post(
                    upload_url,
                    files=files,
                    data=data,
                    timeout=self.upload_timeout
                )
                
                response.raise_for_status()
                logger.info(f"Successfully uploaded chunk {chunk_number + 1}/{total_chunks}")
                
        except Exception as e:
            logger.error(f"Failed to upload chunk {chunk_number + 1}: {e}")
            raise
                
    
    
    
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

        print("RDF TRIPlETS\n")
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
        # print("Check Elasticsearch: ", result, "\n")

    def clean_DBs(self):
        """Clean all databases, removing existing data."""
        self.RDFHandler.reset_db()
        self.IndexHandler.clean_indices()
        self.SQLHandler.clean_all_tables()
