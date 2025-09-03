import os
import shutil
import time
import logging
import argparse
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from rdflib import Graph
import pandas as pd

from mlentory_load.core import LoadProcessor, GraphHandlerForKG
from mlentory_load.dbHandler import RDFHandler, SQLHandler, IndexHandler

class BaseETLComponent(ABC):
    """
    Base class for ETL (Extract, Transform, Load) operations.
    Provides common functionality and standardized interface for different data sources.
    
    This class follows the Template Method pattern, defining the skeleton of the ETL
    process while allowing subclasses to override specific steps.
    """
    
    def __init__(self, 
                 source_name: str,
                 config_path: str,
                 output_dir: str = None,
                 kg_files_directory: str = "./../kg_files"):
        """
        Initialize the ETL component.

        Args:
            source_name (str): Name of the data source (e.g., 'huggingface', 'openml')
            config_path (str): Path to configuration directory
            output_dir (str): Directory for output files
            kg_files_directory (str): Directory for knowledge graph files
        """
        self.source_name = source_name
        self.config_path = config_path
        self.output_dir = output_dir or f"./{source_name}_etl/outputs/files"
        self.kg_files_directory = kg_files_directory
        self.logger = self._setup_logging()
        
        # Load environment variables with defaults
        self.db_config = {
            "postgres": {
                "host": os.getenv("POSTGRES_HOST", "postgres"),
                "user": os.getenv("POSTGRES_USER", "user"),
                "password": os.getenv("POSTGRES_PASSWORD", "password"),
                "database": os.getenv("POSTGRES_DB", "history_DB"),
            },
            "virtuoso": {
                "host": os.getenv("VIRTUOSO_HOST", "virtuoso_db"),
                "user": os.getenv("VIRTUOSO_USER", "dba"),
                "password": os.getenv("VIRTUOSO_PASSWORD", "my_strong_password"),
                "sparql_endpoint": os.getenv("VIRTUOSO_SPARQL_ENDPOINT", None),  # Will be set in init
            },
            "elasticsearch": {
                "host": os.getenv("ELASTICSEARCH_HOST", "elastic_db"),
                "port": int(os.getenv("ELASTICSEARCH_PORT", "9200")),
            },
            "remote_api": {
                "base_url": os.getenv("REMOTE_API_BASE_URL", "http://localhost:8000"),
            }
        }
        
        # Set Virtuoso SPARQL endpoint if not provided
        if not self.db_config["virtuoso"]["sparql_endpoint"]:
            self.db_config["virtuoso"]["sparql_endpoint"] = f"http://{self.db_config['virtuoso']['host']}:8890/sparql"

    def _setup_logging(self) -> logging.Logger:
        """Sets up logging configuration."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_log_path = f"./{self.source_name}_etl/outputs/execution_logs"
        os.makedirs(base_log_path, exist_ok=True)
        log_file = f"{base_log_path}/etl_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            handlers=[
                logging.FileHandler(log_file, mode="w"),
                logging.StreamHandler(),
            ],
        )
        logger = logging.getLogger(self.source_name)
        logger.setLevel(logging.INFO)
        return logger

    def initialize_folder_structure(self, clean_folders: bool = False) -> None:
        """
        Initializes the folder structure for ETL outputs.

        Args:
            clean_folders (bool): If True, removes existing folders before creation
        """
        if clean_folders:
            shutil.rmtree(self.output_dir, ignore_errors=True)
            
        folders = [
            self.output_dir,
            f"{self.output_dir}/models",
            f"{self.output_dir}/datasets",
            f"{self.output_dir}/articles",
            f"{self.output_dir}/kg",
            f"{self.output_dir}/extraction_metadata",
            f"{self.output_dir}/chunks"
        ]
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def initialize_load_processor(self) -> LoadProcessor:
        """Initializes the load processor with database connections."""
        # Initialize SQL Handler
        sql_handler = SQLHandler(
            host=self.db_config["postgres"]["host"],
            user=self.db_config["postgres"]["user"],
            password=self.db_config["postgres"]["password"],
            database=self.db_config["postgres"]["database"],
        )
        sql_handler.connect()

        # Initialize RDF Handler
        rdf_handler = RDFHandler(
            container_name=self.db_config["virtuoso"]["host"],
            kg_files_directory=self.kg_files_directory,
            _user=self.db_config["virtuoso"]["user"],
            _password=self.db_config["virtuoso"]["password"],
            sparql_endpoint=self.db_config["virtuoso"]["sparql_endpoint"],
        )

        # Initialize Elasticsearch Handler
        es_handler = IndexHandler(
            es_host=self.db_config["elasticsearch"]["host"],
            es_port=self.db_config["elasticsearch"]["port"],
        )
        
        # Initialize indices
        es_handler.initialize_HF_index(index_name="hf_models")
        es_handler.initialize_OpenML_index(index_name="openml_models")
        es_handler.initialize_AI4Life_index(index_name="ai4life_models")

        # Initialize Graph Handler
        graph_handler = GraphHandlerForKG(
            SQLHandler=sql_handler,
            RDFHandler=rdf_handler,
            IndexHandler=es_handler,
            kg_files_directory=self.kg_files_directory,
            graph_identifier="https://w3id.org/mlentory/mlentory_graph",
            deprecated_graph_identifier="https://w3id.org/mlentory/deprecated_mlentory_graph",
            logger=self.logger,
        )

        # Initialize and return Load Processor
        return LoadProcessor(
            SQLHandler=sql_handler,
            RDFHandler=rdf_handler,
            IndexHandler=es_handler,
            GraphHandler=graph_handler,
            kg_files_directory=self.kg_files_directory,
            remote_api_base_url=self.db_config["remote_api"]["base_url"],
        )

    @abstractmethod
    def get_argument_parser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser with common ETL arguments.
        Subclasses should extend this with source-specific arguments.
        """
        parser = argparse.ArgumentParser(description=f"{self.source_name.upper()} ETL Process")
        
        # Common arguments across all ETL processes
        parser.add_argument(
            "--save-extraction",
            action="store_true",
            default=False,
            help="Save the results of the extraction phase",
        )
        parser.add_argument(
            "--save-transformation",
            action="store_true",
            default=False,
            help="Save the results of the transformation phase",
        )
        parser.add_argument(
            "--output-dir",
            default=self.output_dir,
            help="Directory to save intermediate results",
        )
        parser.add_argument(
            "--use-dummy-data",
            "-ud",
            default=False,
            help="Use dummy data for testing purposes.",
        )
        parser.add_argument(
            "--remote-db",
            "-rd",
            default=False,
            help="Use remote databases for loading data.",
        )
        parser.add_argument(
            "--chunking",
            default=True,
            help="Whether or not to chunk the data for the uploading step"
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=1000,
            help="Size of chunks when uploading to remote database"
        )
        parser.add_argument(
            "--upload-timeout",
            type=int,
            default=800,
            help="Timeout in seconds for HTTP uploads to remote database"
        )
        parser.add_argument(
            "--kg-file-path",
            type=str,
            help="Path to existing KG file to load directly"
        )
        parser.add_argument(
            "--metadata-file-path", 
            type=str,
            help="Path to existing metadata file to load directly"
        )
        
        return parser

    def load_existing_files(self, kg_path: str, metadata_path: str) -> Tuple[Graph, Graph]:
        """
        Loads existing KG and metadata files.

        Args:
            kg_path (str): Path to the knowledge graph file
            metadata_path (str): Path to the metadata file

        Returns:
            Tuple[Graph, Graph]: Loaded KG and metadata graphs
        """
        if not os.path.exists(kg_path):
            raise FileNotFoundError(f"KG file not found: {kg_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        start_time = time.time()
        kg_integrated = Graph()
        extraction_metadata_integrated = Graph()
        
        # Determine format based on file extension
        kg_format = "turtle" if kg_path.endswith(('.ttl', '.turtle')) else "nt"
        metadata_format = "turtle" if metadata_path.endswith(('.ttl', '.turtle')) else "nt"
        
        kg_integrated.parse(kg_path, format=kg_format)
        extraction_metadata_integrated.parse(metadata_path, format=metadata_format)
        
        end_time = time.time()
        self.logger.info(f"Loading files took {end_time - start_time:.2f} seconds")
        self.logger.info(f"KG loaded with {len(kg_integrated)} triples")
        self.logger.info(f"Metadata loaded with {len(extraction_metadata_integrated)} triples")
        
        return kg_integrated, extraction_metadata_integrated

    @abstractmethod
    def initialize_extractor(self) -> Any:
        """Initialize the source-specific extractor."""
        pass

    @abstractmethod
    def initialize_transformer(self) -> Any:
        """Initialize the source-specific transformer."""
        pass

    @abstractmethod
    def extract(self, extractor: Any, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Perform the extraction step.
        
        Args:
            extractor: The initialized extractor
            args: Command line arguments
            
        Returns:
            Dict containing extracted entities
        """
        pass

    @abstractmethod
    def transform(self, transformer: Any, extracted_entities: Dict[str, Any], args: argparse.Namespace) -> Tuple[Graph, Graph]:
        """
        Perform the transformation step.
        
        Args:
            transformer: The initialized transformer
            extracted_entities: Dictionary of extracted entities
            args: Command line arguments
            
        Returns:
            Tuple of (knowledge graph, metadata graph)
        """
        pass

    def load(self, loader: LoadProcessor, kg: Graph, metadata: Graph, args: argparse.Namespace) -> None:
        """
        Perform the load step.
        
        Args:
            loader: The initialized loader
            kg: Knowledge graph to load
            metadata: Metadata graph to load
            args: Command line arguments
        """
        if args.remote_db:
            loader.set_upload_timeout(args.upload_timeout)
            self.logger.info(f"Set upload timeout to {args.upload_timeout} seconds for remote database uploads")

        # Clean databases if needed
        # Commented out as it seems to be a common pattern across implementations
        # self.logger.info("Cleaning databases...")
        # start_time = time.time()
        # loader.clean_DBs()
        # end_time = time.time()
        # self.logger.info(f"Database cleaning took {end_time - start_time:.2f} seconds")

        # Load data
        self.logger.info("Starting database update with KG...")
        start_time = time.time()
        
        if args.chunking:
            self.logger.info(f"Using chunk size: {args.chunk_size}")
            loader.update_dbs_with_kg(
                kg,
                metadata,
                extraction_name=f"{self.source_name}_extraction",
                remote_db=args.remote_db,
                kg_chunks_size=args.chunk_size,
                save_load_output=True,
                load_output_dir=f"{self.output_dir}/chunks"
            )
        else:
            loader.update_dbs_with_kg(
                kg,
                metadata,
                extraction_name=f"{self.source_name}_extraction",
                remote_db=args.remote_db,
                kg_chunks_size=0,
                save_load_output=True,
                load_output_dir=f"{self.output_dir}/chunks"
            )
            
        end_time = time.time()
        self.logger.info(f"Database update with KG took {end_time - start_time:.2f} seconds")

    def run(self, args: Optional[List[str]] = None) -> None:
        """
        Main execution flow for the ETL process.
        
        Args:
            args: Optional list of command line arguments
        """
        # Parse arguments
        parser = self.get_argument_parser()
        args = parser.parse_args(args)
        
        # Initialize folder structure
        self.initialize_folder_structure(clean_folders=False)
        
        kg_integrated = Graph()
        extraction_metadata_integrated = Graph()
        
        # Handle file loading mode
        file_loading_mode = args.kg_file_path or args.metadata_file_path
        if file_loading_mode and args.use_dummy_data:
            self.logger.error("Cannot use --kg-file-path/--metadata-file-path with --use-dummy-data")
            return
            
        if file_loading_mode:
            if args.kg_file_path and args.metadata_file_path:
                try:
                    kg_integrated, extraction_metadata_integrated = self.load_existing_files(
                        args.kg_file_path,
                        args.metadata_file_path
                    )
                except FileNotFoundError as e:
                    self.logger.error(str(e))
                    return
            else:
                self.logger.error("Both --kg-file-path and --metadata-file-path must be provided together")
                return
        elif not args.use_dummy_data:
            # Regular ETL process
            try:
                # Extract
                self.logger.info("Initializing extractor...")
                start_time = time.time()
                extractor = self.initialize_extractor()
                end_time = time.time()
                self.logger.info(f"Extractor initialization took {end_time - start_time:.2f} seconds")

                self.logger.info("Starting extraction...")
                extracted_entities = self.extract(extractor, args)

                # Transform
                self.logger.info("Initializing transformer...")
                start_time = time.time()
                transformer = self.initialize_transformer()
                end_time = time.time()
                self.logger.info(f"Transformer initialization took {end_time - start_time:.2f} seconds")

                self.logger.info("Starting transformation...")
                kg_integrated, extraction_metadata_integrated = self.transform(
                    transformer, extracted_entities, args
                )
                
            except Exception as e:
                self.logger.error(f"ETL process failed: {str(e)}")
                raise
        else:
            # Load dummy data - subclasses should override this if needed
            self.logger.error("Dummy data loading not implemented for this source")
            return

        # Load
        self.logger.info("Initializing loader...")
        start_time = time.time()
        loader = self.initialize_load_processor()
        end_time = time.time()
        self.logger.info(f"Loader initialization took {end_time - start_time:.2f} seconds")

        self.load(loader, kg_integrated, extraction_metadata_integrated, args)
        self.logger.info("ETL process completed successfully") 