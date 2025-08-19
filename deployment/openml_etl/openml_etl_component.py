import os
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple
from rdflib import Graph
import pandas as pd

from ..base_etl.base_etl_component import BaseETLComponent
from mlentory_extract.openml_extract import OpenMLExtractor
from mlentory_transform.core.MlentoryTransform import (
    MlentoryTransform,
    KnowledgeGraphHandler,
)

class OpenMLETLComponent(BaseETLComponent):
    """
    OpenML-specific implementation of the ETL component.
    Handles extraction, transformation, and loading of OpenML runs and related entities.
    """
    
    def __init__(self, config_path: str = "./configuration/openml", output_dir: str = None):
        """
        Initialize the OpenML ETL component.

        Args:
            config_path (str): Path to OpenML-specific configuration directory
            output_dir (str): Directory for output files
        """
        super().__init__(
            source_name="openml",
            config_path=config_path,
            output_dir=output_dir or "./openml_etl/outputs"
        )

    def initialize_extractor(self) -> OpenMLExtractor:
        """Initialize the OpenML-specific extractor."""
        schema_file = f"{self.config_path}/extract/metadata_schema.json"
        self.logger.info(f"Using schema file: {schema_file}")
        
        return OpenMLExtractor(schema_file=schema_file, logger=self.logger)

    def initialize_transformer(self) -> MlentoryTransform:
        """Initialize the OpenML-specific transformer."""
        new_schema = pd.read_csv(
            f"{self.config_path}/transform/FAIR4ML_schema.csv",
            sep=",",
            lineterminator="\n"
        )
        
        kg_handler = KnowledgeGraphHandler(
            FAIR4ML_schema_data=new_schema,
            base_namespace="https://w3id.org/mlentory/mlentory_graph/"
        )
        
        return MlentoryTransform(kg_handler, None)

    def get_argument_parser(self) -> argparse.ArgumentParser:
        """Get OpenML-specific argument parser."""
        parser = super().get_argument_parser()
        
        # Add OpenML-specific arguments
        parser.add_argument(
            "--num-instances",
            "-nm",
            type=int,
            default=20,
            help="Number of instances to extract metadata from"
        )
        parser.add_argument(
            "--offset",
            type=int,
            default=0,
            help="Number of instances to skip before extracting"
        )
        parser.add_argument(
            "--threads",
            type=int,
            default=4,
            help="Number of threads for parallel processing"
        )
        
        return parser

    def extract(self, extractor: OpenMLExtractor, args: argparse.Namespace) -> Dict[str, Any]:
        """Perform OpenML-specific extraction."""
        self.logger.info("Extracting run info with additional entities")
        return extractor.extract_run_info_with_additional_entities(
            num_instances=args.num_instances,
            offset=args.offset,
            threads=args.threads,
            output_dir=args.output_dir,
            save_result_in_json=args.save_extraction,
            additional_entities=["dataset"],
        )

    def transform(self, transformer: MlentoryTransform,
                 extracted_entities: Dict[str, Any],
                 args: argparse.Namespace) -> Tuple[Graph, Graph]:
        """Perform OpenML-specific transformation."""
        # Transform runs
        self.logger.info("Transforming OpenML runs")
        runs_kg, runs_extraction_metadata = transformer.transform_OpenML_runs(
            extracted_df=extracted_entities["run"],
            save_output_in_json=args.save_transformation,
            output_dir=f"{args.output_dir}/runs",
        )

        # Transform datasets
        self.logger.info("Transforming OpenML datasets")
        datasets_kg, datasets_extraction_metadata = transformer.transform_OpenML_datasets(
            extracted_df=extracted_entities["dataset"],
            save_output_in_json=args.save_transformation,
            output_dir=f"{args.output_dir}/datasets",
        )

        # Unify graphs
        self.logger.info("Unifying knowledge graphs")
        kg_integrated = transformer.unify_graphs(
            [runs_kg, datasets_kg],
            save_output_in_json=args.save_transformation,
            output_dir=f"{args.output_dir}/kg",
        )
        
        self.logger.info("Unifying extraction metadata")
        extraction_metadata_integrated = transformer.unify_graphs(
            [runs_extraction_metadata, datasets_extraction_metadata],
            save_output_in_json=args.save_transformation,
            output_dir=f"{args.output_dir}/extraction_metadata",
        )

        return kg_integrated, extraction_metadata_integrated

def main():
    """Main entry point for OpenML ETL process."""
    etl = OpenMLETLComponent()
    etl.run()

if __name__ == "__main__":
    main() 