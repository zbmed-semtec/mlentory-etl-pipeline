import os
import argparse
from typing import Dict, Any, List, Tuple
from rdflib import Graph
import pandas as pd

from ..base_etl.base_etl_component import BaseETLComponent
from mlentory_extract.ai4life_extract import AI4LifeExtractor
from mlentory_transform.core import (
    MlentoryTransform,
    KnowledgeGraphHandler,
    MlentoryTransformWithGraphBuilder,
)

class AI4LifeETLComponent(BaseETLComponent):
    """
    AI4Life-specific implementation of the ETL component.
    Handles extraction, transformation, and loading of AI4Life models and related entities.
    """
    
    def __init__(self, config_path: str = "./configuration/ai4life", output_dir: str = None):
        """
        Initialize the AI4Life ETL component.

        Args:
            config_path (str): Path to AI4Life-specific configuration directory
            output_dir (str): Directory for output files
        """
        super().__init__(
            source_name="ai4life",
            config_path=config_path,
            output_dir=output_dir or "./ai4life_etl/outputs/files"
        )

    def initialize_extractor(self) -> AI4LifeExtractor:
        """Initialize the AI4Life-specific extractor."""
        schema_file = f"{self.config_path}/extract/model_mapping.tsv"
        return AI4LifeExtractor(schema_file=schema_file)

    def initialize_transformer(self) -> MlentoryTransformWithGraphBuilder:
        """Initialize the AI4Life-specific transformer."""
        new_schema = pd.read_csv(
            f"{self.config_path}/transform/FAIR4ML_schema.csv",
            sep=",",
            lineterminator="\n"
        )

        return MlentoryTransformWithGraphBuilder(
            base_namespace="https://w3id.org/mlentory/mlentory_graph/",
            FAIR4ML_schema_data=new_schema
        )

    def get_argument_parser(self) -> argparse.ArgumentParser:
        """Get AI4Life-specific argument parser."""
        parser = super().get_argument_parser()
        
        # Add AI4Life-specific arguments
        parser.add_argument(
            "--num-models",
            type=int,
            default=1000,
            help="Number of models to download"
        )
        parser.add_argument(
            "--save-load-data",
            action="store_true",
            default=False,
            help="Save the data that will be loaded into the database"
        )
        parser.add_argument(
            "--load-extraction-and-transform-data",
            "-led",
            default=False,
            help="Load the extraction data into the database"
        )
        
        return parser

    def extract(self, extractor: AI4LifeExtractor, args: argparse.Namespace) -> Dict[str, Any]:
        """Perform AI4Life-specific extraction."""
        return extractor.download_modelfiles_with_additional_entities(
            num_models=args.num_models,
            output_dir=args.output_dir,
            additional_entities=["dataset", "application"]
        )

    def transform(self, transformer: MlentoryTransformWithGraphBuilder,
                 extracted_entities: Dict[str, Any],
                 args: argparse.Namespace) -> Tuple[Graph, Graph]:
        """Perform AI4Life-specific transformation."""
        return transformer.transform_AI4Life_models_with_related_entities(
            extracted_entities=extracted_entities,
            save_output=True,
            kg_output_dir=args.output_dir,
        )

def main():
    """Main entry point for AI4Life ETL process."""
    etl = AI4LifeETLComponent()
    etl.run()

if __name__ == "__main__":
    main() 