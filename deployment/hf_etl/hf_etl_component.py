import os
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple
from rdflib import Graph
import pandas as pd

from ..base_etl.base_etl_component import BaseETLComponent
from mlentory_extract.hf_extract import HFDatasetManager, HFExtractor
from mlentory_extract.core import ModelCardToSchemaParser
from mlentory_transform.core import MlentoryTransformWithGraphBuilder

class HuggingFaceETLComponent(BaseETLComponent):
    """
    HuggingFace-specific implementation of the ETL component.
    Handles extraction, transformation, and loading of HuggingFace models and related entities.
    """
    
    def __init__(self, config_path: str = "./configuration/hf", output_dir: str = None):
        """
        Initialize the HuggingFace ETL component.

        Args:
            config_path (str): Path to HF-specific configuration directory
            output_dir (str): Directory for output files
        """
        super().__init__(
            source_name="hf",
            config_path=config_path,
            output_dir=output_dir or "./hf_etl/outputs/files"
        )

    def load_tsv_file_to_list(self, path: str) -> List[str]:
        """Helper method to load TSV file to list."""
        return [val[0] for val in pd.read_csv(path, sep="\t").values.tolist()]

    def load_tsv_file_to_df(self, path: str) -> pd.DataFrame:
        """Helper method to load TSV file to DataFrame."""
        return pd.read_csv(path, sep="\t")

    def load_models_from_file(self, file_path: str) -> List[str]:
        """
        Load model IDs from a text file.

        Args:
            file_path (str): Path to the text file containing model IDs

        Returns:
            List[str]: List of model IDs
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model list file not found: {file_path}")

        model_ids = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):  # Skip empty lines and comments
                    continue
                
                id_match = line.split("/")[-1]
                if id_match:
                    model_ids.append(id_match)
                else:
                    self.logger.warning(f"Skipping invalid line in model list file: {line}")

        self.logger.info(f"Loaded {len(model_ids)} model IDs from {file_path}")
        return model_ids

    def initialize_extractor(self) -> HFExtractor:
        """Initialize the HuggingFace-specific extractor."""
        # Load configuration data
        tags_language = self.load_tsv_file_to_list(f"{self.config_path}/extract/tags_language.tsv")
        tags_libraries = self.load_tsv_file_to_df(f"{self.config_path}/extract/tags_libraries.tsv")
        tags_other = self.load_tsv_file_to_df(f"{self.config_path}/extract/tags_other.tsv")
        tags_task = self.load_tsv_file_to_df(f"{self.config_path}/extract/tags_task.tsv")
        
        # Initialize dataset manager
        dataset_manager = HFDatasetManager(api_token=os.getenv("HF_TOKEN"))
        
        # Initialize parser
        parser = ModelCardToSchemaParser(
            qa_model_name="Qwen/Qwen3-1.7B",
            matching_model_name="Alibaba-NLP/gte-base-en-v1.5",
            schema_file=f"{self.config_path}/transform/FAIR4ML_schema.tsv",
            tags_language=tags_language,
            tags_libraries=tags_libraries,
            tags_other=tags_other,
            tags_task=tags_task,
        )
        
        return HFExtractor(
            dataset_manager=dataset_manager,
            parser=parser
        )

    def initialize_transformer(self) -> MlentoryTransformWithGraphBuilder:
        """Initialize the HuggingFace-specific transformer."""
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
        """Get HuggingFace-specific argument parser."""
        parser = super().get_argument_parser()
        
        # Add HuggingFace-specific arguments
        parser.add_argument(
            "--from-date",
            type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
            default=datetime(2000, 1, 1),
            help="Download models from this date (format: YYYY-MM-DD)",
        )
        parser.add_argument(
            "--num-models",
            "-nm",
            type=int,
            default=20,
            help="Number of models to download"
        )
        parser.add_argument(
            "--num-datasets",
            "-nd",
            type=int,
            default=20,
            help="Number of datasets to download"
        )
        parser.add_argument(
            "--model-list-file",
            "-mlf",
            type=str,
            help="Path to a text file containing a list of Hugging Face model IDs"
        )
        parser.add_argument(
            "--unstructured-text-strategy",
            "-uts",
            type=str,
            default="None",
            help="Strategy to use for unstructured text extraction",
        )
        
        return parser

    def extract(self, extractor: HFExtractor, args: argparse.Namespace) -> Dict[str, Any]:
        """Perform HuggingFace-specific extraction."""
        entities_to_download_config = ["datasets", "articles", "keywords", "base_models", "licenses"]
        
        if args.model_list_file:
            self.logger.info(f"Processing models from file: {args.model_list_file}")
            try:
                model_ids = self.load_models_from_file(args.model_list_file)
                if not model_ids:
                    self.logger.warning("Model list file is empty or contains no valid IDs")
                    return {"models": pd.DataFrame()}
                    
                return extractor.download_specific_models_with_related_entities(
                    model_ids=model_ids,
                    output_dir=args.output_dir,
                    save_result_in_json=args.save_extraction,
                    threads=4,
                    related_entities_to_download=entities_to_download_config,
                    unstructured_text_strategy=args.unstructured_text_strategy,
                )
            except FileNotFoundError as e:
                self.logger.error(str(e))
                return {"models": pd.DataFrame()}
        else:
            self.logger.info(f"Downloading {args.num_models} models from {args.from_date.strftime('%Y-%m-%d')}")
            return extractor.download_models_with_related_entities(
                num_models=args.num_models,
                from_date=args.from_date,
                output_dir=args.output_dir,
                save_initial_data=False,
                save_result_in_json=args.save_extraction,
                update_recent=True,
                related_entities_to_download=entities_to_download_config,
                unstructured_text_strategy=args.unstructured_text_strategy,
                threads=4,
                depth=2,
            )

    def transform(self, transformer: MlentoryTransformWithGraphBuilder, 
                 extracted_entities: Dict[str, Any],
                 args: argparse.Namespace) -> Tuple[Graph, Graph]:
        """Perform HuggingFace-specific transformation."""
        return transformer.transform_HF_models_with_related_entities(
            extracted_entities=extracted_entities,
            save_output=True,
            kg_output_dir=f"{args.output_dir}/kg",
            extraction_metadata_output_dir=f"{args.output_dir}/extraction_metadata",
        )

def main():
    """Main entry point for HuggingFace ETL process."""
    etl = HuggingFaceETLComponent()
    etl.run()

if __name__ == "__main__":
    main() 