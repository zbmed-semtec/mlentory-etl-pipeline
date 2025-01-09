import pandas as pd
import logging
import datetime
from typing import List
from tqdm import tqdm
import os
import argparse

from extractors.hf_extractor import HFExtractor
from transform.core.FieldProcessorHF import FieldProcessorHF
from mlentory_loader.core import LoadProcessor, GraphHandler
from mlentory_loader.dbHandler import RDFHandler, SQLHandler, IndexHandler


def load_tsv_file_to_list(path: str) -> List[str]:
    return [val[0] for val in pd.read_csv(path, sep="\t").values.tolist()]


def setup_logging():
    """
    Sets up the logging system.
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    base_log_path = "./hf_etl/execution_logs"
    os.makedirs(base_log_path, exist_ok=True)
    logging_filename = f"{base_log_path}/transform_{timestamp}.log"

    logging.basicConfig(
        filename=logging_filename,
        filemode="w",
        format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


def initialize_extractor(config_path: str):
    """
    Initializes the extractor with the configuration data.

    Args:
        config_path (str): The path to the configuration data.

    Returns:
        HFExtractor: The extractor instance.
    """
    questions = load_tsv_file_to_list(f"{config_path}/extract/questions.tsv")
    tags_language = load_tsv_file_to_list(f"{config_path}/extract/tags_language.tsv")
    tags_libraries = load_tsv_file_to_list(f"{config_path}/extract/tags_libraries.tsv")
    tags_other = load_tsv_file_to_list(f"{config_path}/extract/tags_other.tsv")
    tags_task = load_tsv_file_to_list(f"{config_path}/extract/tags_task.tsv")
    return HFExtractor(
        qa_model="Intel/dynamic_tinybert",
        questions=questions,
        tags_language=tags_language,
        tags_libraries=tags_libraries,
        tags_other=tags_other,
        tags_task=tags_task,
    )


def initialize_load_processor(kg_files_directory: str):
    """
    Initializes the load processor with the configuration data.

    Args:
        kg_files_directory (str): The path to the kg files directory.

    Returns:
        LoadProcessor: The load processor instance.
    """
    sqlHandler = SQLHandler(
        host="postgres",
        user="user",
        password="password",
        database="history_DB",
    )
    sqlHandler.connect()

    rdfHandler = RDFHandler(
        container_name="virtuoso",
        kg_files_directory=kg_files_directory,
        _user="dba",
        _password="my_strong_password",
        sparql_endpoint="http://virtuoso:8890/sparql",
    )

    elasticsearchHandler = IndexHandler(
        es_host="elastic",
        es_port=9200,
    )

    elasticsearchHandler.initialize_HF_index(index_name="hf_models")

    # Initializing the graph creator
    graphHandler = GraphHandler(
        SQLHandler=sqlHandler,
        RDFHandler=rdfHandler,
        IndexHandler=elasticsearchHandler,
        kg_files_directory=kg_files_directory,
    )

    # Initializing the load processor
    return LoadProcessor(
        SQLHandler=sqlHandler,
        RDFHandler=rdfHandler,
        IndexHandler=elasticsearchHandler,
        GraphHandler=graphHandler,
        kg_files_directory=kg_files_directory,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='HuggingFace ETL Process')
    parser.add_argument('--save-extraction', 
                       action='store_true',
                       default=False,
                       help='Save the results of the extraction phase')
    parser.add_argument('--save-transformation', 
                       action='store_true',
                       default=False,
                       help='Save the results of the transformation phase')
    parser.add_argument('--save-load-data', 
                       action='store_true',
                       default=False,
                       help='Save the data that will be loaded into the database')
    parser.add_argument('--from-date',
                       type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'),
                       default=datetime.datetime(2000, 1, 1),
                       help='Download models from this date (format: YYYY-MM-DD)')
    parser.add_argument('--num-models',
                       type=int,
                       default=5,
                       help='Number of models to download')
    parser.add_argument('--output-dir',
                       default='./output',
                       help='Directory to save intermediate results')
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    
    # Load configuration data
    config_path = "./configuration/hf"  # Path to configuration folder
    kg_files_directory = "./../kg_files"  # Path to kg files directory

    extractor = initialize_extractor(config_path)

    # Download and process models
    extracted_df = extractor.download_models(
        num_models=args.num_models,  # Start with a small number for testing
        from_date=args.from_date,  # Pass the from_date parameter
        save_original=False,
        save_result_in_json=False,
    )

    if args.save_extraction:
        output_path = f"{args.output_dir}/extraction_results.csv"
        extracted_df.to_csv(output_path, index=False)
        print(f"Saved extraction results to {output_path}")

    # Initializing the transformation
    new_schema = pd.read_csv(f"{config_path}/transform/M4ML_schema.tsv", sep="\t")
    transformations = pd.read_csv(
        f"{config_path}/transform/column_transformations.csv",
        lineterminator="\n",
        sep=",",
    )
    fields_processor_HF = FieldProcessorHF(new_schema, transformations)
    processed_models = []

    for row_num, row in tqdm(
        extracted_df.iterrows(), total=len(extracted_df), desc="Transforming progress"
    ):
        model_data = fields_processor_HF.process_row(row)
        processed_models.append(model_data)

    m4ml_models_df = pd.DataFrame(list(processed_models))

    if args.save_transformation:
        output_path = f"{args.output_dir}/transformation_results.csv"
        m4ml_models_df.to_csv(output_path, index=False)
        print(f"Saved transformation results to {output_path}")

    # Initialize the load processor
    load_processor = initialize_load_processor(kg_files_directory)
    
    load_processor.update_dbs_with_df(df=m4ml_models_df)
    
    if args.save_load_data:
        output_path = f"{args.output_dir}/load_data.csv"
        m4ml_models_df.to_csv(output_path, index=False)
        print(f"Saved load data to {output_path}")
    


if __name__ == "__main__":
    main()
