import pandas as pd
import logging
import datetime
from typing import List

from extractors.hf_extractor import HFExtractor
from transform.core.FilesProcessor import FilesProcessor
from transform.core.FieldProcessorHF import FieldProcessorHF

def load_tsv_file_to_list(path: str) -> List[str]:
    return [val[0] for val in pd.read_csv(path, sep="\t").values.tolist()]

def main():
    
    args = parser.parse_args()

    # Setting up logging system
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"./execution_logs/transform_{timestamp}.log"
    
    logging.basicConfig(
        filename=filename,
        filemode="w",
        format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


    # Load configuration data
    config_path = "../config_data"  # Path to configuration folder
    
    questions = load_tsv_file_to_list(f"{config_path}/questions.tsv")
    tags_language = load_tsv_file_to_list(f"{config_path}/tags_language.tsv")
    tags_libraries = load_tsv_file_to_list(f"{config_path}/tags_libraries.tsv")
    tags_other = load_tsv_file_to_list(f"{config_path}/tags_other.tsv")
    tags_task = load_tsv_file_to_list(f"{config_path}/tags_task.tsv")
    
    # Initialize extractor with configuration
    extractor = HFExtractor(
        qa_model="Intel/dynamic_tinybert",
        questions=questions,
        tags_language=tags_language,
        tags_libraries=tags_libraries,
        tags_other=tags_other,
        tags_task=tags_task,
    )
    
    # Download and process models
    df = extractor.download_models(
        num_models=5,  # Start with a small number for testing
        # output_dir="/transform_queue",  # Mount point in container
        save_original=False,
        save_result_in_json=False
    )
    
    print(f"Processed {len(df)} models")
    print("\nSample results:")
    print(df.head())
    
    # Initializing the updater
    fields_processor_HF = FieldProcessorHF(path_to_config_data="./../config_data")

    files_processor = FilesProcessor(
        num_workers=4,
        next_batch_proc_time=30,
        processed_files_log_path="./processing_logs/Processed_files.txt",
        load_queue_path="./../load_queue",
        field_processor_HF=fields_processor_HF,
    )

if __name__ == "__main__":
    main()