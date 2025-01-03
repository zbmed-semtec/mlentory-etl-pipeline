from core.QueueObserver import QueueObserver
from core.FilesProcessor import FilesProcessor
from core.FieldProcessorHF import FieldProcessorHF
import argparse
import datetime
import logging
import time


def main():
    # Handling script arguments
    parser = argparse.ArgumentParser(description="Queue Observer Script")
    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        required=False,
        default="./../transform_queue/",
        help="Path to the folder to observe (default: ./../transform_queue/)",
    )
    args = parser.parse_args()

    # Setting up logging system
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"./processing_logs/transform_{timestamp}.log"
    logging.basicConfig(
        filename=filename,
        filemode="w",
        format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    try:
        # Initializing the updater
        fields_processor_HF = FieldProcessorHF(path_to_config_data="./../config_data")

        files_processor = FilesProcessor(
            num_workers=4,
            next_batch_proc_time=30,
            processed_files_log_path="./processing_logs/Processed_files.txt",
            load_queue_path="./../load_queue",
            field_processor_HF=fields_processor_HF,
        )
        observer = QueueObserver(watch_dir=args.folder, files_processor=files_processor)
        observer.start()

        # Keep the script running to monitor changes
        while True:
            observer.files_processor.update_time_to_process()
            time.sleep(0.5)

    except Exception as e:
        logger.exception("Exception occurred ", e)
    except KeyboardInterrupt:
        logger.info("Server Stop")

    logger.info("Server Shutdown")
    observer.stop()


if __name__ == "__main__":
    main()
