import os
import json
import openml
import argparse
import pandas as pd
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

class OpenMLExtractor:
    """
    A class for extracting and processing model information from OpenML.

    This class provides functionality to:
    - Download run information from OpenML
    - Download dataset information from OpenML
    - Save results in various format

    """

    def __init__(self, schema_file: str):
        """
        Initialize the extractor with a metadata schema.

        Args:
            schema_file (str): Path to the JSON file containing the metadata schema.
        """

        self.schema = self._load_schema(schema_file)

        if not self.schema:
            raise ValueError("Invalid or empty schema file.")

    def _load_schema(self, schema_file: str) -> Dict:
        """
        Load the metadata schema from a JSON file.

        Args:
            schema_file (str): Path to the JSON file.

        Returns:
            Dict: Metadata schema.
        """
        if not os.path.exists(schema_file):
            raise FileNotFoundError(f"Schema file '{schema_file}' not found.")

        with open(schema_file, "r") as f:
            return json.load(f)

    def _get_recent_run_ids(self, num_runs: int = 10) -> List[int]:
        """
        Get a list of recent run IDs.

        Args:
            num_runs (int): Number of run IDs to fetch.

        Returns:
            List[int]: List of run IDs.
        """
        runs = openml.runs.list_runs(size=num_runs, output_format="dataframe")
        return runs.index.tolist()[:num_runs]

    def get_run_metadata(self, run_id: int) -> Dict:
        """
        Fetch metadata for a single run using the schema.

        Args:
            run_id (int): The ID of the run.

        Returns:
            Optional[Dict]: Metadata for the run, or None if an error occurs.
        """
        try:
            run = openml.runs.get_run(run_id)
            dataset = openml.datasets.get_dataset(run.dataset_id)
            flow = openml.flows.get_flow(run.flow_id)
            task = openml.tasks.get_task(run.task_id)

            obj_map = {"run": run, "dataset": dataset, "flow": flow, "task": task}
            metadata = {}
            for key, path in self.schema.get("run", {}).items():
                if isinstance(path, dict): 
                    nested_data = {}
                    for sub_key, sub_path in path.items():
                        if "{" in sub_path and "}" in sub_path:  # Handle formatted strings
                            nested_data[sub_key] = sub_path.format(run=run)
                        else:
                            obj_name, attr = sub_path.split(".")
                            obj = obj_map.get(obj_name)
                            nested_data[sub_key] = getattr(obj, attr, None) if obj else None

                    metadata[key] = nested_data  # Store as a dictionary
                else:
                    if "{" in path and "}" in path: 
                        metadata[key] = path.format(run=run)
                    else:
                        obj_name, attr = path.split(".")
                        obj = obj_map.get(obj_name)
                        if obj:
                            metadata[key] = getattr(obj, attr, None)

            return metadata

        except Exception as e:
            print(f"Error fetching metadata for run {run_id}: {str(e)}")

    def get_multiple_runs_metadata(
        self, 
        num_runs: int, 
        threads: int = 4, 
        output_dir: str = "./outputs",
        save_result_in_json: bool = False
    ) -> pd.DataFrame:
        """
        Fetch metadata for multiple runs using multithreading.

        Args:
            num_runs (int): Number of runs to fetch.
            threads (int): Number of threads for parallel processing. Defaults to 4.
            output_dir (str): Directory to save the output file.
            save_result_in_json (bool): Whether to save the results in JSON format. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing metadata for the runs.
        """
        run_ids = self._get_recent_run_ids(num_runs)
        print("RUN IDS: ", run_ids)
        run_metadata = []

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(self.get_run_metadata, run_id): run_id for run_id in run_ids}

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    run_metadata.append(result)

        metadata_df = pd.DataFrame(run_metadata)

        if save_result_in_json:
            self.save_results(metadata_df, output_dir)

        return metadata_df

    def save_results(self, df: pd.DataFrame, output_dir: str):
        """
        Save the results to a file.

        Args:
            df (pd.DataFrame): DataFrame containing the metadata.
            output_dir (str): Directory to save the output file.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(output_dir, f"openml_runs_metadata_{timestamp}.json")
        df.to_json(output_path, orient="records", indent=4)
    