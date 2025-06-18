import os
import json
import re
import time
import openml
import pandas as pd
from datetime import datetime
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

class OpenMLExtractor:
    """
    A class for extracting and processing model information from OpenML.

    This class provides functionality to:
    - Download run information from OpenML
    - Download dataset information from OpenML
    - Save results in various format

    """

    def __init__(self, schema_file: str, logger):
        """
        Initialize the extractor with a metadata schema.

        Args:
            schema_file (str): Path to the JSON file containing the metadata schema.
            logger : To log the code flow / Errors
        """

        self.schema = self._load_schema(schema_file)
        self.logger = logger

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
        
    def _wrap_metadata(self, value, method="openml_python_package"):
        return [{
            "data": value,
            "extraction_method": method,
            "confidence": 1,
            "extraction_time": datetime.utcnow().isoformat()
        }]
    
    def _scrape_openml_stats(self, dataset_id: int) -> Dict:
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            url = f"https://www.openml.org/search?type=data&id={dataset_id}"
            self.logger.info(f"Starting scrape for dataset ID {dataset_id}: {url}")
            driver.get(url)
            wait = WebDriverWait(driver, 10)
            stats = {}
            for stat in ['status', 'downloads', 'likes', 'issues']:
                try:
                    element = wait.until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, f'span[aria-label="{stat}"]')
                        )
                    )
                    stats[stat] = element.text.strip()
                    self.logger.debug(f"Extracted {stat}: {stats[stat]}")   
                except TimeoutException:
                    stats[stat] = "N/A"
                    self.logger.warning(f"Timeout while scraping '{stat}' for dataset {dataset_id}")
            
            def extract_number(stat_string):
                return int(re.sub(r'\D', '', stat_string)) if stat_string != "N/A" else 0
            
            stats['downloads'] = extract_number(stats['downloads'])
            stats['likes'] = extract_number(stats['likes'])
            stats['issues'] = extract_number(stats['issues'])
            self.logger.info(f"Completed scraping stats for dataset {dataset_id}: {stats}")
            return stats
            
        except Exception as e:
            self.logger.debug(f"Error scraping stats for dataset {dataset_id}: {str(e)}")
            print(f"Error scraping stats for dataset {dataset_id}: {str(e)}")
            return {"status": "N/A", "downloads": 0, "likes": 0, "issues": 0}
        
        finally:
            driver.quit()

    def extract_run_info_with_additional_entities(
        self, 
        num_instances: int, 
        offset: int = 0,
        threads: int = 4, 
        output_dir: str = "./outputs",
        save_result_in_json: bool = False,
        additional_entities: List[str] = ["dataset"]
        )-> Dict:
        """
        Download run info with all adiitional entities specified.

        Args:
            num_instances (int): Number of instances to fetch.
            offset (int): Number of instances to skip before fetching.
            threads (int): Number of threads for parallel processing. Defaults to 4.
            output_dir (str): Directory to save the output file.
            save_result_in_json (bool): Whether to save the results in JSON format. Defaults to False.
            additional_entities

        Returns:
            Dict: Dict containing metadata for the runs and aditional entities.
        """
        self.logger.info(
            f"Starting extraction of run info. "
            f"num_instances={num_instances}, offset={offset}, threads={threads}, "
            f"output_dir={output_dir}, save_result_in_json={save_result_in_json}, "
            f"additional_entities={additional_entities}"
        )
        extracted_entities = {}
        try: 
            self.logger.info("Fetching run metadata...")
            run_metadata_df = self.get_multiple_runs_metadata(num_instances, offset, threads, output_dir, save_result_in_json)
            extracted_entities["run"] = run_metadata_df

            for entity in additional_entities:
                if entity == "dataset":
                    self.logger.info("Fetching dataset metadata...")
                    dataset_metadata_df = self.get_multiple_datasets_metadata(num_instances, offset, threads, output_dir, save_result_in_json)
                    extracted_entities["dataset"] = dataset_metadata_df
                    self.logger.info("Dataset metadata extraction complete.")
                else:
                    self.logger.warning(f"Unsupported additional entity: {entity}")
            return extracted_entities
        except Exception as e:
            self.logger.debug(f"Failed during extraction: {str(e)}", exc_info=True)


    def _get_recent_run_ids(self, num_instances: int = 10, offset: int = 0) -> List[int]:
        """
        Get a list of recent run IDs.

        Args:
            num_instances (int): Number of run IDs to fetch.
            offset (int): Number of runs to skip before fetching.

        Returns:
            List[int]: List of run IDs.
        """ 
    
        self.logger.info(f"Fetching {num_instances} recent run IDs with offset {offset}")
        try:
            runs = openml.runs.list_runs(size=num_instances, offset=offset, output_format="dataframe")
            run_ids = runs["run_id"].tolist()[:num_instances]
            self.logger.debug(f"Fetched run IDs: {run_ids}")
            return run_ids
        except Exception as e:
            self.logger.debug(f"Error fetching recent run IDs: {str(e)}", exc_info=True)
        

    def _get_recent_dataset_ids(self, num_instances: int = 10, offset: int = 0) -> List[int]:
        """
        Get a list of recent dataset IDs.

        Args:
            num_instances (int): Number of dataset IDs to fetch.
            offset (int): Number of runs to skip before fetching.

        Returns:
            List[int]: List of dataset IDs.
        """
        self.logger.info(f"Fetching {num_instances} recent dataset IDs with offset {offset}")
        try:
            datasets = openml.datasets.list_datasets(size=num_instances, offset=offset, output_format="dataframe")
            dataset_ids = datasets["did"].tolist()[:num_instances]
            self.logger.debug(f"Fetched dataset IDs: {dataset_ids}")
            return dataset_ids
        except Exception as e:
            self.logger.debug(f"Error fetching recent dataset IDs: {str(e)}", exc_info=True)

    def get_run_metadata(self, run_id: int) -> Dict:
        """
        Fetch metadata for a single run using the schema.

        Args:
            run_id (int): The ID of the run.

        Returns:
            Optional[Dict]: Metadata for the run, or None if an error occurs.
        """
        self.logger.info(f"Fetching metadata for run_id={run_id}")
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
                            if obj:
                                nested_data[sub_key] = getattr(obj, attr) 

                    metadata[key] = self._wrap_metadata(nested_data)  # Store as a dictionary

                elif isinstance(path, list):
                    # Handle list of commands case
                    combined_result = None
                    for command in path:
                        if "{" in command and "}" in command:  # Handle formatted strings
                            result = command.format(run=run)
                        else:
                            obj_name, attr = command.split(".")
                            obj = obj_map.get(obj_name)
                            if obj:
                                result = getattr(obj, attr)
                            else:
                                result = None
                        
                        # Combine results
                        if result is not None:
                            if combined_result is None:
                                combined_result = result
                            else:
                                # If both are strings, combine with space
                                if isinstance(combined_result, str) and isinstance(result, str):
                                    combined_result = f"{combined_result} {result}"
                                # If both are lists, combine the lists
                                elif isinstance(combined_result, list) and isinstance(result, list):
                                    combined_result.extend(result)
                                # If one is list and other is string, convert string to list and combine
                                elif isinstance(combined_result, list) and isinstance(result, str):
                                    combined_result.append(result)
                                elif isinstance(combined_result, str) and isinstance(result, list):
                                    combined_result = [combined_result] + result
                                # For other types, convert to string and combine
                                else:
                                    combined_result = f"{str(combined_result)} {str(result)}"
                    
                    metadata[key] = self._wrap_metadata(combined_result)
                else:
                    if "{" in path and "}" in path: 
                        metadata[key] = self._wrap_metadata(path.format(run=run))
                    else:
                        obj_name, attr = path.split(".")
                        obj = obj_map.get(obj_name)
                        if obj:
                            metadata[key] = self._wrap_metadata(getattr(obj, attr))
            self.logger.debug(f"Successfully fetched run metadata for run_id={run_id}")
            return metadata

        except Exception as e:
            self.logger.debug(f"Error fetching metadata for run {run_id}: {str(e)}", exc_info=True)

    def get_dataset_metadata(self, dataset_id, datasets_df):
        """
        Fetch metadata for a single dataset using the schema.

        Args:
            dataset_id (int): The ID of the dataset.
            dataset_df (pd.Dataframe) : All the Dataset information in a panda dataframe

        Returns:
            Optional[Dict]: Metadata for the dataset, or None if an error occurs.
        """
        self.logger.info(f"Fetching metadata for dataset_id={dataset_id}")
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            scraped_stats = self._scrape_openml_stats(dataset_id)
            
            obj_map = {"dataset": dataset}
            metadata = {}
            for key, path in self.schema.get("dataset", {}).items():
                if key in ["schema.org:status", "likes", "downloads", "issues"]:
                    if key == "schema.org:status":
                        api_status = datasets_df.loc[datasets_df['did'] == dataset_id, 'status'].values[0] if dataset_id in datasets_df['did'].values else "N/A"
                        scraped_status = scraped_stats.get("status", "N/A")
                        status = scraped_status if scraped_status != "N/A" else api_status
                        metadata[key] = self._wrap_metadata(status, method="web_scraping" if scraped_status != "N/A" else "openml_python_package")
                    elif key == "likes":
                        metadata[key] = self._wrap_metadata(scraped_stats.get("likes", 0), method="web_scraping")
                    elif key == "downloads":
                        metadata[key] = self._wrap_metadata(scraped_stats.get("downloads", 0), method="web_scraping")
                    elif key == "issues":
                        metadata[key] = self._wrap_metadata(scraped_stats.get("issues", 0), method="web_scraping")
                else:
                    obj_name, attr = path.split(".")
                    obj = obj_map.get(obj_name)
                    if obj:
                        value = getattr(obj, attr)
                        if key == "version":
                            value = str(value)
                        metadata[key] = self._wrap_metadata(value, method="openml_python_package")
            self.logger.debug(f"Successfully fetched dataset metadata for dataset_id={dataset_id}")
            return metadata
        except Exception as e:
            self.logger.debug(f"Error fetching metadata for dataset {dataset_id}: {str(e)}", exc_info=True)

    def get_multiple_runs_metadata(
        self, 
        num_instances: int, 
        offset: int = 0,
        threads: int = 4, 
        output_dir: str = "./outputs",
        save_result_in_json: bool = False
    ) -> pd.DataFrame:
        """
        Fetch metadata for multiple runs using multithreading.

        Args:
            num_instances (int): Number of runs to fetch.
            offset (int): Number of runs to skip before fetching.
            threads (int): Number of threads for parallel processing. Defaults to 4.
            output_dir (str): Directory to save the output file.
            save_result_in_json (bool): Whether to save the results in JSON format. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing metadata for the runs.
        """
        self.logger.info(f"Fetching metadata for {num_instances} runs with offset={offset}")
        run_ids = self._get_recent_run_ids(num_instances, offset)
        run_metadata = []

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(self.get_run_metadata, run_id): run_id for run_id in run_ids}

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    run_metadata.append(result)

        run_metadata_df = pd.DataFrame(run_metadata)

        if save_result_in_json:
            self.logger.info("Saving run metadata to JSON")
            self.save_results(run_metadata_df, output_dir, 'run')

        self.logger.debug("Finished fetching run metadata")
        return run_metadata_df
    
    def get_multiple_datasets_metadata(
        self, 
        num_instances: int, 
        offset: int = 0,
        threads: int = 4, 
        output_dir: str = "./outputs",
        save_result_in_json: bool = False
    ) -> pd.DataFrame:
        """
        Fetch metadata for multiple runs using multithreading.

        Args:
            num_instances (int): Number of datasets to fetch.
            offset (int): Number of datasets to skip before fetching.
            threads (int): Number of threads for parallel processing. Defaults to 4.
            output_dir (str): Directory to save the output file.
            save_result_in_json (bool): Whether to save the results in JSON format. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing metadata for the datasets.
        """
        self.logger.info(f"Fetching metadata for {num_instances} datasets with offset={offset}")
        dataset_ids = self._get_recent_dataset_ids(num_instances, offset)
        datasets_df = openml.datasets.list_datasets(output_format="dataframe")
        dataset_metadata = []

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(self.get_dataset_metadata, dataset_id, datasets_df): dataset_id for dataset_id in dataset_ids}

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    dataset_metadata.append(result)

        dataset_metadata_df = pd.DataFrame(dataset_metadata)

        if save_result_in_json:
            self.logger.info("Saving dataset metadata to JSON")
            self.save_results(dataset_metadata_df, output_dir, 'dataset')

        self.logger.debug("Finished fetching dataset metadata")
        return dataset_metadata_df

    def save_results(self, df: pd.DataFrame, output_dir: str, entity: str):
        """
        Save the results to a file.

        Args:
            df (pd.DataFrame): DataFrame containing the metadata.
            output_dir (str): Directory to save the output file.
            entity (str): Entity for which metadata is being saved.
        """
        self.logger.info(f"Saving {entity} metadata to output directory: {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(output_dir, f"openml_{entity}_metadata_{timestamp}.json")
        df.to_json(output_path, orient="records", indent=4)
        self.logger.debug(f"Metadata saved to {output_path}")
    