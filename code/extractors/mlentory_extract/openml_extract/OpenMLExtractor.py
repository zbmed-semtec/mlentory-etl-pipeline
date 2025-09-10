import os
import json
import re
import time
import random
import openml
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.service import Service
from threading import Lock
import threading

class BrowserPool:
    """Thread-safe browser pool for reusing browser instances"""
    def __init__(self, max_browsers=4, timeout=30):
        self.max_browsers = max_browsers
        self.timeout = timeout
        self.available_browsers = []
        self.in_use_browsers = set()
        self.lock = Lock()
        self.browser_creation_lock = Lock()
        
    def _create_browser(self):
        """Create a new browser instance with optimized settings"""
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--disable-javascript")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--memory-pressure-off")
        chrome_options.add_argument("--max_old_space_size=4096")
        
        # Randomize user agent
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        chrome_options.add_argument(f"--user-agent={random.choice(user_agents)}")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(self.timeout)
            driver.implicitly_wait(10)
            return driver
        except Exception as e:
            raise WebDriverException(f"Failed to create browser: {str(e)}")
    
    def get_browser(self):
        """Get an available browser from the pool"""
        with self.lock:
            if self.available_browsers:
                browser = self.available_browsers.pop()
                self.in_use_browsers.add(browser)
                return browser
            
            if len(self.in_use_browsers) < self.max_browsers:
                with self.browser_creation_lock:
                    try:
                        browser = self._create_browser()
                        self.in_use_browsers.add(browser)
                        return browser
                    except Exception:
                        return None
            
            return None
    
    def return_browser(self, browser):
        """Return a browser to the pool"""
        with self.lock:
            if browser in self.in_use_browsers:
                self.in_use_browsers.remove(browser)
                try:
                    # Clear cookies and cache
                    browser.delete_all_cookies()
                    browser.execute_script("window.localStorage.clear();")
                    browser.execute_script("window.sessionStorage.clear();")
                    self.available_browsers.append(browser)
                except Exception:
                    # If cleanup fails, quit the browser
                    try:
                        browser.quit()
                    except:
                        pass
    
    def close_all(self):
        """Close all browsers in the pool"""
        with self.lock:
            all_browsers = list(self.available_browsers) + list(self.in_use_browsers)
            for browser in all_browsers:
                try:
                    browser.quit()
                except:
                    pass
            self.available_browsers.clear()
            self.in_use_browsers.clear()

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
        self.browser_pool = None
        self.scraping_enabled = False
        self.request_delays = [1, 2, 3, 5, 8]  # Exponential backoff delays
        self.max_retries = 3

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
    
    def _init_browser_pool(self, max_browsers=4):
        """Initialize the browser pool if not already initialized"""
        if self.browser_pool is None:
            self.browser_pool = BrowserPool(max_browsers=max_browsers)
    
    def _scrape_openml_stats_with_retry(self, dataset_id: int, max_retries: int = 3) -> Dict:
        """
        Scrape OpenML stats with retry logic and exponential backoff
        
        Args:
            dataset_id (int): The dataset ID to scrape
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            Dict: Scraped statistics or default values
        """
        if not self.scraping_enabled:
            return {"status": "N/A", "downloads": 0, "likes": 0, "issues": 0}
        
        for attempt in range(max_retries + 1):
            try:
                return self._scrape_openml_stats(dataset_id)
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for dataset {dataset_id}: {str(e)}")
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = self.request_delays[min(attempt, len(self.request_delays) - 1)]
                    jitter = random.uniform(0.5, 1.5)
                    time.sleep(delay * jitter)
                else:
                    self.logger.error(f"All {max_retries + 1} attempts failed for dataset {dataset_id}")
                    # Disable scraping if too many failures
                    if "ERR_CONNECTION_CLOSED" in str(e):
                        self.scraping_enabled = False
                        self.logger.error("Disabling scraping due to connection issues")
        
        return {"status": "N/A", "downloads": 0, "likes": 0, "issues": 0}
    
    def _scrape_openml_stats(self, dataset_id: int) -> Dict:
        """
        Scrape OpenML stats using browser pool with robust error handling
        
        Args:
            dataset_id (int): The dataset ID to scrape
            
        Returns:
            Dict: Scraped statistics
        """
        self._init_browser_pool()
        
        browser = self.browser_pool.get_browser()
        if not browser:
            raise Exception("No browser available from pool")
        
        try:
            url = f"https://www.openml.org/search?type=data&id={dataset_id}"
            self.logger.info(f"Starting scrape for dataset ID {dataset_id}: {url}")
            
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(1, 3))
            
            browser.get(url)
            wait = WebDriverWait(browser, 15)
            
            # Wait for page to load completely
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            stats = {}
            
            # Robust selectors with multiple fallbacks
            selector_mappings = {
                'status': [
                    'span[aria-label="status"]',
                    'span[title="status"]',
                    '.status-indicator',
                    '[data-testid="status"]'
                ],
                'downloads': [
                    'span[aria-label="downloads"]',
                    'span[title="downloads"]',
                    '.download-count',
                    '[data-testid="downloads"]'
                ],
                'likes': [
                    'span[aria-label="likes"]',
                    'span[title="likes"]',
                    '.like-count',
                    '[data-testid="likes"]'
                ],
                'issues': [
                    'span[aria-label="issues"]',
                    'span[title="issues"]',
                    '.issue-count',
                    '[data-testid="issues"]'
                ]
            }
            
            for stat in ['status', 'downloads', 'likes', 'issues']:
                stats[stat] = "N/A"
                
                for selector in selector_mappings[stat]:
                    try:
                        element = wait.until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        if element and element.text.strip():
                            stats[stat] = element.text.strip()
                            self.logger.debug(f"Extracted {stat} using selector '{selector}': {stats[stat]}")
                            break
                    except TimeoutException:
                        continue
                    except Exception as e:
                        self.logger.debug(f"Error with selector '{selector}' for {stat}: {str(e)}")
                        continue
                
                if stats[stat] == "N/A":
                    self.logger.warning(f"Could not extract '{stat}' for dataset {dataset_id}")
            
            # Extract numbers from text
            def extract_number(stat_string):
                if stat_string == "N/A":
                    return 0
                # Handle various number formats (1K, 1M, 1.5K, etc.)
                number_match = re.search(r'(\d+(?:\.\d+)?)\s*([KMB]?)', stat_string.upper())
                if number_match:
                    num = float(number_match.group(1))
                    multiplier = number_match.group(2)
                    if multiplier == 'K':
                        return int(num * 1000)
                    elif multiplier == 'M':
                        return int(num * 1000000)
                    elif multiplier == 'B':
                        return int(num * 1000000000)
                    else:
                        return int(num)
                return 0
            
            stats['downloads'] = extract_number(stats['downloads'])
            stats['likes'] = extract_number(stats['likes'])
            stats['issues'] = extract_number(stats['issues'])
            
            self.logger.info(f"Completed scraping stats for dataset {dataset_id}: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error scraping stats for dataset {dataset_id}: {str(e)}")
            raise
        
        finally:
            if browser:
                self.browser_pool.return_browser(browser)

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
        
        # Initialize browser pool with appropriate size
        self._init_browser_pool(max_browsers=min(threads, 4))
        
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
        finally:
            # Clean up browser pool
            if self.browser_pool:
                self.browser_pool.close_all()
                self.browser_pool = None

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
            scraped_stats = self._scrape_openml_stats_with_retry(dataset_id, max_retries=self.max_retries)
            
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

        # Limit threads for scraping to avoid overwhelming the server
        scraping_threads = min(threads, 2)
        
        with ThreadPoolExecutor(max_workers=scraping_threads) as executor:
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
    
    def __del__(self):
        """Cleanup browser pool when object is destroyed"""
        if hasattr(self, 'browser_pool') and self.browser_pool:
            self.browser_pool.close_all()