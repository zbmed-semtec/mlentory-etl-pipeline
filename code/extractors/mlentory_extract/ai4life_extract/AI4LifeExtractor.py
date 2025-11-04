import os
import csv
import json
import requests
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

class AI4LifeExtractor:
    """Extractor for fetching and saving model metadata from the AI4Life platform."""

    def __init__(self, schema_file: str, base_url: str = "https://hypha.aicell.io", parent_id: str = "bioimage-io/bioimage.io"):
        """Initialize the AI4LifeExtractor with base URL and parent ID.

        Args:
            schema_file (str): Path to TSV file for model metadata schema.
            base_url (str): The base URL for the AI4Life API. Defaults to "https://hypha.aicell.io".
            parent_id (str): The parent ID for fetching records. Defaults to "bioimage-io/bioimage.io".
        """
        self.schema = self._load_schema(schema_file)
        self.base_url = base_url
        self.parent_id = parent_id
        self.extraction_timestamp = None

    def _load_schema(self, schema_file: str) -> Dict[str, List[str]]:
        """Load the metadata schema from a TSV file.

        Args:
            schema_file (str): Path to the TSV file.

        Returns:
            Dict[str, List[str]]: Mapping of output keys to lists of paths.

        Raises:
            FileNotFoundError: If the schema file is not found.
        """
        schema_file_path = Path(schema_file)
        if not schema_file_path.exists():
            raise FileNotFoundError(f"Schema file '{schema_file}' not found.")

        mapping = {}
        with schema_file_path.open(encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # Skip headers
            for out_key, paths in reader:
                mapping[out_key] = [p.strip() for p in paths.split(',') if p.strip()]
        return mapping

    def _flatten_dict(self, nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten a nested dictionary into a single-level dictionary with dot-separated keys.

        Args:
            nested_dict (Dict[str, Any]): The nested dictionary to flatten.
            parent_key (str): The parent key for constructing dot-separated paths.
            sep (str): Separator for keys. Defaults to '.'.

        Returns:
            Dict[str, Any]: Flattened dictionary.
        """
        items = []
        for key, value in nested_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, sep).items())
            else:
                items.append((new_key, value))
        return dict(items)

    def _wrap_metadata(self, value: Any, method: str = "hypha_api") -> List[Dict[str, Any]]:
        """Wrap metadata value with extraction details.

        Args:
            value (Any): The metadata value to wrap.
            method (str): The extraction method. Defaults to "hypha_api".

        Returns:
            List[Dict[str, Any]]: Wrapped metadata with extraction details.
        """
        return [{
            "data": value,
            "extraction_method": method,
            "confidence": 1,
            "extraction_time": self.extraction_timestamp
        }]

    def fetch_records(self, num_models: int) -> Dict[str, Any]:
        """Fetch model records from the AI4Life API.

        Args:
            num_models (int): Number of model records to fetch.

        Returns:
            Dict[str, Any]: Fetched records.

        Raises:
            requests.exceptions.RequestException: If the API request fails.
            ValueError: If the API response is invalid JSON.
        """
        try:
            response = requests.get(
                f"{self.base_url}/public/services/artifact-manager/list",
                params={"parent_id": self.parent_id, "limit": num_models},
                timeout=10
            )
            self.extraction_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Failed to fetch records: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid JSON response from API: {e}")

    def _save_json(self, data: Any, output_path: str) -> None:
        """Save data to a JSON file.

        Args:
            data (Any): Data to save.
            output_path (str): Path to the output JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def save_records(self, records: Dict[str, Any], output_dir: str) -> str:
        """Save model records to a JSON file with a timestamped filename.

        Args:
            records (Dict[str, Any]): The records to save.
            output_dir (str): The directory to save the JSON file.

        Returns:
            str: Path to the saved file.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = Path(output_dir) / f'metadata_records_{timestamp}.json'
        self._save_json(records, filename)
        return str(filename)

    def _map_model_metadata(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Map a single model's metadata to the schema.
        
        Args:
            model: The model data to map.
        
        Returns:
            Mapped model metadata following the defined schema.
        """
        flat = self._flatten_dict(model)
        mapped = {out_key: None for out_key in self.schema}

        # Map simple fields
        for out_key, paths in self.schema.items():
            if not paths:
                continue
                
            values = [flat[p] for p in paths if p in flat]
            if values:
                mapped[out_key] = values[0] if len(values) == 1 else values
        
        # Handle special cases
        if isinstance(mapped.get("schema.org:identifier"), list):
            mapped["schema.org:identifier"] = " ".join(
                str(x) for x in mapped["schema.org:identifier"] if x is not None
            )
        
        mapped["schema.org:identifier"] = mapped["schema.org:identifier"].split(" ")[0]
        
        # Handle additional urls
        ai4life_url = f"https://bioimage.io/#/artifacts/{mapped['schema.org:identifier']}"
        mapped["schema.org:url"] = [mapped["schema.org:url"], ai4life_url]
        mapped["schema.org:archivedAt"] = [mapped["schema.org:archivedAt"], ai4life_url]
        
        #Handle licenses
        mapped["schema.org:license"] = mapped["schema.org:license"].split("/")[-1]
        
        # Process dates
        for date_field in ["schema.org:dateCreated", "schema.org:dateModified"]:
            if date_field in mapped and mapped[date_field] is not None:
                mapped[date_field] = datetime.utcfromtimestamp(
                    mapped[date_field]
                ).strftime('%Y-%m-%d')
        
        # Process contributor fields (authors and maintainers)
        for contributor_field in ["schema.org:author", "schema.org:maintainer"]:
            contributors = mapped.get(contributor_field, []) or []
            transformed = []
            
            for contributor in contributors:
                name = contributor.get('name', '')
                orcid = contributor.get('orcid', '')
                github_user = contributor.get('github_user', '')
                
                url = (
                    f"https://orcid.org/{orcid}" if orcid else
                    f"https://github.com/{github_user}" if github_user else
                    ""
                )
                
                transformed.append({'name': name, 'url': url})
            mapped[contributor_field] = transformed
        
        # Handle special case for sharedBy
        shared_by = mapped.get("fair4ml:sharedBy")
        mapped["fair4ml:sharedBy"] = shared_by[0] if shared_by else ""
        
        # Handle special case for version
        version = mapped.get("schema.org:version")
        mapped["schema.org:version"] = version[-1]["version"]
        
        return mapped
    
    def _wrap_mapped_models(self, mapped_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Wrap mapped models with metadata details.

        Args:
            mapped_models (List[Dict[str, Any]]): List of mapped model metadata.

        Returns:
            List[Dict[str, Any]]: Wrapped model metadata.
        """
        return [{k: self._wrap_metadata(v) for k, v in model.items()} for model in mapped_models]

    def _group_records_by_type(self, data: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
        """Group records by type into known and unknown categories.

        Args:
            data (List[Dict[str, Any]]): List of records to group.

        Returns:
            Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]: Known and unknown type records.
        """
        known_types = {'application', 'model', 'dataset'}
        known = defaultdict(list)
        unknown = defaultdict(list)

        for entry in data:
            entry_type = (entry.get('type') or '').strip().lower()
            if entry_type in known_types:
                known[entry_type].append(entry)
            else:
                unknown[entry_type].append(entry)

        return known, unknown

    def _save_grouped_records(self, known: Dict[str, List[Dict[str, Any]]], unknown: Dict[str, List[Dict[str, Any]]], output_dir: str) -> Dict[str, str]:
        """Save grouped records to JSON files.

        Args:
            known (Dict[str, List[Dict[str, Any]]]): Known type records.
            unknown (Dict[str, List[Dict[str, Any]]]): Unknown type records.
            output_dir (str): Directory to save the files.

        Returns:
            Dict[str, str]: Mapping of entity types to output file paths.
        """
        output_files = {
            'model': Path(output_dir) / f'models_metadata_{self.extraction_timestamp}.json',
            'dataset': Path(output_dir) / f'datasets_metadata_{self.extraction_timestamp}.json',
            'application': Path(output_dir) / f'applications_metadata_{self.extraction_timestamp}.json'
        }

        # Save known types
        for entity_type, file_path in output_files.items():
            self._save_json(known.get(entity_type, []), file_path)

        # Save unknown types
        self._save_json(unknown, Path(output_dir) / 'unknown_types.json')

        return {k: str(v) for k, v in output_files.items()}

    def ai4life_extractor(self, output_dir: str, input_filename: str) -> Dict[str, str]:
        """Extract and group records from a JSON file, saving them by type.

        Args:
            output_dir (str): Directory to save the output files.
            input_filename (str): Path to the input JSON file.

        Returns:
            Dict[str, str]: Mapping of entity types to output file paths.
        """
        # Load JSON data
        with Path(input_filename).open('r', encoding='utf-8') as f:
            data = json.load(f)

        # Group records by type
        known, unknown = self._group_records_by_type(data)

        # Save grouped records
        return self._save_grouped_records(known, unknown, output_dir)
    
    def format_citation(self, citation: List[Dict[str, Any]]) -> str:
        """
        Formats a list of citation dictionaries into a semicolon-separated string with clickable DOIs/URLs.
        Args:
            citation: List of citation dictionaries, each containing 'text', 'doi', and/or 'url'.
        Returns:
            Formatted string with text and clickable DOI/URL.
        """
        formatted = []
        for item in citation:
            text = item.get('text', '')
            doi = item.get('doi', '')
            url = item.get('url', '')
            # Use DOI if present, otherwise URL
            ref = doi if doi else url
            if ref:
                formatted.append(f"{text} ({ref})")
            else:
                formatted.append(text)
        return '; '.join(formatted)
        
    def dict_to_dataframe(self, extracted_metadata:Dict, entity) -> pd.DataFrame:
        """
        Converts a Dict to a pandas DataFrame.
        
        Args:
            extracted_metadata: metadata dictionary to be converted to dataframe.
            entity: entity to extract and process from the extracted metadata
            
        Returns:
            pandas DataFrame containing the flattened data.
        """
    
        # Extract the 'entity' list
        extracted_entity = extracted_metadata.get(entity, [])
        
        # Create DataFrame
        df = pd.DataFrame(extracted_entity)
        
        # Identify columns that may contain URLs (only strings starting with 'https')
        url_columns = []
        for col in df.columns:
            # Check if any non-null value is a string starting with 'https'
            has_urls = any(isinstance(val, str) and val.startswith('https') for val in df[col] if pd.notnull(val))
            if has_urls:
                url_columns.append(col)
        
        return df

    def download_modelfiles_with_additional_entities(self, num_models: int, output_dir: str = "./output", additional_entities: List[str] = ["dataset", "application"]) -> Dict[str, List[Dict[str, Any]]]:
        """Download model files and save their metadata to a JSON file.

        Args:
            num_models (int): Number of model records to fetch.
            output_dir (str): Directory to save the metadata file. Defaults to "./output".
            additional_entities (List[str]): Additional entity types to process. Defaults to ["dataset", "application"].

        Returns:
            Dict[str, List[Dict[str, Any]]]: Extracted metadata for each entity type.
        """
        # Fetch and save raw records
        records = self.fetch_records(num_models)
        input_filename = self.save_records(records, Path(output_dir) / "extraction")
        output_files = self.ai4life_extractor(Path(output_dir) / "extraction", input_filename)

        extracted_metadata = {}

        # Process models
        with Path(output_files["model"]).open(encoding='utf-8') as f:
            models = json.load(f)
        extracted_metadata["model"] = self._wrap_mapped_models([self._map_model_metadata(model) for model in models])

        # Process additional entities
        for entity in additional_entities:
            if entity in output_files:
                with Path(output_files[entity]).open(encoding='utf-8') as f:
                    data = json.load(f)
                extracted_metadata[entity] = self._wrap_mapped_models([self._map_model_metadata(item) for item in data])

        # Save extracted metadata
        output_filename = Path(output_dir) / f'extraction_metadata_{self.extraction_timestamp}.json'
        self._save_json(extracted_metadata, output_filename)

        extracted_metadata_df_with_additional_entities = {}
        all_entities = ["model"] + additional_entities 

        for entity in all_entities:
            #convert to dataframe from dictionary
            extracted_metadata_df = self.dict_to_dataframe(extracted_metadata, entity)
            extracted_metadata_df_with_additional_entities[entity] = extracted_metadata_df

        return extracted_metadata_df_with_additional_entities
    