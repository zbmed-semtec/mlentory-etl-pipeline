import pandas as pd
import torch
from typing import Any, Dict, List, Set, Union, Optional
from datetime import datetime
from tqdm import tqdm
import math
import os

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile, RepoFolder

from mlentory_extract.core.QAMatchingEngine import QAMatchingEngine


class ModelCardToSchemaParser:
    """
    A parser for mapping model card information directly to FAIR4ML schema properties.
    
    This class provides functionality to:
    - Extract structured information from model cards
    - Map HuggingFace metadata to FAIR4ML schema properties
    - Process tags and metadata
    - Handle semantic matching for extracting specific information
    
    Attributes:
        device: GPU device ID if available, None otherwise
        tags_language (set): Set of supported language tags
        tags_libraries (set): Set of supported ML library tags
        tags_other (set): Set of miscellaneous tags
        tags_task (set): Set of supported ML task tags
        hf_api: HuggingFace API client
        matching_engine: Engine for semantic matching
        schema_properties (dict): Dictionary of schema properties and their metadata
    """
    
    def __init__(
        self,
        matching_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        tags_language: List[str] = None,
        tags_libraries: List[str] = None,
        tags_other: List[str] = None,
        tags_task: List[str] = None,
        schema_file: str = "data/configuration/hf/transform/FAIR4ML_schema.tsv",
    ) -> None:
        """
        Initialize the Model Card to Schema Parser
        
        Args:
            matching_model (str): Model to use for semantic matching
            tags_language (List[str], optional): List of language tags. Defaults to None.
            tags_libraries (List[str], optional): List of library tags. Defaults to None.
            tags_other (List[str], optional): List of other tags. Defaults to None.
            tags_task (List[str], optional): List of task tags. Defaults to None.
            schema_file (str, optional): Path to the FAIR4ML schema file. 
                Defaults to "data/configuration/hf/transform/FAIR4ML_schema.tsv".
        """
        # Check for GPU availability
        try:
            import torch
            
            if torch.cuda.is_available():
                self.device = 0
                print("\nUSING GPU\n")
            else:
                self.device = None
                print("\nNOT USING GPU\n")
        except ModuleNotFoundError:
            # If torch is not available, assume no GPU
            self.device = None
            
        # Store configuration data
        self.tags_language = set(tag.lower() for tag in tags_language) if tags_language else set()
        self.tags_libraries = set(tag.lower() for tag in tags_libraries) if tags_libraries else set()
        self.tags_other = set(tag.lower() for tag in tags_other) if tags_other else set()
        self.tags_task = set(tag.lower() for tag in tags_task) if tags_task else set()
        
        # Initializing HF API
        self.hf_api = HfApi()
        
        # Initialize matching engine for semantic matching
        self.matching_engine = QAMatchingEngine(matching_model)
        
        # Load schema properties from file
        self.schema_properties = self.load_schema_properties(schema_file)
        
        # Initialize list to store processed properties
        self.processed_properties = []
    
    def load_schema_properties(self, schema_file: str) -> Dict[str, Dict[str, str]]:
        """
        Load schema properties from the FAIR4ML schema file.
        
        Args:
            schema_file (str): Path to the schema file
            
        Returns:
            Dict[str, Dict[str, str]]: Dictionary of schema properties and their metadata
        """
        try:
            # Read the schema file
            schema_df = pd.read_csv(schema_file, sep="\t")
            
            # Initialize properties dictionary with basic properties we always want
            properties = {}
            
            # Process each row in the schema file
            for _, row in schema_df.iterrows():
                
                properties[row["Property"]] = {
                    "source": row["Source"],
                    "range": row["Range"],
                    "description": row["Description"]
                }
            
            return properties
            
        except Exception as e:
            print(f"Error: Could not load schema file {schema_file}. Using default properties. Error: {str(e)}")
            # throw error
            raise e
    
    def add_default_extraction_info(
        self, data: Any, extraction_method: str, confidence: float
    ) -> Dict:
        """
        Create a standardized dictionary for extraction metadata.
        
        Args:
            data (Any): The extracted information
            extraction_method (str): Method used for extraction
            confidence (float): Confidence score of the extraction
            
        Returns:
            dict: Dictionary containing extraction metadata
        """
        return {
            "data": data,
            "extraction_method": extraction_method,
            "confidence": confidence,
            "extraction_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }
    
    def get_repository_weight_HF(self, model_name: str) -> str:
        """
        Calculate the total size of a HuggingFace model repository.
        
        Args:
            model_name (str): Name of the model repository
            
        Returns:
            str: Size of the repository in gigabytes
        """
        try:
            model_repo_weight = 0
            model_tree_file_information = self.hf_api.list_repo_tree(
                f"{model_name}", recursive=True
            )
            for x in list(model_tree_file_information):
                if isinstance(x, RepoFile):
                    # The weight of each file is in Bytes.
                    model_repo_weight += x.size
            return f"{model_repo_weight/(math.pow(10,9)):.3f} Gbytes"
        except:
            print(f"Error: Could not calculate repository weight for {model_name}")
            return "Not available"
    
    def parse_known_fields_HF(self, HF_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse known fields from HuggingFace dataset into FAIR4ML schema properties.
        
        Args:
            HF_df (pd.DataFrame): DataFrame containing HuggingFace model information
            
        Returns:
            pd.DataFrame: DataFrame with parsed fields mapped to FAIR4ML schema
        """
        
        # Map known fields directly
        HF_df.loc[:, "schema.org:author"] = HF_df.loc[:, "author"]
        HF_df.loc[:, "schema.org:dateCreated"] = HF_df.loc[:, "createdAt"].apply(lambda x: str(x))
        HF_df.loc[:, "schema.org:dateModified"] = HF_df.loc[:, "last_modified"].apply(lambda x: str(x))
        HF_df.loc[:, "schema.org:releaseNotes"] = HF_df.loc[:, "card"]
        HF_df.loc[:, "schema.org:description"] = HF_df.loc[:, "card"]
        HF_df.loc[:, "schema.org:name"] = HF_df.loc[:, "modelId"].apply(lambda x: x.split("/")[-1] if "/" in x else x)
        
        # Generate URLs for models
        HF_df.loc[:, "schema.org:identifier"] = HF_df.loc[:, "modelId"].apply(
            lambda x: f"https://huggingface.co/{x}" if x else None
        )
        HF_df.loc[:, "schema.org:url"] = HF_df.loc[:, "modelId"].apply(
            lambda x: f"https://huggingface.co/{x}" if x else None
        )
        HF_df.loc[:, "schema.org:discussionUrl"] = HF_df.loc[:, "modelId"].apply(
            lambda x: f"https://huggingface.co/{x}/discussions" if x else None
        )
        HF_df.loc[:, "codemeta:issueTracker"] = HF_df.loc[:, "modelId"].apply(
            lambda x: f"https://huggingface.co/{x}/discussions" if x else None
        )
        HF_df.loc[:, "schema.org:readme"] = HF_df.loc[:, "modelId"].apply(
            lambda x: f"https://huggingface.co/{x}/blob/main/README.md" if x else None
        )
        
        # Process repository weights
        for index, row in tqdm(
            HF_df.iterrows(), total=len(HF_df), desc="Processing repository weights"
        ):
            HF_df.loc[index, "schema.org:memoryRequirements"] = self.get_repository_weight_HF(
                HF_df.loc[index, "schema.org:identifier"]
            )
        
        # Add extraction metadata to each field
        properties = [
            "schema.org:identifier", 
            "schema.org:author", 
            "schema.org:dateCreated", 
            "schema.org:dateModified",
            "schema.org:releaseNotes",
            "schema.org:description",
            "schema.org:name", 
            "schema.org:url",
            "codemeta:issueTracker",
            "schema:memoryRequirements"
        ]
        
        self.processed_properties.extend(properties)
        
        HF_df = self.add_extraction_metadata_to_fields(
            df=HF_df,
            properties=properties,
            extraction_method="Parsed_from_HF_dataset",
            confidence=1.0,
            description="Adding extraction metadata"
        )
        
        return HF_df
        
    def parse_fields_from_tags_HF(self, HF_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract information from HuggingFace model tags and map to FAIR4ML schema properties.
        
        Args:
            HF_df (pd.DataFrame): DataFrame containing HuggingFace model information
            
        Returns:
            pd.DataFrame: DataFrame with parsed tag information mapped to schema
        """
        
        for index, row in tqdm(HF_df.iterrows(), total=len(HF_df), desc="Parsing tags"):
            # Initialize lists for collecting tag-based information
            ml_tasks = []
            datasets = []
            arxiv_ids = []
            licenses = []
            languages = []
            libraries = []
            keywords = []
            
            # Process each tag
            for tag in row["tags"]:
                # Convert tag to lowercase for consistent matching
                tag_lower = tag.lower()
                
                # Extract ML tasks (fair4ml:mlTask)
                tag_for_task = tag.replace("-", " ").lower()
                if tag_for_task in self.tags_task:
                    ml_tasks.append(tag_for_task)
                
                # Extract datasets (fair4ml:trainedOn, fair4ml:evaluatedOn)
                if "dataset:" in tag:
                    dataset_name = tag.replace("dataset:", "")
                    datasets.append(dataset_name)
                
                # Extract arxiv IDs (citation)
                if "arxiv:" in tag:
                    arxiv_id = tag.replace("arxiv:", "")
                    arxiv_ids.append(f"https://arxiv.org/abs/{arxiv_id}")
                
                # Extract license information (license)
                if "license:" in tag:
                    license_name = tag.replace("license:", "")
                    licenses.append(license_name)
                
                # Extract languages (inLanguage)
                if tag_lower in self.tags_language:
                    languages.append(tag)
                
                # Extract libraries (keywords)
                if tag_lower in self.tags_libraries:
                    libraries.append(tag_lower)
                
                # Collect all tags as keywords
                keywords.append(tag)
            
            # Add pipeline tag to ML tasks if available
            if row["pipeline_tag"] is not None:
                pipeline_task = row["pipeline_tag"].replace("-", " ").lower()
                if pipeline_task not in ml_tasks:
                    ml_tasks.append(pipeline_task)
            
            # Assign collected information to schema properties
            HF_df.at[index, "fair4ml:mlTask"] = ml_tasks
            
            HF_df.at[index, "fair4ml:trainedOn"] = datasets
            HF_df.at[index, "fair4ml:evaluatedOn"] = datasets
            HF_df.at[index, "fair4ml:testedOn"] = datasets
            
            HF_df.at[index, "schema.org:license"] = licenses
            
            HF_df.at[index, "schema.org:inLanguage"] = languages
            
            all_keywords = keywords + libraries
            HF_df.at[index, "schema.org:keywords"] = all_keywords
        
        # Add extraction metadata
        properties = [
            "fair4ml:mlTask",
            "fair4ml:trainedOn",
            "fair4ml:evaluatedOn",
            "fair4ml:testedOn",
            "schema.org:license",
            "schema.org:inLanguage",
            "schema.org:keywords"
        ]
        
        self.processed_properties.extend(properties)
        
        HF_df = self.add_extraction_metadata_to_fields(
            df=HF_df,
            properties=properties,
            extraction_method="Parsed_from_HF_tags",
            confidence=1.0,
            description="Adding tag extraction metadata"
        )
        
        return HF_df
    
    def parse_fields_from_text_HF(self, HF_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract information from model card text using semantic matching and map to FAIR4ML schema.
        
        This method uses semantic matching to find relevant sections in the model card
        that correspond to specific FAIR4ML schema properties. It automatically generates
        questions based on the property descriptions from the schema.
        
        Args:
            HF_df (pd.DataFrame): DataFrame containing HuggingFace model information
            
        Returns:
            pd.DataFrame: DataFrame with extracted information from text fields
        """
        # Generate queries for each schema property based on their descriptions
        context_queries = {}
        
        # Generate a question for each property based on its description
        for prop, metadata in self.schema_properties.items():
            # Skip properties that are not suitable for text extraction
            if prop in self.processed_properties:
                continue
                
            description = metadata.get("description", "")
            if description:
                # Format the property name for better readability
                readable_prop = prop.replace("fair4ml:", "").replace("codemeta:", "").replace("schema.org:", "")
                readable_prop = readable_prop.replace("_", " ").replace(":", " ")
                # Put a space between each uppercase letter: ConditionsOfUse -> Conditions of Use
                temp_readable_prop = ""
                for i in range(len(readable_prop)):
                    if readable_prop[i].isupper() and i != 0 and readable_prop[i-1].islower():
                        temp_readable_prop += " " + readable_prop[i]
                    else:
                        temp_readable_prop += readable_prop[i]
                        
                readable_prop = temp_readable_prop
                
                question = f"What is the {readable_prop} of this model? ({description})"
                
                context_queries[prop] = question
        
        
        # Pre-process all contexts at once
        contexts = []
        for _, row in tqdm(
            HF_df.iterrows(), total=len(HF_df), desc="Pre-processing contexts"
        ):
            context = row["card"]
            if not context or context.strip() == "":
                contexts.append(None)
                print(f"Skipping row {_} because context is empty")
                continue

            contexts.append(context)
        
        
        # Process in batches
        batch_size = 16
        for batch_start in tqdm(
            range(0, len(contexts), batch_size), desc="Processing text batches"
        ):
            batch_end = min(batch_start + batch_size, len(contexts))
            batch_contexts = contexts[batch_start:batch_end]
            
            # Process batch of contexts
            batch_results = []
            for context in batch_contexts:
                if context is None:
                    # Create empty results for all queries
                    empty_results = [(None, 0.0)] * len(contexts)
                    batch_results.append(empty_results)
                else:
                    # Find relevant sections for all queries at once
                    relevant_sections = self.matching_engine.find_relevant_sections(
                        questions=list(context_queries.values()), context=context, top_k=1
                    )
                    batch_results.append(relevant_sections)
            
            # Update DataFrame with batch results
            for i, relevant_sections in enumerate(batch_results):
                df_idx = batch_start + i
                
                # Process each query
                for q_idx, prop in enumerate(context_queries.keys()):
                    if relevant_sections[q_idx][0] is None:
                        # Handle empty/None context
                        print(f"Skipping row {df_idx} because context is empty")
                        continue
                    else:
                        section, score = relevant_sections[q_idx][0]
                        # Only use results with reasonable confidence
                        if score > 0.01:
                            HF_df.at[df_idx, prop] = [
                                {
                                    "data": section.content.strip(),
                                    "extraction_method": f"Semantic Matching with {self.matching_engine.model_name}",
                                    "confidence": score,
                                    "extraction_time": datetime.now().strftime(
                                        "%Y-%m-%d_%H-%M-%S"
                                    ),
                                }
                            ]
            
            # Clear some memory
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        
        return HF_df
    
    def process_dataframe(self, HF_df: pd.DataFrame, clean_columns: bool = True) -> pd.DataFrame:
        """
        Process a HuggingFace DataFrame by applying all parsing methods.
        
        This method combines all parsing methods to extract information from
        HuggingFace model metadata and map it to FAIR4ML schema properties.
        
        Args:
            HF_df (pd.DataFrame): DataFrame containing HuggingFace model information
            clean_columns (bool, optional): Whether to remove original HF columns.
                Defaults to True.
                
        Returns:
            pd.DataFrame: Processed DataFrame with FAIR4ML schema properties
        """
        #Initialize columns that will be populated
        schema_columns = list(self.schema_properties.keys())
        
        # Create columns if they don't exist
        for col in schema_columns:
            if col not in HF_df.columns:
                HF_df[col] = None
        
        # Apply parsing methods in sequence
        HF_df = self.parse_known_fields_HF(HF_df)
        HF_df = self.parse_fields_from_tags_HF(HF_df)
        HF_df = self.parse_fields_from_text_HF(HF_df)
        
        # Clean up columns if requested
        if clean_columns:
            # List of original HF columns to remove
            columns_to_remove = [
                "modelId",
                "author",
                "last_modified",
                "downloads",
                "likes",
                "pipeline_tag",
                "tags",
                "library_name",
                "createdAt",
                "card",
            ]
            
            # Drop columns that exist in the DataFrame
            columns_to_remove = [col for col in columns_to_remove if col in HF_df.columns]
            if columns_to_remove:
                HF_df = HF_df.drop(columns=columns_to_remove)
        
        # Drop any rows with NaN values
        HF_df = HF_df.dropna(how='all')
        
        return HF_df
    
    def add_extraction_metadata_to_fields(
        self, 
        df: pd.DataFrame, 
        properties: List[str], 
        extraction_method: str, 
        confidence: float = 1.0,
        description: str = "Adding extraction metadata"
    ) -> pd.DataFrame:
        """
        Add extraction metadata to multiple DataFrame fields at once.
        
        This utility method adds standardized extraction metadata to specified fields
        in the DataFrame, reducing code duplication across the parser.
        
        Args:
            df (pd.DataFrame): DataFrame to add metadata to
            properties (List[str]): List of property names to add metadata to
            extraction_method (str): Method used for extraction
            confidence (float, optional): Confidence score of the extraction. Defaults to 1.0.
            description (str, optional): Description for the progress bar. Defaults to "Adding extraction metadata".
            
        Returns:
            pd.DataFrame: DataFrame with added extraction metadata
        """
        # Ensure all properties exist in the DataFrame
        for prop in properties:
            if prop not in df.columns:
                df[prop] = None
                
        for index in tqdm(df.index, desc=description):
            for prop in properties:
                if prop in df.columns and df.at[index, prop] is not None:
                    df.at[index, prop] = [
                        self.add_default_extraction_info(
                            df.at[index, prop], extraction_method, confidence
                        )
                    ]
        
        return df
    
    def print_detailed_dataframe(self, HF_df: pd.DataFrame):
        """
        Print detailed information about the DataFrame.
        
        Args:
            HF_df (pd.DataFrame): DataFrame to print information about
        """
        print("\n**DATAFRAME**")
        print("\nColumns:", HF_df.columns.tolist())
        print("\nShape:", HF_df.shape)
        print("\nSample Data:")
        for col in HF_df.columns:
            print(f"\n{col}:")
            for row in HF_df[col].head(3):  # Show only first 3 rows for each column
                # Limit the text to 100 characters
                if isinstance(row, list):
                    row_data = row[0]["data"]
                    if isinstance(row_data, str):
                        print(row_data[:100])
                    else:
                        print(row_data)
                else:
                    print(row)

            print()
        print("\nDataFrame Info:")
        print(HF_df.info())
