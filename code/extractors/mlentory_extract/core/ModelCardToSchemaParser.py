import json
import pandas as pd
import torch
import yaml
import spdx_lookup
from typing import Any, Dict, List, Set, Tuple, Union, Optional
from datetime import datetime
from tqdm import tqdm
import math
import os
import logging
import pprint
import re

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile, RepoFolder

from mlentory_extract.core.QAMatchingEngine import QAMatchingEngine, RelevantSectionMatch
from mlentory_extract.core.QAInferenceEngine import QAInferenceEngine, QAResult
from mlentory_extract.core.SchemaPropertyExtractor import SchemaPropertyExtractor

# Define the structure for prepared grouped inputs
# {index: [{"questions": List[str], "properties": List[str], "context": str, "avg_score": float}, ...]}
PreparedGroupedInput = Dict[int, List[Dict[str, Any]]]
# Define the structure for results from grouped QA run
# {index: {property_name: QAResult}}
GroupedQAResults = Dict[int, Dict[str, QAResult]]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        qa_engine: Engine for extractive Question Answering
        schema_properties (dict): Dictionary of schema properties and their metadata
    """
    
    def __init__(
        self,
        qa_model_name: str = "Qwen/Qwen2.5-Coder-3B",
        matching_model_name: str = "Alibaba-NLP/gte-modernbert-base",
        tags_language: List[str] = None,
        tags_libraries: pd.DataFrame = None,
        tags_other: pd.DataFrame = None,
        tags_task: pd.DataFrame = None,
        schema_file: str = "data/configuration/hf/transform/FAIR4ML_schema.tsv",
    ) -> None:
        """
        Initialize the Model Card to Schema Parser
        
        Args:
            qa_model_name (str): Model name for the QAInferenceEngine.
                Defaults to "Qwen/Qwen2.5-Coder-3B".
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
                logger.info("\nUSING GPU\n")
            else:
                self.device = None
                logger.info("\nNOT USING GPU\n")
        except ModuleNotFoundError:
            # If torch is not available, assume no GPU
            self.device = None
            logger.info("\nNOT USING GPU\n")
            
        # Store configuration data
        self.tags_language = set(tag.lower() for tag in tags_language) if tags_language else set()
        self.tags_libraries_names = set(tag.lower() for tag in tags_libraries["tag_name"].values.tolist())
        self.tags_other_names = set(tag.lower() for tag in tags_other["tag_name"].values.tolist())
        self.tags_task_names = set(tag.lower() for tag in tags_task["tag_name"].values.tolist())
        
        self.tags_libraries_df = tags_libraries
        self.tags_other_df = tags_other
        self.tags_task_df = tags_task
        
        # Initializing HF API
        self.hf_api = HfApi()
        
        # Initialize matching engine for semantic matching
        self.matching_engine = QAMatchingEngine(matching_model_name)
        
        # Initialize QA engine for extractive QA
        # TODO: Uncomment this when the QA engine is ready
        # self.qa_engine = QAInferenceEngine(model_name=qa_model_name, batch_size=4)
        self.qa_engine = None
        
        # Load schema properties from file
        self.schema_properties = self.load_schema_properties(schema_file)
        
        # Initialize list to store processed properties
        self.processed_properties = []
        
        # Initialize schema property extractor
        self.schema_property_extractor = SchemaPropertyExtractor(
            qa_matching_engine=self.matching_engine,
            qa_inference_engine=self.qa_engine,
            schema_properties=self.schema_properties
        )
    
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
                    "description": row["Description"],
                    "HF_Readme_Section": row["HF_Readme_Section"]
                }
            
            return properties
            
        except Exception as e:
            logger.error(f"Error: Could not load schema file {schema_file}. Using default properties. Error: {str(e)}")
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
        # if type(data) != list:
        #     data = [data]
            
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
            logger.error(f"Error: Could not calculate repository weight for {model_name}")
            return "Information not found"
    
    def parse_known_fields_HF(self, HF_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse known fields from HuggingFace dataset into FAIR4ML schema properties.
        
        Args:
            HF_df (pd.DataFrame): DataFrame containing HuggingFace model information
            
        Returns:
            pd.DataFrame: DataFrame with parsed fields mapped to FAIR4ML schema
        """
        HF_df.loc[:, "fair4ml:sharedBy"] = HF_df.loc[:, "author"]
        
        # Format dates properly to ensure ISO format
        HF_df.loc[:, "schema.org:dateCreated"] = HF_df.loc[:, "createdAt"].apply(
            lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x)
        )
        HF_df.loc[:, "schema.org:datePublished"] = HF_df.loc[:, "createdAt"].apply(
            lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x)
        )
        HF_df.loc[:, "schema.org:dateModified"] = HF_df.loc[:, "last_modified"].apply(
            lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x)
        )
        
        HF_df.loc[:, "schema.org:description"] = HF_df.loc[:, "card"].apply(
            lambda x: re.sub(r'---.*?---', '', x, count=1, flags=re.DOTALL) if isinstance(x, str) else x
        )
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
        HF_df.loc[:, "schema.org:archivedAt"] = HF_df.loc[:, "modelId"].apply(
            lambda x: f"https://huggingface.co/{x}" if x else None
        )
        HF_df.loc[:, "codemeta:readme"] = HF_df.loc[:, "modelId"].apply(
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
            "fair4ml:sharedBy", 
            "schema.org:dateCreated", 
            "schema.org:dateModified",
            "schema.org:datePublished",
            "schema.org:description",
            "schema.org:name", 
            "schema.org:url",
            "schema.org:discussionUrl",
            "codemeta:readme",
            "codemeta:issueTracker",
            "schema.org:archivedAt",
            "schema.org:memoryRequirements"
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
            ml_tasks = set()
            base_models = set()
            datasets = set()
            arxiv_ids = set()
            languages = set()
            libraries = set()
            keywords = set()
            
            # Process each tag
            for tag in row["tags"]:
                # Convert tag to lowercase for consistent matching
                tag_lower = tag.lower()
                
                
                # Extract datasets (fair4ml:trainedOn, fair4ml:evaluatedOn)
                if "dataset:" in tag:
                    dataset_name = tag.replace("dataset:", "")
                    datasets.add(dataset_name)
                    # keywords.add(tag)
                
                # Extract arxiv IDs (citation)
                if "arxiv:" in tag:
                    arxiv_id = tag.replace("arxiv:", "")
                    arxiv_ids.add(f"https://arxiv.org/abs/{arxiv_id}")
                    # keywords.add(tag)
                
                # Extract base models (fair4ml:baseModel)
                if "base_model:" in tag:
                    base_model = tag.split(":")[-1]
                    base_models.add(base_model)
                    # keywords.add(tag)
                
                # Extract languages (inLanguage)
                if tag_lower in self.tags_language:
                    languages.add(tag)
                
                # Extract libraries (keywords)
                if tag_lower in self.tags_libraries_names:
                    libraries.add(tag_lower)
                
                # Extract ML tasks (fair4ml:mlTask)
                tag_for_task = tag.replace("-", " ").lower()
                if tag_for_task in self.tags_task_names:
                    ml_tasks.add(tag_for_task)
                
                # Find a better way to ignore country tags
                if ":" not in tag_lower and tag_lower not in self.tags_language:
                    keywords.add(tag_lower)
                
            
            # Add pipeline tag to ML tasks if available
            if row["pipeline_tag"] is not None:
                pipeline_task = row["pipeline_tag"].replace("-", " ").lower()
                if pipeline_task not in ml_tasks:
                    ml_tasks.add(pipeline_task)
                    keywords.add(pipeline_task.lower())
            
            # Assign collected information to schema properties
            HF_df.at[index, "fair4ml:mlTask"] = list(ml_tasks)
            HF_df.at[index, "fair4ml:trainedOn"] = list(datasets)
            HF_df.at[index, "fair4ml:evaluatedOn"] = list(datasets)
            HF_df.at[index, "fair4ml:testedOn"] = list(datasets)
            HF_df.at[index, "fair4ml:fineTunedFrom"] = list(base_models)
            HF_df.at[index, "schema.org:inLanguage"] = list(languages)
            HF_df.at[index, "codemeta:referencePublication"] = list(arxiv_ids)
            HF_df.at[index, "schema.org:keywords"] = list(keywords)
        
        # Add extraction metadata
        properties = [
            "fair4ml:mlTask",
            "fair4ml:trainedOn",
            "fair4ml:evaluatedOn",
            "fair4ml:testedOn",
            "fair4ml:fineTunedFrom",
            "schema.org:inLanguage",
            "schema.org:keywords",
            "codemeta:referencePublication",
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
    
    def parse_fields_from_yaml_HF(self, HF_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract information from YAML section of model card and map to FAIR4ML schema properties.
        """
        for index, row in tqdm(HF_df.iterrows(), total=len(HF_df), desc="Parsing YAML"):
            card_text = row.get("card", "")
            yaml_dict = None
            
            if isinstance(card_text, str):
                # Regex to find the first YAML block (---...---)
                match = re.search(r"^---\s*$(.*?)^---\s*$", card_text, re.MULTILINE | re.DOTALL)
                if match:
                    yaml_text = match.group(1)
                    try:
                        yaml_dict = yaml.safe_load(yaml_text)
                    except yaml.YAMLError as e:
                        logger.error(f"Error parsing YAML for index {index}: {e}")
                        yaml_dict = {"error": f"YAML parsing error: {e}"}
                else:
                    # print(f"No YAML block found for index {index}")
                    yaml_dict = {"error": "No YAML block found"}
            else:
                yaml_dict = {"error": "Card text is not a string"}
                
            if (yaml_dict is None) or (type(yaml_dict) != dict) or ("error" in yaml_dict):
                continue
            
            # Check if the model is gated
            gated_info = self.get_model_gated_info(yaml_dict)
            
            if gated_info != "":
                HF_df.at[index, "schema.org:conditionsOfAccess"] = gated_info
            
            # Check if the model is licensed
            licensed_info = self.get_model_licensed_info(yaml_dict)
            
            if licensed_info != "":
                HF_df.at[index, "schema.org:license"] = licensed_info

        properties = [
            "schema.org:conditionsOfAccess",
            "schema.org:license"
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
    
    def get_model_gated_info(self, yaml_dict: Dict) -> bool:
        """
        Check if the model is gated based on the YAML dictionary.
        """
        gated_info = ""
        
        if isinstance(yaml_dict, dict):
            for key, value in yaml_dict.items():
                if "extra_gated" in key and isinstance(value, str):
                    gated_info += value + "\n"
            
        return gated_info
    
    def get_model_licensed_info(self, yaml_dict: Dict) -> bool:
        """
        Check if the model is licensed based on the YAML dictionary.
        """
        licensed_info = ""
        
        if "license_name" in yaml_dict:
            if isinstance(yaml_dict["license_name"], str):
                licensed_info = yaml_dict["license_name"]
            elif isinstance(yaml_dict["license_name"], list):
                licensed_info = yaml_dict["license_name"][0]
        elif "license" in yaml_dict:
            if isinstance(yaml_dict["license"], str):
                licensed_info = yaml_dict["license"]
            elif isinstance(yaml_dict["license"], list):
                licensed_info = yaml_dict["license"][0]
        
        # Check if the license is a SPDX license
        spdx_license_from_id = spdx_lookup.by_id(licensed_info)
        spdx_license_from_name = spdx_lookup.by_name(licensed_info)
        
        spdx_license = spdx_license_from_id or spdx_license_from_name
        
        if spdx_license:
            licensed_info = spdx_license.id
        else:
            # Put everything that has license in the key
            if isinstance(yaml_dict, dict):
                licensed_info+="\n"
                for key, value in yaml_dict.items():
                    if "license" in key:
                        if isinstance(value, str):
                            licensed_info += key + ": " + value + "\n"
                        elif isinstance(value, list):
                            for item in value:
                                licensed_info += key + ": " + item + "\n"
            
        return licensed_info
                
    def _prepare_qa_inputs(self, HF_df: pd.DataFrame, schema_property_contexts: Dict[str, str]) -> Tuple[List[Dict], Set[str]]:
        """Prepares inputs for the QA engine by finding relevant context sections and their scores."""
        qa_inputs_for_df = []
        properties_to_process = set()
        
        logger.info("Preparing QA inputs by matching questions to context sections...")
        for index, row in tqdm(
            HF_df.iterrows(), total=len(HF_df), desc="Matching questions to contexts"
        ):
            context = row.get("card", "") # Use .get for safety
            # logger.info(f"\n \n Context: {context} \n \n")
            if not context or not isinstance(context, str):
                continue # Skip if no valid context
            
            question_items = schema_property_contexts.items()
            questions = [question for question, _ in question_items]
            
            match_results_for_all_questions = self.matching_engine.find_relevant_sections(
                questions=questions,
                context=context,
                top_k=4
            )
            
            for question_item, individual_question_match_results in zip(question_items, match_results_for_all_questions):
                question, property = question_item
                # logger.info(f"\n \n Question: {question} \n \n")
                new_context = "\n".join([match.section.title + ": " + match.section.content for match in individual_question_match_results])
                # logger.info(f"\n\n New context for question:\n\n {new_context} \n \n")
                match_scores = [match.score for match in individual_question_match_results]
                
                qa_inputs_for_df.append(
                                {
                                    "index": index,
                                    "prop": property,
                                    "question": question,
                                    "context": new_context,
                                    "scores": match_scores,
                                }
                            )
                properties_to_process.add(property)

        return qa_inputs_for_df, properties_to_process

    def _run_batch_qa(self, qa_inputs_for_df: List[Dict]) -> List[QAResult]:
        """Runs batch inference using the QA engine."""
        logger.info(f"Performing batch QA inference with {self.qa_engine.model_name}...")
        questions = [item["question"] for item in qa_inputs_for_df]
        contexts = [item["context"] for item in qa_inputs_for_df]

        try:
            qa_results = self.qa_engine.batch_inference(questions, contexts)
            return qa_results
        except Exception as e:
            logger.error(f"Error during batch QA inference: {e}. Skipping QA population.")
            return [] # Return empty list on error

    def _populate_dataframe_with_qa_results(
        self,
        HF_df: pd.DataFrame,
        qa_inputs_for_df: List[Dict],
        qa_results: List[QAResult]
    ) -> pd.DataFrame:
        """Populates the DataFrame with QA results and calculated confidence."""
        logger.info("Populating DataFrame with QA results...")
        if len(qa_results) != len(qa_inputs_for_df):
            logger.info(f"Warning: Mismatch between QA inputs ({len(qa_inputs_for_df)}) and results ({len(qa_results)}). Results may be incomplete.")
            num_items_to_process = min(len(qa_inputs_for_df), len(qa_results))
        else:
            num_items_to_process = len(qa_inputs_for_df)

        for i in tqdm(range(num_items_to_process), desc="Updating DataFrame"):
            item = qa_inputs_for_df[i]
            result = qa_results[i]
            index = item["index"]
            property = item["prop"]
            match_scores = item["scores"]

            average_confidence = sum(match_scores) / len(match_scores) if match_scores else 0.0

            extraction_info = self.add_default_extraction_info(
                data=result.answer,
                extraction_method=f"GenQA (Context: {self.matching_engine.model_name}, Answer: {self.qa_engine.model_name})",
                confidence=average_confidence,
            )

            if property not in HF_df.columns:
                HF_df[property] = pd.Series([[] for _ in range(len(HF_df))], index=HF_df.index, dtype=object)

            current_val = HF_df.loc[index, property]
            if not isinstance(current_val, list):
                HF_df.loc[index, property] = [extraction_info]
            else:
                HF_df.loc[index, property] = [extraction_info]
                
        return HF_df
    
    def _populate_dataframe_with_context_matching_results(self, HF_df: pd.DataFrame, matching_results: List[List[RelevantSectionMatch]]) -> pd.DataFrame:
        """Populates the DataFrame with context matching results."""
        logger.info("Populating DataFrame with context matching results...")
        for index, row in tqdm(HF_df.iterrows(), total=len(HF_df), desc="Updating DataFrame"):
            for property_name, context in matching_results.items():
                matching_section = context[0]
                row[property_name] = self.add_default_extraction_info(
                    data=matching_section.section.content,
                    extraction_method=f"Context Matching (Context: {self.matching_engine.model_name})",
                    confidence=matching_section.score
                )
        
        return HF_df
                

    def create_schema_property_contexts(self) -> Dict[str, str]:
        """
        Build a text‐context for each schema property that you can use
        to match against a list of markdown sections.

        Returns:
            Dict[str, str]: 
                property_name → formatted context block
        """
        contexts: Dict[str, str] = {}

        for prop, meta in self.schema_properties.items():
            # skip ones we've already extracted
            if prop in self.processed_properties:
                continue

            description = meta.get("description", "").strip()
            raw_sections = meta.get("HF_Readme_Section", "").strip()

            # skip if no description
            if not description:
                continue

            # turn "Uses > Direct Use ; Uses > Downstream Use" into a Python list
            sections = [s.strip() for s in raw_sections.split(";") if s.strip()]

            # humanize the property name: "fair4ml:intendedUse" → "Intended Use"
            base = prop.split(":", 1)[-1]
            # insert spaces before internal caps, replace underscores
            human = re.sub(r"(?<=[a-z])([A-Z])", r" \1", base).replace("_", " ").title()

            # build the context block
            context = (
                f"Property: **{human}**\n"
                f"Description: {description}\n"
                f"Likely HF Sections: {', '.join(sections)}"
            )

            contexts[prop] = context

        return contexts
    
    def parse_fields_from_text_HF(self, HF_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract information from model card text using semantic matching and QA.

        Orchestrates the process of preparing inputs, running QA, and populating results.

        Args:
            HF_df (pd.DataFrame): DataFrame containing HuggingFace model information

        Returns:
            pd.DataFrame: DataFrame with extracted information from text fields using QA
        """
        # Generate queries for each schema property based on their descriptions
        schema_property_contexts = self.create_schema_property_contexts()
        logger.info(f"Schema property contexts: {schema_property_contexts}")
        if len(schema_property_contexts) == 0:
            logger.error("No properties identified for text extraction. Skipping.")
            return HF_df

        # 1. Prepare QA inputs
        qa_inputs_for_df, properties_to_process = self._prepare_qa_inputs(HF_df, schema_property_contexts)

        if not qa_inputs_for_df:
            logger.error("No valid question-context pairs found for QA. Skipping QA step.")
            return HF_df

        # 2. Perform batch QA inference
        qa_results = self._run_batch_qa(qa_inputs_for_df)

        # 3. Populate DataFrame with QA results if inference was successful
        if qa_results:
            HF_df = self._populate_dataframe_with_qa_results(HF_df, qa_inputs_for_df, qa_results)
            # Add processed properties to the list only if population occurred
            self.processed_properties.extend(list(properties_to_process))
            logger.info(f"Processed text fields via QA: {list(properties_to_process)}")
        else:
             logger.error("QA inference did not produce results. Skipping DataFrame population for QA.")

        # Clear GPU memory if possible
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

        return HF_df
    
    def parse_fields_from_text_by_context_matching(self, HF_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract information from model card text by finding the best-matching section
        for each schema property using semantic context matching.

        For each property needing extraction, this method identifies the single
        markdown section in the model card that best matches the property's
        description and known relevant HF Readme sections. The content of this
        best-matching section is directly used as the extracted value.

        Args:
            HF_df (pd.DataFrame): DataFrame containing HuggingFace model information,
                including a 'card' column with the model card text.

        Returns:
            pd.DataFrame: DataFrame with extracted information populated into
                columns corresponding to schema properties.
        """
        logging.info("Starting 'parse_fields_from_text_by_context_matching'")
        # 1. Create the context for each schema property
        schema_property_contexts = self.create_schema_property_contexts()
        
        # 2. Find the best-matching section for each schema property
        for index, row in tqdm(HF_df.iterrows(), total=len(HF_df), desc="Finding best-matching sections"):
            model_card_text = row.get("card", "")
            matching_results = self.matching_engine.find_relevant_sections(
                list(schema_property_contexts.values()),
                model_card_text,
                top_k=2
            )
            
            
            for property_name, result in zip(schema_property_contexts.keys(), matching_results):
                if property_name not in HF_df.columns:
                    logger.error(f"Warning: Property {property_name} not found in DataFrame.")
                    continue
                
                row[property_name] = self.add_default_extraction_info(
                    data=result[0].section.content,
                    extraction_method=f"Context Matching (Context: {self.matching_engine.model_name})",
                    confidence=result[0].score
                )
            
        return HF_df
    
    def _prepare_inputs_with_question_grouping(
        self, HF_df: pd.DataFrame, schema_property_questions: Dict[str, str], max_questions_per_group: int = 5
    ) -> Tuple[PreparedGroupedInput, Set[str]]:
        """
        Prepares inputs for grouped QA by clustering questions semantically within each context.

        Args:
            HF_df (pd.DataFrame): DataFrame containing model information.
            schema_property_questions (Dict[str, str]): Mapping from question string to property name.

        Returns:
            Tuple[PreparedGroupedInput, Set[str]]:
                - Dictionary mapping row index to a list of prepared groups.
                - Set of properties being processed.
        """
        prepared_data: PreparedGroupedInput = {}
        properties_to_process = set()
        input_questions = list(schema_property_questions.keys())

        logger.info("Preparing QA inputs with question grouping...")
        for index, row in tqdm(
            HF_df.iterrows(), total=len(HF_df), desc="Grouping questions and finding sections"
        ):
            context = row.get("card", "")
            if not context or not isinstance(context, str):
                continue

            # Find grouped relevant sections (questions grouped by similarity)
            try:
                grouped_sections_info = self.matching_engine.find_grouped_relevant_sections(
                    input_questions, context, top_k=4, max_questions_per_group=max_questions_per_group, max_section_length=500
                )
            except Exception as e:
                logger.error(f"Error finding grouped sections for index {index}: {e}")
                continue

            prepared_groups_for_index = []
            processed_q_indices_in_row = set()

            for group_match in grouped_sections_info:  # Iterate through GroupedRelevantSectionMatch objects
                q_indices = group_match.question_indices
                relevant_sections_for_group = group_match.relevant_sections # This is List[RelevantSectionMatch]
                
                if not relevant_sections_for_group:  # Skip if no relevant sections found for this group
                    continue

                group_questions = [input_questions[i] for i in q_indices]
                group_properties = [schema_property_questions[q] for q in group_questions]

                # Combine relevant sections into a single context for the group
                group_context = "\n".join(
                    [f"{match.section.title}: {match.section.content}" for match in relevant_sections_for_group]
                )
                # Calculate average score for the context found for this group
                group_scores = [match.score for match in relevant_sections_for_group]
                avg_group_score = sum(group_scores) / len(group_scores) if group_scores else 0.0

                prepared_groups_for_index.append({
                    "questions": group_questions,
                    "properties": group_properties,
                    "context": group_context,
                    "avg_score": avg_group_score,
                })

                # Track processed properties and indices
                properties_to_process.update(group_properties)
                processed_q_indices_in_row.update(q_indices)
                
            # Handle questions that might not have been grouped (e.g., if find_grouped_relevant_sections filtered some out)
            # This ensures we attempt to process all requested properties
            # Note: This part might need refinement depending on how find_grouped_relevant_sections behaves
            unprocessed_q_indices = [i for i, q in enumerate(input_questions) if i not in processed_q_indices_in_row]
            if unprocessed_q_indices:
                 # Option 1: Process them individually (less efficient) - requires find_relevant_sections
                 # Option 2: Log a warning or attempt to find context individually here
                 logger.warning(f"Warning: {len(unprocessed_q_indices)} questions were not processed in groups for index {index}.")
                 # For simplicity, we'll skip them for now, assuming find_grouped_relevant_sections covers all necessary questions.


            if prepared_groups_for_index:
                prepared_data[index] = prepared_groups_for_index

        return prepared_data, properties_to_process

    def _run_grouped_batch_qa(
        self, prepared_data: PreparedGroupedInput
    ) -> GroupedQAResults:
        """
        Runs batch inference for questions grouped by similarity and context.

        Args:
            prepared_data (PreparedGroupedInput): Data prepared by _prepare_inputs_with_question_grouping.

        Returns:
            GroupedQAResults: Dictionary mapping row index to property->QAResult mapping.
        """
        results_data: GroupedQAResults = {}
        logger.info(f"Running grouped batch QA inference with {self.qa_engine.model_name}...")
        # logger.info(prepared_data)
        logger.info(f"Number of groups: {len(prepared_data)}")
        logger.debug(prepared_data)

        for index, groups_for_index in tqdm(prepared_data.items(), desc="Processing QA for grouped questions"):
            results_data[index] = {}
            group_counter = 0 # For more informative extraction method string
            

            for group_info in groups_for_index:
                group_questions = group_info["questions"]
                group_properties = group_info["properties"]
                group_context = group_info["context"]
                avg_group_score = group_info["avg_score"] # Confidence derived from section matching
                group_counter += 1
                
                try:
                    # Get results for this batch of questions using the shared group context
                    batch_results: List[QAResult] = self.qa_engine.batch_questions_single_context(
                        group_questions, group_context
                    )

                    # Map results back to their original properties
                    for j, result in enumerate(batch_results):
                        if j < len(group_properties): # Safety check
                            original_property = group_properties[j]

                            # Assign confidence based on the relevance of the context found for the group
                            result.confidence = avg_group_score
                            # Update extraction method to be more descriptive
                            result.extraction_method = (
                                f"GroupedSimilarityQA"
                                f"CtxMatch: {self.matching_engine.model_name}, "
                                f"QA: {self.qa_engine.model_name})"
                            )
                            results_data[index][original_property] = result
                        else:
                            logger.warning(f"Warning: More results in batch ({len(batch_results)}) than properties ({len(group_properties)}) for index {index}, group {group_counter}. Result index {j} is out of bounds.")

                except Exception as e:
                    logger.error(f"Error during QA inference for index {index}, group {group_counter}: {e}")
                    # Log error results for this batch
                    error_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    for prop in group_properties:
                        results_data[index][prop] = QAResult(
                            answer="Error during generation",
                            extraction_time=error_time,
                            confidence=0.0,
                            extraction_method="GroupedSimilarityQA Error"
                        )

        return results_data
    
    def _populate_dataframe_with_grouped_qa_results(
        self,
        HF_df: pd.DataFrame,
        results_data: GroupedQAResults
    ) -> pd.DataFrame:
        """
        Populates the DataFrame with grouped QA results stored by property.

        Args:
            HF_df (pd.DataFrame): DataFrame to populate.
            results_data (GroupedQAResults): Results keyed by index and property.

        Returns:
            pd.DataFrame: Populated DataFrame.
        """
        logger.info("Populating DataFrame with grouped QA results...")

        for index, property_results in tqdm(results_data.items(), desc="Updating DataFrame with grouped results"):
            if index not in HF_df.index:
                logger.info(f"Warning: Index {index} from QA results not found in DataFrame.")
                continue

            for property_name, result in property_results.items():
                # Ensure the column exists
                if property_name not in HF_df.columns:
                    HF_df[property_name] = pd.Series([[] for _ in range(len(HF_df))], index=HF_df.index, dtype=object)

                # Create the standard extraction metadata dictionary
                extraction_info = self.add_default_extraction_info(
                    data=result.answer,
                    extraction_method=result.extraction_method,
                    confidence=result.confidence if result.confidence is not None else 0.0, # Handle potential None
                )

                # Update the cell
                # Ensure we are updating a list within the cell
                HF_df.loc[index, property_name] = [extraction_info]

        return HF_df


    def parse_fields_from_text_by_grouping_HF(self, HF_df: pd.DataFrame, max_questions_per_group: int = 10) -> pd.DataFrame:
        """
        Extract information from model card text using semantic question grouping and QA.

        This method groups questions by semantic similarity first, finds relevant context
        for each group, and then uses batched QA for efficiency.

        Args:
            HF_df (pd.DataFrame): DataFrame containing HuggingFace model information.
            max_questions_per_group (int, optional): Maximum questions to group per QA prompt.
                Defaults to 15.

        Returns:
            pd.DataFrame: DataFrame with extracted information from text fields using grouped QA.
        """
        # Generate queries for each schema property based on their descriptions
        schema_property_contexts = self.create_schema_property_contexts()
        
        if not schema_property_contexts:
            logger.info("No properties identified for text extraction. Skipping.")
            return HF_df

        # 1. Prepare inputs: Group questions by similarity within each row's context
        #    and find relevant sections for each question group.
        prepared_data, properties_to_process = self._prepare_inputs_with_question_grouping(
            HF_df, schema_property_contexts, max_questions_per_group
        )

        if not prepared_data:
            logger.info("No valid question groups found for QA. Skipping QA step.")
            return HF_df

        # 2. Perform batch QA inference on the prepared groups.
        results_data = self._run_grouped_batch_qa(
            prepared_data
        )

        # 3. Populate DataFrame with QA results if inference was successful.
        if results_data:
            HF_df = self._populate_dataframe_with_grouped_qa_results(
                HF_df, results_data
            )
            # Add processed properties to the list
            self.processed_properties.extend(list(properties_to_process))
            logger.info(f"Processed text fields via similarity-grouped QA: {list(properties_to_process)}")
        else:
            logger.info("Grouped QA inference did not produce results. Skipping DataFrame population.")

        # Clear GPU memory if possible
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

        return HF_df
    
    
    def process_dataframe(self, HF_df: pd.DataFrame, clean_columns: bool = True, unstructured_text_strategy: str = "context_matching", max_questions_per_group: int = 15) -> pd.DataFrame:
        """
        Process a HuggingFace DataFrame by applying all parsing methods.
        
        This method combines all parsing methods to extract information from
        HuggingFace model metadata and map it to FAIR4ML schema properties.
        
        Args:
            HF_df (pd.DataFrame): DataFrame containing HuggingFace model information
            clean_columns (bool, optional): Whether to remove original HF columns.
                Defaults to True.
            unstructured_text_strategy (str, optional): Strategy to use for unstructured text extraction.
                Options: "context_matching", "grouped", "individual". Defaults to "context_matching".
            max_questions_per_group (int, optional): Maximum questions per group when using
                grouped QA. Defaults to 15.
                
        Returns:
            pd.DataFrame: Processed DataFrame with FAIR4ML schema properties
        """
        # Initialize columns that will be populated
        all_schema_properties = list(self.schema_properties.keys())
        
        self.processed_properties = [] # Reset for each new dataframe processing call
        
        # Create columns if they don't exist, initializing with empty lists for consistency
        for col in all_schema_properties:
            if col not in HF_df.columns:
                # Initialize with empty lists stored as objects
                 HF_df[col] = pd.Series([[] for _ in range(len(HF_df))], dtype=object, index=HF_df.index)


        # Apply parsing methods in sequence
        logger.info("Step 1: Parsing known fields from HF metadata...")
        HF_df = self.parse_known_fields_HF(HF_df)
        logger.info("Step 2: Parsing fields from HF tags...")
        HF_df = self.parse_fields_from_tags_HF(HF_df)
        logger.info("Step 3: Parsing fields from YAML section of model card ...")
        HF_df = self.parse_fields_from_yaml_HF(HF_df)
        
        # Step 3: Determine properties for SchemaPropertyExtractor to process
        properties_for_extractor = [
            prop for prop in all_schema_properties if prop not in self.processed_properties
        ]
        
        if properties_for_extractor:
            logger.info(f"Step 4: Extracting schema properties from model cards using {unstructured_text_strategy} strategy for properties: {properties_for_extractor}")
            if unstructured_text_strategy != "None":
                HF_df = self.schema_property_extractor.extract_dataframe_schema_properties(
                    df=HF_df,
                    strategy=unstructured_text_strategy,
                    max_questions_per_group=max_questions_per_group,
                    properties_to_process=properties_for_extractor
                )
                self.processed_properties.extend(properties_for_extractor)
            else:
                logger.info("Step 4: No extraction for model card text.")
        else:
            logger.info("Step 4: No remaining properties for model card text extraction.")
        
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
                "card", # Remove the raw card text
            ]
            
            # Drop columns that exist in the DataFrame
            columns_to_remove = [col for col in columns_to_remove if col in HF_df.columns]
            if columns_to_remove:
                logger.info(f"Cleaning up columns: {columns_to_remove}")
                HF_df = HF_df.drop(columns=columns_to_remove)


        logger.info("DataFrame processing complete.")
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
        for property in properties:
            if property not in df.columns:
                df[property] = None
                
        for index in tqdm(df.index, desc=description):
            for property in properties:
                if property in df.columns:
                    if df.at[index, property] is None or df.at[index, property] == []:
                        df.at[index, property] = [
                            self.add_default_extraction_info(
                                "Information not found", extraction_method, confidence
                            )
                        ]
                    else:
                        # Check if the data is already in the desired format (list of dicts)
                        current_data = df.at[index, property]
                        if isinstance(current_data, list) and all(isinstance(item, dict) and 'data' in item for item in current_data):
                            # Data is already formatted, leave it as is
                            pass
                        else:
                            # Wrap the existing data
                            df.at[index, property] = [
                                self.add_default_extraction_info(
                                        current_data, extraction_method, confidence
                                )
                            ]
        
        return df
    
    def print_detailed_dataframe(self, HF_df: pd.DataFrame):
        """
        Print detailed information about the DataFrame.
        
        Args:
            HF_df (pd.DataFrame): DataFrame to print information about
        """
        logger.info("\n**DATAFRAME**")
        logger.info("\nColumns:", HF_df.columns.tolist())
        logger.info("\nShape:", HF_df.shape)
        logger.info("\nSample Data:")
        for col in HF_df.columns:
            logger.info(f"\n{col}:")
            for row in HF_df[col].head(3):  # Show only first 3 rows for each column
                # Limit the text to 100 characters
                if isinstance(row, list) and row: # Check if list is not empty
                    first_item = row[0]
                    if isinstance(first_item, dict) and 'data' in first_item:
                        row_data = first_item["data"]
                    if isinstance(row_data, str):
                        logger.info(row_data[:100])
                        if isinstance(row_data, list):
                             logger.info(f"[List: {len(row_data)} items]") # Avoid printing long lists
                        else:
                            logger.info(row_data)
                    else:
                        logger.info(f"[List item not dict or missing 'data']: {str(first_item)[:100]}")
                else:
                     logger.info(str(row)[:100]) # Print other types, truncated

            logger.info("")
        logger.info("\nDataFrame Info:")
        logger.info(HF_df.info())
