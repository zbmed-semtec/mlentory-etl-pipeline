import json
import pandas as pd
import torch
from typing import Any, Dict, List, Set, Tuple, Union, Optional
from datetime import datetime
from tqdm import tqdm
import math
import os

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile, RepoFolder

from mlentory_extract.core.QAMatchingEngine import QAMatchingEngine
from mlentory_extract.core.QAInferenceEngine import QAInferenceEngine, QAResult


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
        matching_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
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
                print("\nUSING GPU\n")
            else:
                self.device = None
                print("\nNOT USING GPU\n")
        except ModuleNotFoundError:
            # If torch is not available, assume no GPU
            self.device = None
            print("\nNOT USING GPU\n")
            
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
        self.qa_engine = QAInferenceEngine(model_name=qa_model_name, batch_size=1)
        
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
        # HF_df.loc[:, "schema.org:author"] = HF_df.loc[:, "author"]
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
            "schema.org:releaseNotes",
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
            ml_tasks = []
            base_models = set()
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
                if tag_for_task in self.tags_task_names:
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
                
                # Extract base models (fair4ml:baseModel)
                if "base_model:" in tag:
                    base_model = tag.split(":")[-1]
                    base_models.add(base_model)
                
                # Extract languages (inLanguage)
                if tag_lower in self.tags_language:
                    languages.append(tag)
                
                # Extract libraries (keywords)
                if tag_lower in self.tags_libraries_names:
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
            
            HF_df.at[index, "fair4ml:fineTunedFrom"] = list(base_models)
            
            HF_df.at[index, "schema.org:license"] = licenses
            
            HF_df.at[index, "schema.org:inLanguage"] = languages
            
            HF_df.at[index, "codemeta:referencePublication"] = arxiv_ids
            
            all_keywords = keywords + libraries
            HF_df.at[index, "schema.org:keywords"] = all_keywords
        
        # Add extraction metadata
        properties = [
            "fair4ml:mlTask",
            "fair4ml:trainedOn",
            "fair4ml:evaluatedOn",
            "fair4ml:testedOn",
            "fair4ml:fineTunedFrom",
            "schema.org:license",
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
    
    def _prepare_qa_inputs(self, HF_df: pd.DataFrame, schema_property_questions: Dict[str, str]) -> Tuple[List[Dict], Set[str]]:
        """Prepares inputs for the QA engine by finding relevant context sections and their scores."""
        qa_inputs_for_df = []
        properties_to_process = set()
        
        print("Preparing QA inputs by matching questions to context sections...")
        for index, row in tqdm(
            HF_df.iterrows(), total=len(HF_df), desc="Matching questions to contexts"
        ):
            context = row.get("card", "") # Use .get for safety
            print(f"\n \n Context: {context} \n \n")
            if not context or not isinstance(context, str):
                continue # Skip if no valid context
            
            question_items = schema_property_questions.items()
            questions = [question for question, _ in question_items]
            
            match_results = self.matching_engine.find_relevant_sections(
                questions=questions,
                context=context,
                top_k=4
            )
            
            for question_item, match_results in zip(question_items, match_results):
                question, property = question_item
                print(f"\n \n Question: {question} \n \n")
                new_context = "\n".join([match[0].title + ": " + match[0].content for match in match_results])
                print(f"\n\n New context for question:\n\n {new_context} \n \n")
                match_scores = [match[1] for match in match_results]
                
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
            
            
            # for question, prop in schema_property_questions.items():
            #     try:
            #         print(f"\n \n Questionssssssssss!!!!!!!!!!!!!!!: {question} \n \n")
            #         # Find the most relevant context snippets for this specific question
            #         match_results = self.matching_engine.find_relevant_sections(
            #             questions=[question],
            #             context=context,
            #             top_k=4
            #         )
                    
            #         best_matches = match_results[0] if match_results else []
                    
            #         if best_matches:
            #             new_context = "\n".join([match[0].content for match in best_matches])
            #             print(f"\n \n New context for question: {question} \n \n {new_context} \n \n")
            #             match_scores = [match[1] for match in best_matches]
                        
            #             if new_context:
            #                 qa_inputs_for_df.append(
            #                     {
            #                         "index": index,
            #                         "prop": prop,
            #                         "question": question,
            #                         "context": new_context,
            #                         "scores": match_scores,
            #                     }
            #                 )
            #                 properties_to_process.add(prop)
            #     except Exception as e:
            #         import traceback
            #         print(f"Error getting context/scores for Q: '{question}' on index {index}:\n{traceback.format_exc()}")
        
        return qa_inputs_for_df, properties_to_process

    def _run_batch_qa(self, qa_inputs_for_df: List[Dict]) -> List[QAResult]:
        """Runs batch inference using the QA engine."""
        print(f"Performing batch QA inference with {self.qa_engine.model_name}...")
        questions = [item["question"] for item in qa_inputs_for_df]
        contexts = [item["context"] for item in qa_inputs_for_df]

        try:
            qa_results = self.qa_engine.batch_inference(questions, contexts)
            return qa_results
        except Exception as e:
            print(f"Error during batch QA inference: {e}. Skipping QA population.")
            return [] # Return empty list on error

    def _populate_dataframe_with_qa_results(
        self,
        HF_df: pd.DataFrame,
        qa_inputs_for_df: List[Dict],
        qa_results: List[QAResult]
    ) -> pd.DataFrame:
        """Populates the DataFrame with QA results and calculated confidence."""
        print("Populating DataFrame with QA results...")
        if len(qa_results) != len(qa_inputs_for_df):
            print(f"Warning: Mismatch between QA inputs ({len(qa_inputs_for_df)}) and results ({len(qa_results)}). Results may be incomplete.")
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
        schema_property_questions = self.create_schema_property_questions()
        if not schema_property_questions:
            print("No properties identified for text extraction. Skipping.")
            return HF_df

        # 1. Prepare QA inputs
        qa_inputs_for_df, properties_to_process = self._prepare_qa_inputs(HF_df, schema_property_questions)

        if not qa_inputs_for_df:
            print("No valid question-context pairs found for QA. Skipping QA step.")
            return HF_df

        # 2. Perform batch QA inference
        qa_results = self._run_batch_qa(qa_inputs_for_df)

        # 3. Populate DataFrame with QA results if inference was successful
        if qa_results:
            HF_df = self._populate_dataframe_with_qa_results(HF_df, qa_inputs_for_df, qa_results)
            # Add processed properties to the list only if population occurred
            self.processed_properties.extend(list(properties_to_process))
            print(f"Processed text fields via QA: {list(properties_to_process)}")
        else:
             print("QA inference did not produce results. Skipping DataFrame population for QA.")

        # Clear GPU memory if possible
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

        return HF_df
    
    def create_schema_property_questions(self) -> Dict[str, str]:
        """
        Generate a dictionary of questions for each schema property based on their descriptions.
        
        Returns:
            Dict[str, str]: Dictionary of questions for each schema property
        """
        context_queries = {}
        
        # Generate a question for each property based on its description
        for property_name, metadata in self.schema_properties.items():
            # Skip properties that are not suitable for text extraction
            if property_name in self.processed_properties:
                continue
                
            description = metadata.get("description", "")
            posible_sections = metadata.get("HF_Readme_Section", "")
            
            if description:
                # Format the property name for better readability
                readable_prop = property_name.replace("fair4ml:", "").replace("codemeta:", "").replace("schema.org:", "")
                readable_prop = readable_prop.replace("_", " ").replace(":", " ")
                
                # Put a space between each uppercase letter: ConditionsOfUse -> Conditions of Use
                temp_readable_prop = ""
                for i in range(len(readable_prop)):
                    if readable_prop[i].isupper() and i != 0 and readable_prop[i-1].islower():
                        temp_readable_prop += " " + readable_prop[i]
                    else:
                        temp_readable_prop += readable_prop[i]
                        
                readable_prop = temp_readable_prop
                
                question = f"Find the {readable_prop} in the following text, here is a description of the property: ({description}) here are some related sections: {posible_sections}"
                
                context_queries[question] = property_name
                
        return context_queries
    
    
    
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
        # Initialize columns that will be populated
        schema_columns = list(self.schema_properties.keys())
        
        # Create columns if they don't exist, initializing with empty lists for consistency
        for col in schema_columns:
            if col not in HF_df.columns:
                # Initialize with empty lists stored as objects
                 HF_df[col] = pd.Series([[] for _ in range(len(HF_df))], dtype=object, index=HF_df.index)


        # Apply parsing methods in sequence
        print("Step 1: Parsing known fields from HF metadata...")
        HF_df = self.parse_known_fields_HF(HF_df)
        print("Step 2: Parsing fields from HF tags...")
        HF_df = self.parse_fields_from_tags_HF(HF_df)
        print("Step 3: Parsing fields from model card text using Matching + QA...")
        HF_df = self.parse_fields_from_text_HF(HF_df) # This now uses QA
        
        
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
                print(f"Cleaning up columns: {columns_to_remove}")
                HF_df = HF_df.drop(columns=columns_to_remove)


        print("DataFrame processing complete.")
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
                        df.at[index, property] = [
                            self.add_default_extraction_info(
                                df.at[index, property], extraction_method, confidence
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
