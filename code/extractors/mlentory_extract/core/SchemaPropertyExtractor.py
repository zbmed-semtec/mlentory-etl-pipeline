import json
import pandas as pd
import torch
from typing import Any, Dict, List, Set, Tuple, Union, Optional
from datetime import datetime
from tqdm import tqdm
import math
import os
import logging
import pprint
import re


from mlentory_extract.core.QAMatchingEngine import QAMatchingEngine, RelevantSectionMatch, GroupedRelevantSectionMatch
from mlentory_extract.core.QAInferenceEngine import QAInferenceEngine, QAResult


class SchemaPropertyExtractor:
    """
    A class to extract structured schema properties from unstructured text.
    
    This class encapsulates multiple strategies for extracting schema property values
    from unstructured text (like model cards). It provides three main extraction strategies:
    1. Direct context matching - Find the best matching section and use its content directly
    2. Grouped QA - Group similar questions, find relevant context for each group, run batch QA
    3. Individual QA - Process each question individually with matching and QA
    
    Attributes:
        qa_matching_engine: Engine for semantic matching between questions and text sections
        qa_inference_engine: Engine for extractive Question Answering
        schema_properties: Dictionary of schema properties and their metadata
    """
    def __init__(
        self, 
        qa_matching_engine: QAMatchingEngine, 
        qa_inference_engine: QAInferenceEngine,
        schema_properties: Dict[str, Dict[str, str]] = None
    ):
        """
        Initialize the Schema Property Extractor.
        
        Args:
            qa_matching_engine: Engine for semantic matching
            qa_inference_engine: Engine for extractive QA
            schema_properties: Dictionary of schema properties and their metadata
        """
        self.qa_matching_engine = qa_matching_engine
        self.qa_inference_engine = qa_inference_engine
        self.schema_properties = schema_properties or {}
    
    def _add_default_extraction_info(
        self, data: Any, extraction_method: str, confidence: float
    ) -> Dict:
        """
        Create a standardized dictionary for extraction metadata.
        
        Args:
            data: The extracted information
            extraction_method: Method used for extraction
            confidence: Confidence score of the extraction
            
        Returns:
            dict: Dictionary containing extraction metadata
        """
        return {
            "data": data,
            "extraction_method": extraction_method,
            "confidence": confidence,
            "extraction_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }
    
    def create_schema_property_contexts(
        self, properties_to_process: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Build a text-context for each schema property that can be used
        to match against a list of markdown sections.

        Args:
            properties_to_process: Optional list of property names to generate contexts for.
                                   If None, all schema properties will be processed.

        Returns:
            Dict[str, str]: Dictionary mapping property names to formatted context blocks
        """
        contexts: Dict[str, str] = {}
        
        target_properties = properties_to_process if properties_to_process is not None else self.schema_properties.keys()

        for prop in target_properties:
            if prop not in self.schema_properties:
                logging.warning(f"Property {prop} requested for context creation but not found in schema_properties.")
                continue
            
            meta = self.schema_properties[prop]
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
        
    def extract_schema_properties_from_model_card(
        self, 
        model_card_text: str, 
        strategy: str = "context_matching",
        max_questions_per_group: int = 10,
        properties_to_process: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract schema properties from a model card text using the specified strategy.
        
        Args:
            model_card_text: The model card text to extract from
            strategy: Extraction strategy to use (context_matching, grouped, individual)
            max_questions_per_group: Maximum questions per group when using grouped strategy
            properties_to_process: Optional list of property names to extract.
                                   If None, all eligible properties will be extracted.
            
        Returns:
            Dict[str, List[Dict]]: Dictionary mapping property names to lists of extraction info dictionaries
        """
        # Select the appropriate extraction strategy
        if strategy == "context_matching":
            return self.extract_by_context_matching(model_card_text, properties_to_process=properties_to_process)
        elif strategy == "grouped":
            return self.extract_by_grouped_qa(model_card_text, max_questions_per_group, properties_to_process=properties_to_process)
        else:  # Default to individual QA
            return self.extract_by_individual_qa(model_card_text, properties_to_process=properties_to_process)
    
    def extract_by_context_matching(
        self, model_card_text: str, properties_to_process: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract schema properties by finding the best-matching section for each property.
        
        For each property, finds the single markdown section in the model card that best matches
        the property's description and uses its content directly as the extracted value.
        
        Args:
            model_card_text: The model card text to extract from
            properties_to_process: Optional list of property names to extract.
                                   If None, all eligible properties will be extracted.
            
        Returns:
            Dict[str, List[Dict]]: Dictionary mapping property names to lists of extraction info dictionaries
        """
        logging.info("Starting schema property extraction using context matching")
        result = {}
        
        # Skip if no model card text is provided
        if not model_card_text or not isinstance(model_card_text, str):
            logging.warning("No model card text provided for context matching extraction")
            return result
            
        # Generate descriptive queries for each schema property
        schema_property_queries_map = self.create_schema_property_contexts(properties_to_process=properties_to_process)
        
        if not schema_property_queries_map:
            logging.warning("No schema properties eligible for context matching extraction")
            return result
            
        property_names = list(schema_property_queries_map.keys())
        descriptive_queries = list(schema_property_queries_map.values())
        
        try:
            # Find the most relevant sections for each property
            all_matches = self.qa_matching_engine.find_relevant_sections(
                questions=descriptive_queries,
                context=model_card_text,
                top_k=1  # We only want the single best section for each property
            )
            
            # Process each property and its corresponding match result
            for i, prop_name in enumerate(property_names):
                if i < len(all_matches):  # Safety check
                    top_match_list = all_matches[i]
                    
                    if top_match_list:  # If a match was found
                        best_match = top_match_list[0]
                        extracted_data = best_match.section.content
                        confidence_score = best_match.score
                        
                        # Create extraction info dictionary
                        extraction_info = self._add_default_extraction_info(
                            data=extracted_data,
                            extraction_method="DirectContextMatch_SectionContent",
                            confidence=confidence_score
                        )
                        
                        # Add to results
                        result[prop_name] = [extraction_info]
                    else:
                        # No match found for this property
                        result[prop_name] = [self._add_default_extraction_info(
                            data="Information not found by context matching",
                            extraction_method="DirectContextMatch_NoSectionFound",
                            confidence=0.0
                        )]
            
        except Exception as e:
            logging.error(f"Error in context matching extraction: {e}", exc_info=True)
        
        # Clear GPU memory if possible
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
        
        return result
    
    def extract_by_grouped_qa(
        self, 
        model_card_text: str, 
        max_questions_per_group: int = 10,
        properties_to_process: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract schema properties by grouping similar questions and running batch QA.
        
        Groups questions by semantic similarity, finds relevant context for each group,
        and then uses batched QA for efficiency.
        
        Args:
            model_card_text: The model card text to extract from
            max_questions_per_group: Maximum questions per group
            properties_to_process: Optional list of property names to extract.
                                   If None, all eligible properties will be extracted.
            
        Returns:
            Dict[str, List[Dict]]: Dictionary mapping property names to lists of extraction info dictionaries
        """
        logging.info("Starting schema property extraction using grouped QA")
        result = {}
        
        # Skip if no model card text is provided
        if not model_card_text or not isinstance(model_card_text, str):
            logging.warning("No model card text provided for grouped QA extraction")
            return result
            
        # Generate descriptive queries for each schema property
        schema_property_queries_map = self.create_schema_property_contexts(properties_to_process=properties_to_process)
        
        if not schema_property_queries_map:
            logging.warning("No schema properties eligible for grouped QA extraction")
            return result
        
        try:
            # Find grouped relevant sections for all properties
            grouped_sections = self.qa_matching_engine.find_grouped_relevant_sections(
                questions=list(schema_property_queries_map.keys()),
                context=model_card_text,
                top_k=3,
                max_questions_per_group=max_questions_per_group
            )
            
            # Process each group
            for group_match in grouped_sections:
                question_indices = group_match.question_indices
                relevant_sections = group_match.relevant_sections
                
                if not relevant_sections:  # Skip if no relevant sections found
                    continue
                
                # Get the properties for this group
                group_properties = [list(schema_property_queries_map.keys())[i] for i in question_indices]
                group_questions = [list(schema_property_queries_map.values())[i] for i in question_indices]
                
                # Combine relevant sections into a single context
                group_context = "\n".join(
                    [f"{match.section.title}: {match.section.content}" for match in relevant_sections]
                )
                
                # Calculate average score for the context
                group_scores = [match.score for match in relevant_sections]
                avg_group_score = sum(group_scores) / len(group_scores) if group_scores else 0.0
                
                try:
                    # Run batch QA on this group
                    batch_results = self.qa_inference_engine.batch_questions_single_context(
                        group_questions, 
                        group_context
                    )
                    
                    # Process results
                    for i, prop_name in enumerate(group_properties):
                        if i < len(batch_results):  # Safety check
                            qa_result = batch_results[i]
                            
                            # Create extraction info dictionary
                            extraction_info = self._add_default_extraction_info(
                                data=qa_result.answer,
                                extraction_method=f"GroupedQA (Context: {self.qa_matching_engine.model_name}, QA: {self.qa_inference_engine.model_name})",
                                confidence=avg_group_score
                            )
                            
                            # Add to results
                            result[prop_name] = [extraction_info]
            
                except Exception as e:
                    logging.error(f"Error in batch QA for a group: {e}", exc_info=True)
                    # Mark all properties in this group as error
                    for prop_name in group_properties:
                        result[prop_name] = [self._add_default_extraction_info(
                            data=f"Error during QA: {str(e)}",
                            extraction_method="GroupedQA_Error",
                            confidence=0.0
                        )]
            
        except Exception as e:
            logging.error(f"Error in grouped QA extraction: {e}", exc_info=True)
        
        # Clear GPU memory if possible
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
        
        return result
    
    def extract_by_individual_qa(
        self, model_card_text: str, properties_to_process: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract schema properties by processing each question individually.
        
        For each property, finds relevant sections in the model card and uses QA
        to extract an answer from those sections.
        
        Args:
            model_card_text: The model card text to extract from
            properties_to_process: Optional list of property names to extract.
                                   If None, all eligible properties will be extracted.
            
        Returns:
            Dict[str, List[Dict]]: Dictionary mapping property names to lists of extraction info dictionaries
        """
        logging.info("Starting schema property extraction using individual QA")
        result = {}
        
        # Skip if no model card text is provided
        if not model_card_text or not isinstance(model_card_text, str):
            logging.warning("No model card text provided for individual QA extraction")
            return result
            
        # Generate descriptive queries for each schema property
        schema_property_queries_map = self.create_schema_property_contexts(properties_to_process=properties_to_process)
        
        if not schema_property_queries_map:
            logging.warning("No schema properties eligible for individual QA extraction")
            return result
            
        property_names = list(schema_property_queries_map.keys())
        descriptive_queries = list(schema_property_queries_map.values())
        
        try:
            # Find relevant sections for each property
            all_matches = self.qa_matching_engine.find_relevant_sections(
                questions=descriptive_queries,
                context=model_card_text,
                top_k=3  # Get multiple sections for better context
            )
            
            # Prepare QA inputs
            qa_inputs = []
            for i, prop_name in enumerate(property_names):
                if i < len(all_matches):  # Safety check
                    matches = all_matches[i]
                    
                    if matches:  # If matches were found
                        # Combine the relevant sections into a single context
                        context = "\n".join([
                            f"{match.section.title}: {match.section.content}" 
                            for match in matches
                        ])
                        
                        # Calculate average confidence
                        scores = [match.score for match in matches]
                        avg_score = sum(scores) / len(scores) if scores else 0.0
                        
                        qa_inputs.append({
                            "prop_name": prop_name,
                            "question": descriptive_queries[i],
                            "context": context,
                            "score": avg_score
                        })
            
            if qa_inputs:
                # Run batch QA
                questions = [item["question"] for item in qa_inputs]
                contexts = [item["context"] for item in qa_inputs]
                
                qa_results = self.qa_inference_engine.batch_inference(questions, contexts)
                
                # Process results
                for i, qa_result in enumerate(qa_results):
                    if i < len(qa_inputs):  # Safety check
                        input_item = qa_inputs[i]
                        prop_name = input_item["prop_name"]
                        
                        # Create extraction info dictionary
                        extraction_info = self._add_default_extraction_info(
                            data=qa_result.answer,
                            extraction_method=f"IndividualQA (Context: {self.qa_matching_engine.model_name}, QA: {self.qa_inference_engine.model_name})",
                            confidence=input_item["score"]
                        )
                        
                        # Add to results
                        result[prop_name] = [extraction_info]
            
        except Exception as e:
            logging.error(f"Error in individual QA extraction: {e}", exc_info=True)
        
        # Clear GPU memory if possible
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
        
        return result
    
    def extract_dataframe_schema_properties(
        self, 
        df: pd.DataFrame,
        strategy: str = "context_matching",
        max_questions_per_group: int = 10,
        properties_to_process: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract schema properties from a DataFrame containing model cards.
        
        Args:
            df: DataFrame containing model cards in a 'card' column
            strategy: Extraction strategy to use (context_matching, grouped, individual)
            max_questions_per_group: Maximum questions per group when using grouped strategy
            properties_to_process: Optional list of property names to extract.
                                   If None, all eligible properties will be extracted.
            
        Returns:
            pd.DataFrame: DataFrame with extracted schema properties
        """
        logging.info(f"Processing DataFrame with {len(df)} rows using {strategy} strategy")
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting schema properties ({strategy})"):
            model_card_text = row.get("card", "")
            
            if not model_card_text or not isinstance(model_card_text, str):
                logging.debug(f"No model card text for index {index}. Skipping.")
                continue
                
            # Extract properties from this model card
            extracted_properties = self.extract_schema_properties_from_model_card(
                model_card_text, 
                strategy=strategy,
                max_questions_per_group=max_questions_per_group,
                properties_to_process=properties_to_process
            )
            
            # Update DataFrame with extracted properties
            for prop_name, extraction_info in extracted_properties.items():
                if prop_name not in df.columns:
                    df[prop_name] = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)
                
                df.loc[index, prop_name] = extraction_info
        
        return df