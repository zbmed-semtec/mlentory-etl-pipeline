#!/usr/bin/env python3
"""
Tests for the SchemaPropertyExtractor class.

This module contains unit tests for the SchemaPropertyExtractor class,
testing all three extraction strategies (context_matching, grouped, individual)
and both single model card and DataFrame operations.
"""

import os
import unittest
from unittest import mock
import pandas as pd
import torch
import logging
import json
from typing import Dict, List, Any

from mlentory_extract.core.QAMatchingEngine import QAMatchingEngine, RelevantSectionMatch
from mlentory_extract.core.QAInferenceEngine import QAInferenceEngine, QAResult
from mlentory_extract.core.SchemaPropertyExtractor import SchemaPropertyExtractor
from mlentory_extract.core.MarkdownParser import Section


# Setup basic logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class TestSchemaPropertyExtractor(unittest.TestCase):
    """Test cases for the SchemaPropertyExtractor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock QAMatchingEngine
        self.mock_matching_engine = mock.create_autospec(QAMatchingEngine)
        self.mock_matching_engine.model_name = "mock_matching_model"
        
        # Create a mock QAInferenceEngine
        self.mock_qa_engine = mock.create_autospec(QAInferenceEngine)
        self.mock_qa_engine.model_name = "mock_qa_model"
        
        # Test schema properties
        self.schema_properties = {
            "fair4ml:mlTask": {
                "description": "The machine learning task that this model performs",
                "HF_Readme_Section": "Task; Model Description"
            },
            "schema.org:description": {
                "description": "A description of the model including its key capabilities",
                "HF_Readme_Section": "Model Description; Overview"
            }
        }
        
        # Create the extractor with mock engines
        self.extractor = SchemaPropertyExtractor(
            qa_matching_engine=self.mock_matching_engine,
            qa_inference_engine=self.mock_qa_engine,
            schema_properties=self.schema_properties
        )
        
        # Example model card text for testing
        self.example_model_card = """
        # GPT-2 Model
        
        ## Model Description
        
        GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset of 8 million web pages. 
        GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some text.
        
        ## Task
        
        GPT-2 is designed for text generation tasks. It can be used for:
        - Completing a prompt with coherent text
        - Answering questions in a conversational context
        - Summarizing documents
        - Translation between languages
        
        ## Training Data
        
        The model was trained on WebText, a diverse dataset of web pages filtered for quality.
        
        ## Limitations
        
        GPT-2 has several limitations:
        - It can generate biased or harmful content
        - It lacks knowledge of events after its training cutoff
        - It may generate plausible-sounding but incorrect information
        """
        
        # Create test sections for mocking
        self.test_sections = [
            Section(
                title="Model Description",
                content="GPT-2 is a large transformer-based language model with 1.5 billion parameters.",
                start_idx=3,
                end_idx=6
            ),
            Section(
                title="Task",
                content="GPT-2 is designed for text generation tasks.",
                start_idx=8,
                end_idx=14
            )
        ]
        
        # Test QA results
        self.test_qa_results = [
            QAResult(
                answer="Text generation",
                extraction_time="2023-07-01_12:00:00",
                confidence=0.85,
                extraction_method="Test QA"
            ),
            QAResult(
                answer="GPT-2 is a large transformer-based language model",
                extraction_time="2023-07-01_12:00:00",
                confidence=0.92,
                extraction_method="Test QA"
            )
        ]

    def test_create_schema_property_contexts(self):
        """Test that schema property contexts are created correctly."""
        # Expected format of contexts
        contexts = self.extractor.create_schema_property_contexts()
        
        # Check if all properties were converted to contexts
        self.assertEqual(len(contexts), len(self.schema_properties))
        
        # Check if keys match
        for key in self.schema_properties.keys():
            self.assertIn(key, contexts)
            
        # Check if contexts contain the expected information
        for prop, context in contexts.items():
            # Context should contain the property description
            self.assertIn(self.schema_properties[prop]["description"], context)
            
            # Context should contain the HF_Readme_Section
            sections = self.schema_properties[prop]["HF_Readme_Section"].split(";")
            for section in sections:
                self.assertIn(section.strip(), context)

    def test_add_default_extraction_info(self):
        """Test the _add_default_extraction_info method."""
        data = "Test data"
        method = "Test method"
        confidence = 0.9
        
        result = self.extractor._add_default_extraction_info(data, method, confidence)
        
        # Check if the result has all expected keys
        self.assertIn("data", result)
        self.assertIn("extraction_method", result)
        self.assertIn("confidence", result)
        self.assertIn("extraction_time", result)
        
        # Check if the values are correct
        self.assertEqual(result["data"], data)
        self.assertEqual(result["extraction_method"], method)
        self.assertEqual(result["confidence"], confidence)

    def test_extract_by_context_matching(self):
        """Test extraction using the context matching strategy."""
        # Configure mock matching engine
        match1 = RelevantSectionMatch(section=self.test_sections[0], score=0.92)
        match2 = RelevantSectionMatch(section=self.test_sections[1], score=0.85)
        
        self.mock_matching_engine.find_relevant_sections.return_value = [
            [match2],  # For "fair4ml:mlTask" (first property processed)
            [match1]   # For "schema.org:description" (second property processed)
        ]
        
        # Call the method under test
        results = self.extractor.extract_by_context_matching(
            self.example_model_card,
            properties_to_process=None
        )
        
        # Verify mock was called with correct arguments
        self.mock_matching_engine.find_relevant_sections.assert_called_once()
        call_args = self.mock_matching_engine.find_relevant_sections.call_args[1]
        self.assertEqual(call_args["context"], self.example_model_card)
        self.assertEqual(call_args["top_k"], 1)
        
        # Check results
        self.assertIn("schema.org:description", results)
        self.assertIn("fair4ml:mlTask", results)
        
        # Check that extraction info contains the correct data
        self.assertEqual(results["schema.org:description"][0]["data"], self.test_sections[0].content)
        self.assertEqual(results["fair4ml:mlTask"][0]["data"], self.test_sections[1].content)
        
        # Check confidence scores
        self.assertEqual(results["schema.org:description"][0]["confidence"], 0.92)
        self.assertEqual(results["fair4ml:mlTask"][0]["confidence"], 0.85)

    def test_extract_by_grouped_qa(self):
        """Test extraction using the grouped QA strategy."""
        # Configure mock matching engine for grouped sections
        from mlentory_extract.core.QAMatchingEngine import GroupedRelevantSectionMatch
        
        relevant_sections = [
            RelevantSectionMatch(section=self.test_sections[0], score=0.92),
            RelevantSectionMatch(section=self.test_sections[1], score=0.85)
        ]
        
        grouped_match = GroupedRelevantSectionMatch(
            question_indices=[0, 1],  # Indices of questions in the list
            relevant_sections=relevant_sections
        )
        
        self.mock_matching_engine.find_grouped_relevant_sections.return_value = [grouped_match]
        
        # Configure mock QA engine
        self.mock_qa_engine.batch_questions_single_context.return_value = self.test_qa_results
        
        # Call the method under test
        results = self.extractor.extract_by_grouped_qa(
            self.example_model_card, 
            max_questions_per_group=5,
            properties_to_process=None
        )
        
        # Verify matching engine mock was called correctly
        self.mock_matching_engine.find_grouped_relevant_sections.assert_called_once()
        call_args = self.mock_matching_engine.find_grouped_relevant_sections.call_args[1]
        self.assertEqual(call_args["context"], self.example_model_card)
        self.assertEqual(call_args["top_k"], 3)
        self.assertEqual(call_args["max_questions_per_group"], 5)
        
        # Verify QA engine mock was called correctly
        self.mock_qa_engine.batch_questions_single_context.assert_called_once()
        
        # Check results
        props = list(self.schema_properties.keys())
        self.assertIn(props[0], results)
        self.assertIn(props[1], results)
        
        # Check that extraction info contains the correct data
        self.assertEqual(results[props[0]][0]["data"], self.test_qa_results[0].answer)
        self.assertEqual(results[props[1]][0]["data"], self.test_qa_results[1].answer)

    def test_extract_by_individual_qa(self):
        """Test extraction using the individual QA strategy."""
        # Configure mock matching engine
        match1 = RelevantSectionMatch(section=self.test_sections[0], score=0.92)
        match2 = RelevantSectionMatch(section=self.test_sections[1], score=0.85)
        
        self.mock_matching_engine.find_relevant_sections.return_value = [
            [match1],  # For first property
            [match2]   # For second property
        ]
        
        # Configure mock QA engine
        self.mock_qa_engine.batch_inference.return_value = self.test_qa_results
        
        # Call the method under test
        results = self.extractor.extract_by_individual_qa(
            self.example_model_card,
            properties_to_process=None
        )
        
        # Verify matching engine mock was called correctly
        self.mock_matching_engine.find_relevant_sections.assert_called_once()
        call_args = self.mock_matching_engine.find_relevant_sections.call_args[1]
        self.assertEqual(call_args["context"], self.example_model_card)
        self.assertEqual(call_args["top_k"], 3)
        
        # Verify QA engine mock was called
        self.mock_qa_engine.batch_inference.assert_called_once()
        
        # Check results
        props = list(self.schema_properties.keys())
        if props[0] in results and props[1] in results:
            # Check that extraction info contains the correct data
            self.assertEqual(results[props[0]][0]["data"], self.test_qa_results[0].answer)
            self.assertEqual(results[props[1]][0]["data"], self.test_qa_results[1].answer)

    def test_extract_schema_properties_from_model_card(self):
        """Test the extract_schema_properties_from_model_card method with different strategies."""
        # Mock the strategy-specific methods
        with mock.patch.object(self.extractor, 'extract_by_context_matching') as mock_context_matching, \
             mock.patch.object(self.extractor, 'extract_by_grouped_qa') as mock_grouped_qa, \
             mock.patch.object(self.extractor, 'extract_by_individual_qa') as mock_individual_qa:
            
            # Set up return values
            mock_context_matching.return_value = {"prop1": [{"data": "result1"}]}
            mock_grouped_qa.return_value = {"prop2": [{"data": "result2"}]}
            mock_individual_qa.return_value = {"prop3": [{"data": "result3"}]}
            
            # Test context_matching strategy
            result1 = self.extractor.extract_schema_properties_from_model_card(
                self.example_model_card, strategy="context_matching", properties_to_process=None
            )
            mock_context_matching.assert_called_once_with(self.example_model_card, properties_to_process=None)
            self.assertEqual(result1, {"prop1": [{"data": "result1"}]})
            
            # Test grouped strategy
            result2 = self.extractor.extract_schema_properties_from_model_card(
                self.example_model_card, strategy="grouped", max_questions_per_group=10, properties_to_process=None
            )
            mock_grouped_qa.assert_called_once_with(self.example_model_card, 10, properties_to_process=None)
            self.assertEqual(result2, {"prop2": [{"data": "result2"}]})
            
            # Test individual strategy
            result3 = self.extractor.extract_schema_properties_from_model_card(
                self.example_model_card, strategy="individual", properties_to_process=None
            )
            mock_individual_qa.assert_called_once_with(self.example_model_card, properties_to_process=None)
            self.assertEqual(result3, {"prop3": [{"data": "result3"}]})

    def test_extract_dataframe_schema_properties(self):
        """Test the extract_dataframe_schema_properties method."""
        # Create a test DataFrame
        df = pd.DataFrame({
            "modelId": ["example/gpt2", "example/bert"],
            "card": [
                self.example_model_card,
                "# BERT Model\n\nBERT is a transformer model for natural language understanding."
            ]
        })
        
        # Mock the extract_schema_properties_from_model_card method
        with mock.patch.object(
            self.extractor, 'extract_schema_properties_from_model_card'
        ) as mock_extract:
            # Set up return values for each call
            mock_extract.side_effect = [
                {
                    "fair4ml:mlTask": [{"data": "text generation", "confidence": 0.9, "extraction_method": "test", "extraction_time": "2023-07-01"}],
                    "schema.org:description": [{"data": "GPT-2 description", "confidence": 0.8, "extraction_method": "test", "extraction_time": "2023-07-01"}]
                },
                {
                    "fair4ml:mlTask": [{"data": "natural language understanding", "confidence": 0.85, "extraction_method": "test", "extraction_time": "2023-07-01"}],
                    "schema.org:description": [{"data": "BERT description", "confidence": 0.75, "extraction_method": "test", "extraction_time": "2023-07-01"}]
                }
            ]
            
            # Call the method under test
            result_df = self.extractor.extract_dataframe_schema_properties(
                df=df,
                strategy="context_matching",
                properties_to_process=None
            )
            
            # Verify the extract_schema_properties_from_model_card method was called twice
            self.assertEqual(mock_extract.call_count, 2)
            
            # Check that the mock was called with properties_to_process=None
            for call_args in mock_extract.call_args_list:
                self.assertIn("properties_to_process", call_args[1])
                self.assertIsNone(call_args[1]["properties_to_process"])
            
            # Check that the DataFrame has the expected columns
            self.assertIn("fair4ml:mlTask", result_df.columns)
            self.assertIn("schema.org:description", result_df.columns)
            
            # Check that the data was correctly added to the DataFrame
            self.assertEqual(
                result_df.loc[0, "fair4ml:mlTask"][0]["data"],
                "text generation"
            )
            self.assertEqual(
                result_df.loc[1, "schema.org:description"][0]["data"],
                "BERT description"
            )

    def test_extract_with_empty_model_card(self):
        """Test extraction with an empty model card."""
        empty_model_card = ""
        
        # Test all three strategies with empty input
        context_results = self.extractor.extract_by_context_matching(empty_model_card, properties_to_process=None)
        grouped_results = self.extractor.extract_by_grouped_qa(empty_model_card, properties_to_process=None)
        individual_results = self.extractor.extract_by_individual_qa(empty_model_card, properties_to_process=None)
        
        # All should return empty dictionaries
        self.assertEqual(context_results, {})
        self.assertEqual(grouped_results, {})
        self.assertEqual(individual_results, {})
        
        # Verify that none of the mock methods were called
        self.mock_matching_engine.find_relevant_sections.assert_not_called()
        self.mock_matching_engine.find_grouped_relevant_sections.assert_not_called()
        self.mock_qa_engine.batch_inference.assert_not_called()
        self.mock_qa_engine.batch_questions_single_context.assert_not_called()

    def test_extract_with_exception_handling(self):
        """Test that exceptions in engines are properly handled."""
        # Make the matching engine raise an exception
        self.mock_matching_engine.find_relevant_sections.side_effect = Exception("Test exception")
        
        # Call the method and check that it handles the exception
        results = self.extractor.extract_by_context_matching(
            self.example_model_card,
            properties_to_process=None
        )
        
        # Should return an empty dictionary
        self.assertEqual(results, {})
        
        # Reset the mock
        self.mock_matching_engine.find_relevant_sections.side_effect = None
        
        # Make the QA engine raise an exception
        self.mock_matching_engine.find_relevant_sections.return_value = [
            [RelevantSectionMatch(section=self.test_sections[0], score=0.92)],
            [RelevantSectionMatch(section=self.test_sections[1], score=0.85)]
        ]
        self.mock_qa_engine.batch_inference.side_effect = Exception("Test exception")
        
        # Call the method and check that it handles the exception
        results = self.extractor.extract_by_individual_qa(
            self.example_model_card,
            properties_to_process=None
        )
        
        # Should return an empty dictionary
        self.assertEqual(results, {})

    def test_create_schema_property_contexts_with_subset(self):
        """Test that schema property contexts are created correctly with a subset of properties."""
        properties_to_test = ["fair4ml:mlTask"]
        # Expected format of contexts
        contexts = self.extractor.create_schema_property_contexts(properties_to_process=properties_to_test)
        
        # Check if all properties were converted to contexts
        self.assertEqual(len(contexts), len(properties_to_test))
        
        # Check if keys match only the ones specified
        for key in properties_to_test:
            self.assertIn(key, contexts)
        
        for key in self.schema_properties.keys():
            if key not in properties_to_test:
                self.assertNotIn(key, contexts)
            
        # Check if contexts contain the expected information
        for prop, context in contexts.items():
            # Context should contain the property description
            self.assertIn(self.schema_properties[prop]["description"], context)
            
            # Context should contain the HF_Readme_Section
            sections = self.schema_properties[prop]["HF_Readme_Section"].split(";")
            for section in sections:
                self.assertIn(section.strip(), context)

    def test_extract_by_context_matching_with_subset(self):
        """Test extraction using context matching with a subset of properties."""
        properties_to_process = ["fair4ml:mlTask"]

        # Configure mock matching engine to return a match only for the specified property
        match_task = RelevantSectionMatch(section=self.test_sections[1], score=0.85) # Corresponds to "Task"
        
        # The mock should be configured based on the questions generated for properties_to_process
        # For simplicity, we assume create_schema_property_contexts generates one question per property
        # and that "fair4ml:mlTask" will be the first (and only) question passed.
        self.mock_matching_engine.find_relevant_sections.return_value = [
            [match_task] # Match for "fair4ml:mlTask"
        ]
        
        # Call the method under test
        results = self.extractor.extract_by_context_matching(
            self.example_model_card, 
            properties_to_process=properties_to_process
        )
        
        # Verify mock was called with correct arguments
        self.mock_matching_engine.find_relevant_sections.assert_called_once()
        call_args = self.mock_matching_engine.find_relevant_sections.call_args[1]
        
        # Check that the questions passed to find_relevant_sections correspond to properties_to_process
        generated_contexts_for_subset = self.extractor.create_schema_property_contexts(properties_to_process=properties_to_process)
        expected_questions = list(generated_contexts_for_subset.values())
        self.assertEqual(call_args["questions"], expected_questions)
        self.assertEqual(call_args["context"], self.example_model_card)
        self.assertEqual(call_args["top_k"], 1)
        
        # Check results: Should only contain the processed property
        self.assertEqual(len(results), len(properties_to_process))
        self.assertIn("fair4ml:mlTask", results)
        self.assertNotIn("schema.org:description", results) # Ensure other properties are not processed
        
        # Check that extraction info contains the correct data for the processed property
        self.assertEqual(results["fair4ml:mlTask"][0]["data"], self.test_sections[1].content)
        self.assertEqual(results["fair4ml:mlTask"][0]["confidence"], 0.85)

    def test_extract_by_grouped_qa_with_subset(self):
        """Test extraction using grouped QA with a subset of properties."""
        properties_to_process = ["fair4ml:mlTask"]
        property_key_to_process = "fair4ml:mlTask"

        # Configure mock matching engine for grouped sections
        from mlentory_extract.core.QAMatchingEngine import GroupedRelevantSectionMatch
        
        # For one property, we expect one group with one question index (0)
        relevant_sections_for_group = [
            RelevantSectionMatch(section=self.test_sections[1], score=0.85) # "Task" section
        ]
        
        grouped_match = GroupedRelevantSectionMatch(
            question_indices=[0],  # Index of the single question for "fair4ml:mlTask"
            relevant_sections=relevant_sections_for_group
        )
        
        self.mock_matching_engine.find_grouped_relevant_sections.return_value = [grouped_match]
        
        # Configure mock QA engine to return a single result for the single property
        single_qa_result = [self.test_qa_results[0]] # Corresponds to "Text generation"
        self.mock_qa_engine.batch_questions_single_context.return_value = single_qa_result
        
        # Call the method under test
        results = self.extractor.extract_by_grouped_qa(
            self.example_model_card, 
            max_questions_per_group=5,
            properties_to_process=properties_to_process
        )
        
        # Verify matching engine mock was called correctly
        self.mock_matching_engine.find_grouped_relevant_sections.assert_called_once()
        call_args = self.mock_matching_engine.find_grouped_relevant_sections.call_args[1]
        
        # Check that the questions passed to find_grouped_relevant_sections correspond to properties_to_process
        generated_contexts_for_subset = self.extractor.create_schema_property_contexts(properties_to_process=properties_to_process)
        expected_questions = list(generated_contexts_for_subset.values()) # Should be a list with one question string
        
        # Verify QA engine mock was called correctly
        self.mock_qa_engine.batch_questions_single_context.assert_called_once()
        # Ensure the questions passed to QA engine also match
        qa_call_args = self.mock_qa_engine.batch_questions_single_context.call_args[0]
        self.assertEqual(qa_call_args[0], expected_questions) # First argument is the list of questions
        
        # Check results: Should only contain the processed property
        self.assertEqual(len(results), len(properties_to_process))
        self.assertIn(property_key_to_process, results)
        self.assertNotIn("schema.org:description", results)
        
        # Check that extraction info contains the correct data for the processed property
        self.assertEqual(results[property_key_to_process][0]["data"], single_qa_result[0].answer)

    def test_extract_by_individual_qa_with_subset(self):
        """Test extraction using individual QA with a subset of properties."""
        properties_to_process = ["fair4ml:mlTask"]
        property_key_to_process = "fair4ml:mlTask"

        # Configure mock matching engine
        match_task = RelevantSectionMatch(section=self.test_sections[1], score=0.85) # Corresponds to "Task"
        
        self.mock_matching_engine.find_relevant_sections.return_value = [
            [match_task] # Match for "fair4ml:mlTask"
        ]
        
        # Configure mock QA engine
        self.mock_qa_engine.batch_inference.return_value = self.test_qa_results
        
        # Call the method under test
        results = self.extractor.extract_by_individual_qa(
            self.example_model_card,
            properties_to_process=properties_to_process
        )
        
        # Verify matching engine mock was called correctly
        self.mock_matching_engine.find_relevant_sections.assert_called_once()
        call_args = self.mock_matching_engine.find_relevant_sections.call_args[1]
        
        # Check that the questions passed to find_relevant_sections correspond to properties_to_process
        generated_contexts_for_subset = self.extractor.create_schema_property_contexts(properties_to_process=properties_to_process)
        expected_questions = list(generated_contexts_for_subset.values())
        self.assertEqual(call_args["questions"], expected_questions)
        self.assertEqual(call_args["context"], self.example_model_card)
        self.assertEqual(call_args["top_k"], 3)
        
        # Verify QA engine mock was called correctly
        self.mock_qa_engine.batch_inference.assert_called_once()
        
        # Check results: Should only contain the processed property
        self.assertEqual(len(results), len(properties_to_process))
        self.assertIn(property_key_to_process, results)
        self.assertNotIn("schema.org:description", results)
        
        # Check that extraction info contains the correct data for the processed property
        self.assertEqual(results[property_key_to_process][0]["data"], self.test_qa_results[0].answer)


if __name__ == "__main__":
    unittest.main() 