#!/usr/bin/env python3
"""
Integration tests for the SchemaPropertyExtractor class.

This module contains integration tests that demonstrate using the SchemaPropertyExtractor
with real models and data. These tests are designed to be run manually
rather than as part of automated CI/CD pipelines due to their resource requirements.
"""

import os
import unittest
import pandas as pd
import torch
import logging
import json
from typing import Dict, List, Any
import pytest

from mlentory_extract.core.QAMatchingEngine import QAMatchingEngine
from mlentory_extract.core.QAInferenceEngine import QAInferenceEngine
from mlentory_extract.core.SchemaPropertyExtractor import SchemaPropertyExtractor


# Setup logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class TestSchemaPropertyExtractorIntegration():
    """Integration tests for the SchemaPropertyExtractor class."""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures before running tests in the class."""
        # Check for GPU availability
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {cls.device}")
        
        # Define models to use (smaller versions for integration testing)
        cls.matching_model = "sentence-transformers/all-MiniLM-L6-v2"  # Smaller model for testing
        cls.qa_model = "Qwen/Qwen3-0.6B"  # Smaller QA model for testing
        
        # Create sample schema properties
        cls.schema_properties = {
            "fair4ml:mlTask": {
                "description": "The machine learning task that this model performs",
                "HF_Readme_Section": "Task; Model Description"
            },
            "schema.org:description": {
                "description": "A description of the model including its key capabilities",
                "HF_Readme_Section": "Model Description; Overview"
            },
            "fair4ml:evaluationMetrics": {
                "description": "Metrics used to evaluate the model's performance",
                "HF_Readme_Section": "Evaluation; Metrics; Results"
            }
        }
        
        # Example model cards
        cls.example_model_cards = {
            "gpt2": """
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
            
            ## Evaluation
            
            The model was evaluated using perplexity on a held-out validation set, achieving a score of 24.3.
            """,
            
            "bert": """
            # BERT Model
            
            ## Overview
            
            BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning model
            for natural language processing. BERT was pre-trained on a large corpus of unlabeled text including the entire
            Wikipedia and Book Corpus.
            
            ## Model Description
            
            BERT uses a bidirectional transformer architecture, allowing it to consider the context of a word based on all
            of its surroundings (left and right of the word). This is unlike previous models which looked at text sequences
            either from left to right or combined left-to-right and right-to-left training.
            
            ## Tasks
            
            BERT can be fine-tuned for various NLP tasks including:
            - Question answering
            - Natural language inference
            - Named entity recognition
            - Sentiment analysis
            
            ## Metrics
            
            The model achieves state-of-the-art results on the GLUE benchmark with an average score of 80.5%.
            """
        }
        
        try:
            # Initialize engines with real models
            cls.matching_engine = QAMatchingEngine(embedding_model=cls.matching_model)
            cls.qa_engine = QAInferenceEngine(model_name=cls.qa_model, batch_size=2)
            
            cls.test_ready = True
            print("Successfully initialized engines and extractor")
            
        except Exception as e:
            cls.test_ready = False
            print(f"Failed to initialize tests: {e}")
            import traceback
            print("Full stack trace:")
            traceback.print_exc()
    
    def setup_method(cls, method):
        """Skip tests if setup failed."""
        try:
            # Create the extractor
            cls.extractor = SchemaPropertyExtractor(
                qa_matching_engine=cls.matching_engine,
                qa_inference_engine=cls.qa_engine,
                schema_properties=cls.schema_properties
            )
        except Exception as e:
            cls.test_ready = False
            print(f"Failed to initialize tests: {e}")
            import traceback
            print("Full stack trace:")
            traceback.print_exc()
            
        if not cls.test_ready:
            pytest.skip("Test environment not properly set up")
    
    def test_context_matching_strategy(self):
        """Test the context matching strategy with real models."""
        print("Testing context matching strategy")
        
        # Extract properties from GPT-2 model card
        gpt2_results = self.extractor.extract_by_context_matching(
            self.example_model_cards["gpt2"],
            properties_to_process=None
        )
        
        # Log results
        print("GPT-2 Context Matching Results:")
        for prop, extractions in gpt2_results.items():
            for ext in extractions:
                print(f"{prop}: {ext['data'][:100]}... (Confidence: {ext['confidence']:.2f})")
        
        # Verify we got results for at least some properties
        assert len(gpt2_results) > 0, "No properties extracted from GPT-2 model card"
        
        # Extract properties from BERT model card
        bert_results = self.extractor.extract_by_context_matching(
            self.example_model_cards["bert"],
            properties_to_process=None
        )
        
        # Log results
        print("BERT Context Matching Results:")
        for prop, extractions in bert_results.items():
            for ext in extractions:
                print(f"{prop}: {ext['data'][:100]}... (Confidence: {ext['confidence']:.2f})")
        
        # Verify we got results for at least some properties
        assert len(bert_results) > 0, "No properties extracted from BERT model card"
    
    def test_grouped_qa_strategy(self):
        """Test the grouped QA strategy with real models."""
        print("Testing grouped QA strategy")
        
        # Extract properties from GPT-2 model card
        gpt2_results = self.extractor.extract_by_grouped_qa(
            self.example_model_cards["gpt2"],
            max_questions_per_group=2,
            properties_to_process=None
        )
        
        # Log results
        print("GPT-2 Grouped QA Results:")
        for prop, extractions in gpt2_results.items():
            for ext in extractions:
                print(f"{prop}: {ext['data'][:100]}... (Confidence: {ext['confidence']:.2f})")
        
        # Verify we got results for at least some properties
        assert len(gpt2_results) > 0, "No properties extracted from GPT-2 model card"
    
    def test_individual_qa_strategy(self):
        """Test the individual QA strategy with real models."""
        print("Testing individual QA strategy")
        
        # Extract properties from BERT model card
        bert_results = self.extractor.extract_by_individual_qa(
            self.example_model_cards["bert"],
            properties_to_process=None
        )
        
        # Log results
        print("BERT Individual QA Results:")
        for prop, extractions in bert_results.items():
            for ext in extractions:
                print(f"{prop}: {ext['data'][:100]}... (Confidence: {ext['confidence']:.2f})")
        
        # Verify we got results for at least some properties
        assert len(bert_results) > 0, "No properties extracted from BERT model card"
    
    def test_dataframe_extraction(self):
        """Test extracting properties from a DataFrame."""
        print("Testing DataFrame extraction")
        
        # Create a test DataFrame
        df = pd.DataFrame({
            "modelId": ["example/gpt2", "example/bert"],
            "card": [self.example_model_cards["gpt2"], self.example_model_cards["bert"]]
        })
        
        # Process DataFrame with the extractor
        result_df = self.extractor.extract_dataframe_schema_properties(
            df=df,
            strategy="context_matching",  # Use context matching for speed
            properties_to_process=None
        )
        
        # Log results
        print("DataFrame Extraction Results:")
        for index, row in result_df.iterrows():
            print(f"Model: {row['modelId']}")
            for prop in self.schema_properties.keys():
                if prop in row and row[prop]:
                    data = row[prop][0]['data']
                    confidence = row[prop][0]['confidence']
                    print(f"  {prop}: {data[:100]}... (Confidence: {confidence:.2f})")
        
        # Check for expected columns
        print(f"Result DataFrame columns: {result_df.columns}")
        for prop in self.schema_properties.keys():
            assert prop in result_df.columns, f"Property {prop} not found in DataFrame columns"
    
    def test_comparison_of_strategies(self):
        """Compare the results of different extraction strategies."""
        print("Comparing extraction strategies")
        
        model_card = self.example_model_cards["gpt2"]
        
        # Extract using all three strategies
        context_results = self.extractor.extract_by_context_matching(
            model_card,
            properties_to_process=None
        )
        grouped_results = self.extractor.extract_by_grouped_qa(
            model_card,
            max_questions_per_group=2,
            properties_to_process=None
        )
        individual_results = self.extractor.extract_by_individual_qa(
            model_card,
            properties_to_process=None
        )
        
        # Log and compare results
        print("Strategy Comparison:")
        for prop in self.schema_properties.keys():
            print(f"\nProperty: {prop}")
            
            if prop in context_results:
                context_data = context_results[prop][0]['data']
                context_conf = context_results[prop][0]['confidence']
                print(f"  Context: {context_data[:100]}... (Confidence: {context_conf:.2f})")
            else:
                print("  Context: Not found")
            
            if prop in grouped_results:
                grouped_data = grouped_results[prop][0]['data']
                grouped_conf = grouped_results[prop][0]['confidence']
                print(f"  Grouped: {grouped_data[:100]}... (Confidence: {grouped_conf:.2f})")
            else:
                print("  Grouped: Not found")
            
            if prop in individual_results:
                individual_data = individual_results[prop][0]['data']
                individual_conf = individual_results[prop][0]['confidence']
                print(f"  Individual: {individual_data[:100]}... (Confidence: {individual_conf:.2f})")
            else:
                print("  Individual: Not found")
        
        # We don't make assertions about the content of the results since they'll vary,
        # but we log them for manual inspection


if __name__ == "__main__":
    unittest.main() 