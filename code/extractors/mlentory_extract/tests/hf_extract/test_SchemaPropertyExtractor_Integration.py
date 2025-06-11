#!/usr/bin/env python3
"""
Integration tests for the SchemaPropertyExtractor class.

This module contains integration tests that demonstrate using the SchemaPropertyExtractor
with real models and data. These tests are designed to be run manually
rather than as part of automated CI/CD pipelines due to their resource requirements.
"""

import os
import pandas as pd
import torch
import logging
import json
import re
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

    def _log_context_matching_details(self, model_card: str, schema_property_contexts: dict, properties_to_analyze: list):
        """Log detailed information about the context matching process."""
        print("\n" + "-"*60)
        print("DETAILED CONTEXT MATCHING PROCESS")
        print("-"*60)
        
        # Create questions for each property using the generated contexts
        questions = list(schema_property_contexts.values())
        property_mapping = {context: prop for prop, context in schema_property_contexts.items()}
        
        print(f"Using {len(questions)} generated contexts as questions:")
        for i, (prop, context) in enumerate(schema_property_contexts.items()):
            print(f"\nQuestion {i+1} for {prop}:")
            print(f"  Context: {context}")
        
        print(f"\nFinding relevant sections for {len(questions)} questions...")
        
        # Use the matching engine to find relevant sections
        all_match_results = self.extractor.qa_matching_engine.find_relevant_sections(
            questions=questions,
            context=model_card,
            top_k=3
        )
        
        # Log detailed matching results
        for question, matches in zip(questions, all_match_results):
            prop = property_mapping[question]
            print(f"\n--- MATCHING RESULTS FOR {prop} ---")
            print(f"Question: '{question}'")
            print(f"Found {len(matches)} relevant sections:")
            
            for j, match in enumerate(matches):
                print(f"\n  Match {j+1}:")
                print(f"    Section Title: '{match.section.title}'")
                print(f"    Confidence Score: {match.score:.4f}")
                print(f"    Content Length: {len(match.section.content)} chars")
                print(f"    Content Preview: {match.section.content[:150]}...")
                print(f"    Full Content: {match.section.content}")

    def _log_grouped_qa_details(self, model_card: str, properties_to_analyze: list):
        """Log detailed information about the grouped QA process."""
        print("\n" + "-"*60)
        print("DETAILED GROUPED QA PROCESS")
        print("-"*60)
        
        if self.extractor.qa_inference_engine is None:
            print("QA engine is not available, cannot demonstrate grouped QA details.")
            return
            
        print("QA engine is available, showing grouped QA process...")
        print("Note: This will show how questions are grouped and processed together.")
        
        print("Grouped QA process involves:")
        print("1. Grouping similar questions together")
        print("2. Finding relevant context for each group")
        print("3. Running batch QA on grouped questions")
        print("4. Mapping results back to individual properties")

    def _log_individual_qa_details(self, model_card: str, properties_to_analyze: list):
        """Log detailed information about the individual QA process."""
        print("\n" + "-"*60)
        print("DETAILED INDIVIDUAL QA PROCESS")
        print("-"*60)
        
        if self.extractor.qa_inference_engine is None:
            print("QA engine is not available, cannot demonstrate individual QA details.")
            return
            
        print("QA engine is available, showing individual QA process...")
        print("Note: This processes each question individually with its own context.")
        
        print("Individual QA process involves:")
        print("1. Processing each property question separately")
        print("2. Finding relevant context for each individual question")
        print("3. Running QA inference for each question-context pair")
        print("4. Collecting individual results")

    # @pytest.mark.parametrize("strategy", ["context_matching", "grouped", "individual"])
    @pytest.mark.parametrize("strategy", ["context_matching"])
    def test_full_pipeline_with_logs(self, strategy):
        """
        Test the full pipeline with extensive logging to observe the step-by-step process.
        
        This test demonstrates:
        1. How markdown sections are extracted from model cards
        2. How these sections are matched against schema properties
        3. The confidence scores and matching details
        4. The final extraction results for the specified strategy
        
        Args:
            strategy: The extraction strategy to test (context_matching, grouped, individual)
        
        No assertions are made - this is purely for observation and debugging.
        """
        print("\n" + "="*80)
        print(f"STARTING FULL PIPELINE TEST WITH DETAILED LOGS - STRATEGY: {strategy.upper()}")
        print("="*80)
        
        # Use the GPT-2 model card as our test case
        model_card = self.example_model_cards["gpt2"]
        
        print("\n" + "-"*60)
        print("STEP 1: ANALYZING INPUT MODEL CARD")
        print("-"*60)
        print(f"Model card length: {len(model_card)} characters")
        
        print("\n" + "-"*60)
        print("STEP 2: EXTRACTING MARKDOWN SECTIONS")
        print("-"*60)
        
        # Extract sections using the matching engine directly to see the process
        sections = self.extractor.qa_matching_engine.markdown_parser.extract_hierarchical_sections(model_card)
        
        print(f"Found {len(sections)} sections in the model card:")
        for i, section in enumerate(sections):
            print(f"\nSection {i+1}:")
            print(f"  Title: '{section.title}'")
            print(f"  Content length: {len(section.content)} characters")
            print(f"  Content preview: {section.content[:100]}...")
            print(f"  Content: {section.content}")
        
        print("\n" + "-"*60)
        print("STEP 3: ANALYZING SCHEMA PROPERTIES TO EXTRACT")
        print("-"*60)
        
        properties_to_analyze = ["fair4ml:mlTask", "schema.org:description", "fair4ml:evaluationMetrics"]
        
        for prop in properties_to_analyze:
            if prop in self.schema_properties:
                prop_info = self.schema_properties[prop]
                print(f"\nProperty: {prop}")
                print(f"  Description: {prop_info['description']}")
                print(f"  Expected HF Sections: {prop_info['HF_Readme_Section']}")
        
        print("\n" + "-"*60)
        print("STEP 4: SCHEMA PROPERTY CONTEXT GENERATION")
        print("-"*60)
        
        print("Generating descriptive contexts for schema properties...")
        
        # Use the extractor's method to generate contexts
        schema_property_contexts = self.extractor.create_schema_property_contexts(
            properties_to_process=properties_to_analyze
        )
        
        print(f"Generated {len(schema_property_contexts)} property contexts:")
        for prop, context in schema_property_contexts.items():
            print(f"\n--- CONTEXT FOR {prop} ---")
            print("Raw property metadata:")
            if prop in self.schema_properties:
                prop_info = self.schema_properties[prop]
                print(f"  Description: {prop_info['description']}")
                print(f"  HF_Readme_Section: {prop_info.get('HF_Readme_Section', 'Not specified')}")
                
                # Show the humanization process
                base = prop.split(":", 1)[-1]
                human = re.sub(r"(?<=[a-z])([A-Z])", r" \1", base).replace("_", " ").title()
                print(f"  Property name transformation: '{prop}' -> '{base}' -> '{human}'")
                
                # Show section parsing
                raw_sections = prop_info.get("HF_Readme_Section", "").strip()
                sections = [s.strip() for s in raw_sections.split(";") if s.strip()]
                print(f"  Parsed sections: {sections}")
            
            print("Generated context:")
            print(f"  {context}")
        
        print("\n" + "-"*60)
        print("STEP 5: STRATEGY-SPECIFIC PROCESSING")
        print("-"*60)
        
        # Call the appropriate detailed logging function based on strategy
        if strategy == "context_matching":
            self._log_context_matching_details(model_card, schema_property_contexts, properties_to_analyze)
        elif strategy == "grouped":
            self._log_grouped_qa_details(model_card, properties_to_analyze)
        elif strategy == "individual":
            self._log_individual_qa_details(model_card, properties_to_analyze)
        
        print("\n" + "-"*60)
        print(f"STEP 6: RUNNING {strategy.upper()} EXTRACTION")
        print("-"*60)
        
        # Run the extraction using the specified strategy
        if strategy == "context_matching":
            results = self.extractor.extract_by_context_matching(
                model_card,
                properties_to_process=properties_to_analyze
            )
        elif strategy == "grouped":
            if self.extractor.qa_inference_engine is not None:
                results = self.extractor.extract_by_grouped_qa(
                    model_card,
                    max_questions_per_group=2,
                    properties_to_process=properties_to_analyze
                )
            else:
                print("QA engine is not available, skipping grouped QA extraction.")
                results = {}
        elif strategy == "individual":
            if self.extractor.qa_inference_engine is not None:
                results = self.extractor.extract_by_individual_qa(
                    model_card,
                    properties_to_process=properties_to_analyze
                )
            else:
                print("QA engine is not available, skipping individual QA extraction.")
                results = {}
        
        print(f"{strategy.capitalize()} extraction results:")
        for prop, extractions in results.items():
            print(f"\n{prop}:")
            for i, extraction in enumerate(extractions):
                print(f"  Extraction {i+1}:")
                print(f"    Data: {extraction['data']}")
                print(f"    Confidence: {extraction['confidence']:.4f}")
                print(f"    Method: {extraction['extraction_method']}")
                print(f"    Time: {extraction['extraction_time']}")
        
        print("\n" + "-"*60)
        print("STEP 7: DATAFRAME EXTRACTION TEST")
        print("-"*60)
        
        # Create a small test DataFrame
        test_df = pd.DataFrame({
            "modelId": ["test/gpt2-example"],
            "card": [model_card]
        })
        
        print(f"Created test DataFrame with {len(test_df)} rows")
        print(f"Running DataFrame extraction with {strategy} strategy...")
        
        result_df = self.extractor.extract_dataframe_schema_properties(
            df=test_df,
            strategy=strategy,
            properties_to_process=properties_to_analyze
        )
        
        print("DataFrame extraction results:")
        for index, row in result_df.iterrows():
            print(f"\nRow {index} (Model: {row.get('modelId', 'Unknown')}):")
            for prop in properties_to_analyze:
                if prop in row and row[prop]:
                    extraction = row[prop][0]  # Get first extraction
                    print(f"  {prop}:")
                    print(f"    Data: {extraction['data'][:200]}...")
                    print(f"    Confidence: {extraction['confidence']:.4f}")
                    print(f"    Method: {extraction['extraction_method']}")
                else:
                    print(f"  {prop}: No extraction found")
        
        print("\n" + "-"*60)
        print("STEP 8: SUMMARY")
        print("-"*60)
        
        print(f"Pipeline test completed successfully for {strategy} strategy!")
        print(f"- Analyzed model card with {len(sections)} sections")
        print(f"- Tested extraction for {len(properties_to_analyze)} properties")
        print(f"- Strategy used: {strategy}")
        print(f"- Extraction results: {'✓' if results else '✗'}")
        print(f"- QA engine available: {'✓' if self.extractor.qa_inference_engine else '✗'}")
        print("- DataFrame processing: ✓")
        
        print("\n" + "="*80)
        print(f"FULL PIPELINE TEST COMPLETED - {strategy.upper()}")
        print("="*80)