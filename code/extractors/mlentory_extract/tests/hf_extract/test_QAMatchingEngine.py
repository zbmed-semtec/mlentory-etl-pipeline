import os
import sys
import pytest
import torch
import numpy as np
import logging
import time
from unittest.mock import patch, MagicMock
from typing import List

from pathlib import Path

# Add the parent directory to the Python path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from mlentory_extract.core.QAMatchingEngine import QAMatchingEngine
from mlentory_extract.core.MarkdownParser import MarkdownParser, Section

# Configure logging
LOG_FILENAME = "qa_matching_engine_test_log.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Set log file path in tests directory
log_file_path = os.path.join(os.path.dirname(__file__), LOG_FILENAME)

@pytest.fixture(scope="session")
def test_logger():
    """
    Pytest fixture to configure logging for the test session.
    
    Returns:
        logging.Logger: Configured logger for test session
    """
    logger = logging.getLogger("QAMatchingEngineTests")
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if tests are run multiple times in one session
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Log session start
    logger.info("=" * 30 + " TEST SESSION STARTED " + "=" * 30)
    yield logger
    # Log session end
    logger.info("=" * 30 + " TEST SESSION ENDED " + "=" * 30)


class TestQAMatchingEngine:
    """
    Test suite for QAMatchingEngine class.
    
    Tests the initialization, embedding generation, section finding,
    and context retrieval functionalities using real models and parser.
    """

    @pytest.fixture(scope="module")
    def engine(self):
        """
        Fixture for QAMatchingEngine with real components.
        
        Returns:
            QAMatchingEngine: Engine instance with real components
        """
        # Use a small and fast model for testing
        # engine = QAMatchingEngine(embedding_model="cross-encoder/ms-marco-MiniLM-L6-v2")
        engine = QAMatchingEngine(embedding_model="Alibaba-NLP/gte-modernbert-base")
        return engine

    @pytest.fixture
    def sample_context(self):
        """
        Fixture providing representative markdown content inspired by a model card.

        Returns:
            str: Sample markdown content with various sections.
        """
        # Content inspired by Gemma README, but simplified for testing
        return """
# Gemma 2 Model Overview

**License**: gemma
**Authors**: Google

## Description
Gemma is a family of lightweight, state-of-the-art open models from Google. Built from the same research and technology used to create the Gemini models. Suitable for question answering, summarization, and reasoning.

## Usage Examples

### Running on GPU
Use `AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", device_map="auto")`.

### Precisions
You can use `torch.bfloat16` (native), `torch.float16`, or upcast to `torch.float32`. 
Example with float16: `torch_dtype=torch.float16, revision="float16"`.

### Quantization
Supports 8-bit and 4-bit quantization via `bitsandbytes`. Use `BitsAndBytesConfig`.

## Training Data

Trained on a diverse mix of web documents, code, and mathematics. 
The 9B model used 8 trillion tokens.
Data was filtered for CSAM and sensitive information.

## Implementation Details

### Hardware
Training utilized Google's Tensor Processing Unit (TPU) hardware, specifically TPUv5p.
TPUs offer performance and scalability advantages.

### Software
Training employed JAX and ML Pathways, similar to the Gemini models.

## Evaluation Results

| Benchmark   | Metric | Gemma PT 9B |
|-------------|--------|-------------|
| MMLU        | 5-shot | 71.3        |
| HellaSwag   | 10-shot| 81.9        |
| HumanEval   | pass@1 | 40.2        |

## Ethics and Safety

Evaluations included content safety (CSAM, hate speech) and representational harms (WinBias, BBQ). Memorization and CBRN risks were also assessed. Results meet internal policies.

## Limitations

Models may reflect biases from training data. Factual accuracy is not guaranteed. Performance depends on prompt quality and context.
"""

    @pytest.fixture
    def sample_sections(self):
        """
        Fixture providing sample sections for testing.
        
        Returns:
            list: List of Section objects
        """
        return [
            Section("Machine Learning", "Machine learning is a subset of artificial intelligence.", 1, 3),
            Section("Neural Networks", "Neural networks are a key component in deep learning.", 5, 7),
            Section("Backpropagation", "Backpropagation is an algorithm used to train neural networks.", 9, 11)
        ]

    def test_initialization(self, test_logger):
        """
        Test initialization of QAMatchingEngine with actual components.
        
        Args:
            test_logger: Logger fixture for test execution tracking
        """
        test_logger.info("--- Starting test_initialization ---")
        start_time = time.time()
        
        try:
            engine = QAMatchingEngine(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
            
            assert engine.model_name == "sentence-transformers/all-MiniLM-L6-v2"
            assert engine.last_question_embeddings is None
            assert engine.model is not None
            assert engine.tokenizer is not None
            assert isinstance(engine.device, torch.device)
            # Check if markdown_parser is an instance of MarkdownParser
            assert isinstance(engine.markdown_parser, MarkdownParser)
            
            end_time = time.time()
            test_logger.info(f"QAMatchingEngine initialized successfully with model: {engine.model_name}")
            test_logger.info(f"Device used: {engine.device}")
            test_logger.info(f"Initialization test completed in {end_time - start_time:.4f} seconds")
        except Exception as e:
            test_logger.error(f"Initialization test failed: {str(e)}", exc_info=True)
            raise
        finally:
            test_logger.info("--- Finished test_initialization ---")

    def test_compute_similarity(self, engine, test_logger):
        """
        Test similarity computation between query and section embeddings.
        
        Args:
            engine: QAMatchingEngine fixture
            test_logger: Logger fixture for test execution tracking
        """
        test_logger.info("--- Starting test_compute_similarity ---")
        start_time = time.time()
        
        try:
            # Generate real embeddings
            texts = ["What is machine learning?", "What are neural networks?", "How does backpropagation work?"]
            test_logger.info(f"Computing similarity with {len(texts)} texts")
            
            with torch.no_grad():
                embeddings = engine._get_embeddings(texts)
            
            query_embedding = embeddings[0]
            section_embeddings = embeddings[1:]
            
            # Compute similarity
            similarities = engine._compute_similarity(query_embedding, section_embeddings)
            
            # Verify output shape and values
            assert similarities.shape == torch.Size([2])
            assert all(0 <= score <= 1 for score in similarities.tolist())
            
            end_time = time.time()
            test_logger.info(f"Similarity scores: {similarities.tolist()}")
            test_logger.info(f"Similarity computation completed in {end_time - start_time:.4f} seconds")
        except Exception as e:
            test_logger.error(f"Similarity computation test failed: {str(e)}", exc_info=True)
            raise
        finally:
            test_logger.info("--- Finished test_compute_similarity ---")

    def test_get_embeddings(self, engine, test_logger):
        """
        Test embedding generation with real model.
        
        Args:
            engine: QAMatchingEngine fixture
            test_logger: Logger fixture for test execution tracking
        """
        test_logger.info("--- Starting test_get_embeddings ---")
        start_time = time.time()
        
        try:
            texts = ["Text 1", "Text 2"]
            test_logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Generate embeddings
            embeddings = engine._get_embeddings(texts)
            
            # Verify output
            assert isinstance(embeddings, torch.Tensor)
            assert embeddings.shape[0] == len(texts)
            # Check embedding dimension based on the actual model used
            assert embeddings.shape[1] > 0
            
            end_time = time.time()
            test_logger.info(f"Successfully generated embeddings with shape: {embeddings.shape}")
            test_logger.info(f"Embedding generation completed in {end_time - start_time:.4f} seconds")
        except Exception as e:
            test_logger.error(f"Embedding generation test failed: {str(e)}", exc_info=True)
            raise
        finally:
            test_logger.info("--- Finished test_get_embeddings ---")

    def test_find_relevant_sections(self, engine, sample_context, test_logger):
        """
        Test finding relevant sections using distilled README-inspired context.

        Args:
            engine: QAMatchingEngine fixture
            sample_context: Distilled markdown content
            test_logger: Logger fixture for test execution tracking
        """
        test_logger.info("--- Starting test_find_relevant_sections ---")
        start_time = time.time()
        
        try:
            questions = [
                "What is Gemma?", # Expect Description
                "What data was used for training Gemma?", # Expect Training Data
                "How to run Gemma with float16 precision?", # Expect Precisions / Usage
                "Were ethical considerations evaluated?", # Expect Ethics and Safety
                "What hardware powered the training?", # Expect Hardware / Implementation
                "Which license is used for Gemma? A license is a legal document that grants permission to use a product or service." # Expect License info near top
            ]

            top_k = 3
            test_logger.info(f"Finding relevant sections for {len(questions)} questions with top_k={top_k}")
            
            results = engine.find_relevant_sections(questions, sample_context, top_k=top_k)

            # Verify results structure remains the same
            assert len(results) == len(questions)
            for idx, question_results in enumerate(results):
                test_logger.info(f"Question {idx+1}: '{questions[idx]}'")
                assert len(question_results) <= top_k
                assert len(question_results) > 0
                for section, score in question_results:
                    test_logger.info(f"  Section: Title='{section.title}', Score={score:.4f}")
                    test_logger.info(f"  Section: Content='{section.content}'")
                    assert isinstance(section, Section)
                    assert isinstance(score, float)
                    assert 0 <= score <= 1

            # --- Verify logical relevance by checking content keywords --- 

            # Helper to check if any section content for a question contains any keyword
            def check_content_keywords(question_index: int, keywords: List[str]) -> bool:
                section_contents = [r[0].content.lower() for r in results[question_index]]
                return any(any(kw in content for kw in keywords) for content in section_contents)

            # Q1: What is Gemma? -> Keywords: 'lightweight', 'google', 'gemini'
            assert check_content_keywords(0, ["lightweight", "google", "gemini"])
            test_logger.info("Question 1 (What is Gemma?) - Keywords check passed")

            # Q2: Training data? -> Keywords: 'trillion tokens', 'web documents', 'csam'
            assert check_content_keywords(1, ["trillion tokens", "web documents", "csam"])
            test_logger.info("Question 2 (Training data) - Keywords check passed")

            # Q3: GPU Precisions? -> Keywords: 'torch.float16', 'revision="float16"', 'bfloat16'
            assert check_content_keywords(2, ['torch.float16', 'revision="float16"', 'bfloat16'])
            test_logger.info("Question 3 (GPU Precisions) - Keywords check passed")

            # Q4: Ethical Considerations? -> Keywords: 'ethics', 'safety', 'csam', 'winbias', 'bbq'
            assert check_content_keywords(3, ["ethics", "safety", "csam", "winbias", "bbq"])
            test_logger.info("Question 4 (Ethical Considerations) - Keywords check passed")

            # Q5: Hardware? -> Keywords: 'tpu', 'tpuv5p', 'tensor processing unit'
            assert check_content_keywords(4, ["tpu", "tpuv5p", "tensor processing unit"])
            test_logger.info("Question 5 (Hardware) - Keywords check passed")

            # Q6: License? -> Keywords: 'license: gemma', '**license**: gemma'
            # Check specifically the first few sections as license info is usually at the top
            q6_top_contents = [r[0].content.lower() for r in results[5]]
            assert any("license" in content for content in q6_top_contents)
            test_logger.info("Question 6 (License) - Keywords check passed")
            
            end_time = time.time()
            test_logger.info(f"Finding relevant sections completed in {end_time - start_time:.4f} seconds")
        except Exception as e:
            test_logger.error(f"Finding relevant sections test failed: {str(e)}", exc_info=True)
            raise
        finally:
            test_logger.info("--- Finished test_find_relevant_sections ---")

    def test_get_best_context(self, engine, sample_context, test_logger):
        """
        Test getting best context using distilled README-inspired context.

        Args:
            engine: QAMatchingEngine fixture
            sample_context: Distilled markdown content
            test_logger: Logger fixture for test execution tracking
        """
        test_logger.info("--- Starting test_get_best_context ---")
        start_time = time.time()
        
        try:
            question = "What software framework like JAX was used for training?"
            test_logger.info(f"Finding best context for question: '{question}'")

            best_context_sections = engine.find_relevant_sections([question], sample_context, top_k=3)[0]
            best_context_combined = " \n---\n".join([section.content for section, score in best_context_sections])
            test_logger.info(f"Best context found:\n{best_context_combined}")

            assert isinstance(best_context_combined, str)
            assert len(best_context_combined) > 0
            # Check for keywords in the combined context content
            assert "jax" in best_context_combined.lower() or "ml pathways" in best_context_combined.lower()
            assert "bitsandbytes" not in best_context_combined.lower()
            test_logger.info("Context content verification passed")
            
            end_time = time.time()
            test_logger.info(f"Getting best context completed in {end_time - start_time:.4f} seconds")
        except Exception as e:
            test_logger.error(f"Getting best context test failed: {str(e)}", exc_info=True)
            raise
        finally:
            test_logger.info("--- Finished test_get_best_context ---")

    def test_batch_processing(self, engine, test_logger):
        """
        Test processing multiple texts in batches.
        
        Args:
            engine: QAMatchingEngine fixture
            test_logger: Logger fixture for test execution tracking
        """
        test_logger.info("--- Starting test_batch_processing ---")
        start_time = time.time()
        
        try:
            # Create a larger batch of texts to test batch processing
            texts = [f"Sample text {i}" for i in range(10)]
            test_logger.info(f"Processing a batch of {len(texts)} texts")
            
            # Generate embeddings
            embeddings = engine._get_embeddings(texts)
            
            # Verify output
            assert isinstance(embeddings, torch.Tensor)
            assert embeddings.shape[0] == len(texts)
            assert embeddings.shape[1] > 0
            test_logger.info(f"Successfully generated embeddings with shape: {embeddings.shape}")
            
            end_time = time.time()
            test_logger.info(f"Batch processing completed in {end_time - start_time:.4f} seconds")
        except Exception as e:
            test_logger.error(f"Batch processing test failed: {str(e)}", exc_info=True)
            raise
        finally:
            test_logger.info("--- Finished test_batch_processing ---")

    def test_empty_sections_handling(self, engine, test_logger):
        """
        Test handling of context that yields minimal sections from the real parser.
        
        Args:
            engine: QAMatchingEngine fixture
            test_logger: Logger fixture for test execution tracking
        """
        test_logger.info("--- Starting test_empty_sections_handling ---")
        start_time = time.time()
        
        try:
            questions = ["What is this about?"]
            # Context that might not have standard markdown headers
            context_without_headers = "This is a simple paragraph. It does not contain any markdown headers. Just plain text content."
            test_logger.info(f"Testing context without headers: '{context_without_headers[:50]}...'")
            
            # Use the real markdown parser
            results = engine.find_relevant_sections(questions, context_without_headers)
            
            test_logger.info(f"Number of question embeddings: {len(engine.last_question_embeddings)}")
            test_logger.info(f"Results: {results}")
                
            # Check assumptions based on MarkdownParser behavior:
            # If no headers, it might return the whole text as one section.
            assert len(results) == 1 # Results for the one question
            assert len(results[0]) == 1 # Should find one matching section (the default one)
            
            # Verify the section content and score
            found_section = results[0][0][0]
            found_score = results[0][0][1]
            
            test_logger.info(f"Found section title: {found_section.title}")
            test_logger.info(f"Found section content: {found_section.content}")
            test_logger.info(f"Match score: {found_score}")
            
            # The title might be None or default, content should be the whole text
            assert found_section.title is None or "Paragraph 1" in found_section.title # Adjust if MarkdownParser has different default
            assert found_section.content.strip() == context_without_headers.strip()
            test_logger.info("Section content verification passed")
            
            end_time = time.time()
            test_logger.info(f"Empty sections handling completed in {end_time - start_time:.4f} seconds")
        except Exception as e:
            test_logger.error(f"Empty sections handling test failed: {str(e)}", exc_info=True)
            raise
        finally:
            test_logger.info("--- Finished test_empty_sections_handling ---")
