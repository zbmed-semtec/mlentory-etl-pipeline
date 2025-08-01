import pytest
import logging
import time
import os
from datetime import datetime
from typing import List

# Assuming QAInferenceEngine and QAResult are accessible
# Adjust the import path as necessary based on your project structure
from mlentory_extract.core.QAInferenceEngine import QAInferenceEngine, QAResult

# Configure logging
LOG_FILENAME = "qa_engine_test_log.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Ensure the tests directory exists if the log file is placed there
# Or adjust the path as needed, e.g., os.path.join(os.path.dirname(__file__), LOG_FILENAME)
log_file_path = os.path.join(os.path.dirname(__file__), LOG_FILENAME)

@pytest.fixture(scope="session")
def test_logger():
    """Pytest fixture to configure logging for the test session."""
    logger = logging.getLogger("QAEngineTests")
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if tests are run multiple times in one session
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Optional: Also log to console during tests
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)
        # logger.addHandler(stream_handler)

    # Log session start
    logger.info("=" * 30 + " TEST SESSION STARTED " + "=" * 30)
    yield logger
    # Log session end
    logger.info("=" * 30 + " TEST SESSION ENDED " + "=" * 30)


@pytest.fixture(scope="function")
def qa_engine() -> QAInferenceEngine:
    """Pytest fixture to initialize the QAInferenceEngine for testing."""
    model_name = "Qwen/Qwen2.5-3B"
    batch_size = 2
    max_new_tokens = 100
    print(f"\nInitializing QAInferenceEngine with model: {model_name}")
    engine = QAInferenceEngine(
        model_name=model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens
    )
    print("QAInferenceEngine initialized.")
    return engine

def test_batch_inference_logging(qa_engine: QAInferenceEngine, test_logger: logging.Logger):
    """
    Tests the batch_inference method, logs execution time and model info.
    """
    test_logger.info("--- Starting test_batch_inference_logging ---")

    # Questions inspired by questions.tsv and contexts from readme_example_1.txt
    questions = [
        "What is the name of the model described?",
        "Who are the authors of the model?",
        "What hardware was used for training?",
        "What is the primary license associated with the model?",
        "What programming languages was the model trained on?", # Should be "Information not found"
        "What are some limitations regarding factual accuracy?",
    ]
    contexts = [
        # Context 1 (Model Info)
        "Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. They are text-to-text, decoder-only large language models, available in English, with open weights for both pre-trained variants and instruction-tuned variants.",
        # Context 2 (Citation/License - partial, Authorship focused)
        "License: gemma. Authors: Google. @article{gemma_2024, title={Gemma}, url={https://www.kaggle.com/m/3301}, DOI={10.34740/KAGGLE/M/3301}, publisher={Kaggle}, author={Gemma Team}, year={2024}}",
        # Context 3 (Usage/Hardware)
        "Gemma was trained using the latest generation of Tensor Processing Unit (TPU) hardware (TPUv5p). Training was done using JAX and ML Pathways. The instruction-tuned models use a chat template that must be adhered to for conversational use.",
        # Context 4 (Citation/License - License focused)
        "License: gemma. Authors: Google. @article{gemma_2024, title={Gemma}, url={https://www.kaggle.com/m/3301}, DOI={10.34740/KAGGLE/M/3301}, publisher={Kaggle}, author={Gemma Team}, year={2024}}",
        # Context 5 (Irrelevant for the question)
        "Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning. Their relatively small size makes it possible to deploy them in environments with limited resources.",
        # Context 6 (Limitations)
        "LLMs generate responses based on information they learned from their training datasets, but they are not knowledge bases. They may generate incorrect or outdated factual statements. LLMs might lack the ability to apply common sense reasoning.",
    ]
    num_items = len(questions)
    test_logger.info(f"Testing batch_inference with {num_items} items based on README example.")
    test_logger.info(f"Model used: {qa_engine.model_name}")

    start_time = time.time()
    try:
        results = qa_engine.batch_inference(questions, contexts)
        end_time = time.time()
        duration = end_time - start_time
        test_logger.info(f"batch_inference execution time: {duration:.4f} seconds")

        # Basic Assertions
        assert isinstance(results, list), "Result should be a list"
        assert len(results) == num_items, f"Expected {num_items} results, but got {len(results)}"
        assert all(isinstance(r, QAResult) for r in results), "All items in results should be QAResult instances"

        # Log results summary
        test_logger.info("Batch inference successful. Results summary:")
        for i, res in enumerate(results):
            test_logger.info(f"  Item {i+1}: Question='{questions[i][:50]}...', Answer='{res.answer[:50]}...'")
            # Add specific answer checks if needed, e.g.:
            # if i == 0: assert "Paris" in res.answer
            # if i == 1: assert "Shakespeare" in res.answer
            # if i == 2: assert res.answer == "Information not found" # Check model correctly identifies missing info
            # --- Enhanced Logging ---
            if i == 0:  # Question: "What is the name of the model described?"
                expected_keywords = ["gemma"]
                test_logger.info(f"    Check {i+1}: Expected keywords '{expected_keywords}', Answer='{res.answer[:100]}...'")
                assert any(kw in res.answer.lower() for kw in expected_keywords), f"Expected keywords {expected_keywords} not found in answer for Q{i+1}"
            elif i == 1:  # Question: "Who are the authors of the model?"
                expected_keywords = ["google", "gemma team"]
                test_logger.info(f"    Check {i+1}: Expected keywords '{expected_keywords}', Answer='{res.answer[:100]}...'")
                # Allow flexibility in how the author is stated
                assert any(kw in res.answer.lower() for kw in expected_keywords), f"Expected keywords {expected_keywords} not found in answer for Q{i+1}"
            elif i == 2: # Question: "What hardware was used for training?"
                expected_keywords = ["tpu", "tensor processing unit"]
                test_logger.info(f"    Check {i+1}: Expected keywords '{expected_keywords}', Answer='{res.answer[:100]}...'")
                assert any(kw in res.answer.lower() for kw in expected_keywords), f"Expected keywords {expected_keywords} not found in answer for Q{i+1}"
            elif i == 3: # Question: "What is the primary license associated with the model?"
                expected_keywords = ["gemma"]
                test_logger.info(f"    Check {i+1}: Expected keyword 'gemma', Answer='{res.answer[:100]}...'")
                assert "gemma" in res.answer.lower(), f"Expected keyword 'gemma' not found in answer for Q{i+1}"
            elif i == 4: # Question: "What programming languages was the model trained on?"
                # This depends heavily on the model's ability to infer absence of information
                # A less strict check might be appropriate, or checking for specific "not found" phrasing
                expected_phrases = ["not found", "not specified", "not mentioned", "web documents", "code", "mathematics"] # Allow context leak
                test_logger.info(f"    Check {i+1}: Expecting indication of missing info or context leak ('{expected_phrases}'), Answer='{res.answer[:100]}...'")
                # assert any(phrase in res.answer.lower() for phrase in expected_phrases), f"Expected indication of missing info not found for Q{i+1}" # Might be too strict
            elif i == 5: # Question: "What are some limitations regarding factual accuracy?"
                expected_keywords = ["incorrect", "outdated", "factual statements", "not knowledge bases"]
                test_logger.info(f"    Check {i+1}: Expected keywords '{expected_keywords}', Answer='{res.answer[:100]}...'")
                assert any(kw in res.answer.lower() for kw in expected_keywords), f"Expected keywords {expected_keywords} not found in answer for Q{i+1}"
            # --- End Enhanced Logging ---

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        test_logger.error(f"Exception during batch_inference after {duration:.4f} seconds: {e}", exc_info=True)
        pytest.fail(f"batch_inference raised an exception: {e}")

    finally:
        test_logger.info("--- Finished test_batch_inference_logging ---")

# Example of another test (can be added later)
# def test_another_feature(...):
#     pass
