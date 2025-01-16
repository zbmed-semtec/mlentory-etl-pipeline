import pytest
import pandas as pd
from datasets import Dataset
from unittest.mock import Mock
from mlentory_extract.core.QAInferenceEngine import QAInferenceEngine, QAResult


class TestQAInferenceEngine:
    """Test suite for QAInferenceEngine class"""

    @pytest.fixture
    def qa_engine(self) -> QAInferenceEngine:
        """
        Creates a QAInferenceEngine instance for testing.

        Returns:
            QAInferenceEngine: Configured inference engine
        """
        return QAInferenceEngine(model_name="Intel/dynamic_tinybert")

    def test_answer_single_question(self, qa_engine: QAInferenceEngine) -> None:
        """
        Test answering a single question with the inference engine.

        Args:
            qa_engine (QAInferenceEngine): The inference engine instance
        """
        question = "What is the base model?"
        context = "This model is based on BERT."

        result = qa_engine.answer_single_question(question, context)

        assert isinstance(result, QAResult)
        assert isinstance(result.answer, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.extraction_time, str)
        assert result.answer == "BERT"
        assert result.confidence > 0.5
        assert result.extraction_method == "Pipeline"

    def test_batch_inference(self, qa_engine: QAInferenceEngine) -> None:
        """
        Test processing multiple questions in batch.

        Args:
            qa_engine (QAInferenceEngine): The inference engine instance
        """
        questions = [
            "What is the base model?",
            "What metrics were used?",
            "What hyperparameters were optimized?"
        ]
        contexts = [
            "The base model used is BERT.",
            "The evaluation metrics used were accuracy and F1 score.",
            "The hyperparameters optimized were learning rate and batch size."
        ]

        results = qa_engine.batch_inference(questions, contexts)

        assert len(results) == 3
        assert all(isinstance(result, QAResult) for result in results)
        assert results[0].answer == "BERT"
        assert results[1].answer == "accuracy and F1 score"
        assert results[2].answer == "learning rate and batch size"
        assert all(result.confidence > 0.5 for result in results)

    def test_process_dataset(self, qa_engine: QAInferenceEngine) -> None:
        """
        Test processing a HuggingFace dataset.

        Args:
            qa_engine (QAInferenceEngine): The inference engine instance
        """
        # Create a test dataset
        dataset_dict = {
            "question": ["What model?", "What metrics?"],
            "context": ["Model is BERT.", "Metrics are accuracy and F1."],
            "row_index": [0, 1],
            "question_id": ["q1", "q2"]
        }
        dataset = Dataset.from_dict(dataset_dict)

        result_dataset = qa_engine.process_dataset(dataset)

        assert isinstance(result_dataset, Dataset)
        assert "answer" in result_dataset.features
        assert "score" in result_dataset.features
        assert "extraction_time" in result_dataset.features
        assert len(result_dataset) == 2
        
        # Check first result
        assert result_dataset[0]["answer"] == "BERT"
        assert result_dataset[0]["score"] > 0.5
        
        # Check second result
        assert "accuracy" in result_dataset[1]["answer"]
        assert result_dataset[1]["score"] > 0.5

    def test_empty_context_handling(self, qa_engine: QAInferenceEngine) -> None:
        """
        Test handling of empty or invalid context.

        Args:
            qa_engine (QAInferenceEngine): The inference engine instance
        """
        question = "What is the second president of the United States?"
        
        # Test empty context
        with pytest.raises(ValueError, match="context"):
            qa_engine.answer_single_question(question, "")

        # Test with very irrelevant context
        result = qa_engine.answer_single_question(question, "The weather is nice today.")
        assert result.confidence < 0.4
