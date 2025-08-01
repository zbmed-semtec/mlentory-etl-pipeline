import pytest
import pandas as pd
from datasets import Dataset
from unittest.mock import Mock, patch, MagicMock
from mlentory_extract.core.QAInferenceEngine import QAInferenceEngine, QAResult, QABatchResult


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
            "What hyperparameters were optimized?",
        ]
        contexts = [
            "The base model used is BERT.",
            "The evaluation metrics used were accuracy and F1 score.",
            "The hyperparameters optimized were learning rate and batch size.",
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
            "question_id": ["q1", "q2"],
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
        result = qa_engine.answer_single_question(
            question, "The weather is nice today."
        )
        assert result.confidence < 0.4
        
    def test_parse_multi_question_response(self, qa_engine: QAInferenceEngine) -> None:
        """
        Test the parsing of responses containing multiple question answers.
        
        Args:
            qa_engine (QAInferenceEngine): The inference engine instance
        """
        # Test standard format
        response = """
        Question 1: BERT
        Question 2: Accuracy and F1 score
        Question 3: Information not found
        """
        answers = qa_engine._parse_multi_question_response(response, 3)
        assert len(answers) == 3
        assert answers[0] == "BERT"
        assert answers[1] == "Accuracy and F1 score"
        assert answers[2] == "Information not found"
        
        # Test alternative format
        response = """
        Answer to Question 1: RoBERTa
        The model is based on RoBERTa architecture.
        
        Answer to Question 2: The model was trained on a dataset of 10M examples.
        
        Question 3. The training took 5 days on 8 GPUs.
        """
        answers = qa_engine._parse_multi_question_response(response, 3)
        assert len(answers) == 3
        assert "RoBERTa" in answers[0]
        assert "10M examples" in answers[1]
        assert "5 days" in answers[2]
        
        # Test handling of missing questions
        response = "Question 1: This is the only answer"
        answers = qa_engine._parse_multi_question_response(response, 3)
        assert len(answers) == 3
        assert answers[0] == "This is the only answer"
        assert answers[1] == "Information not found"
        assert answers[2] == "Information not found"
    
    @patch("transformers.pipeline")
    def test_batch_questions_single_context(self, mock_pipeline, qa_engine: QAInferenceEngine) -> None:
        """
        Test processing multiple questions with a single shared context.
        
        Args:
            mock_pipeline: Mock for the Hugging Face pipeline
            qa_engine (QAInferenceEngine): The inference engine instance
        """
        # Configure mock to return a formatted answer
        mock_output = [{
            "generated_text": """Context: This is a test context.

I have multiple questions about this context. For each question, provide a concise answer based ONLY on the information in the context above.
If the context does not contain the information needed to answer a question, respond with "Information not found" for that question.

Question 1: What is the model?
Question 2: What dataset was used?
Question 3: What is the performance?

For each question, provide your answer in the format:
Question [number]: [Your answer]

Question 1: The model is BERT
Question 2: Information not found
Question 3: The model achieved 95% accuracy"""
        }]
        qa_engine.pipeline = Mock(return_value=mock_output)
        
        questions = [
            "What is the model?",
            "What dataset was used?",
            "What is the performance?"
        ]
        context = "This is a test context."
        
        results = qa_engine.batch_questions_single_context(questions, context)
        
        assert len(results) == 3
        assert all(isinstance(result, QAResult) for result in results)
        assert results[0].answer == "The model is BERT"
        assert results[1].answer == "Information not found"
        assert results[2].answer == "The model achieved 95% accuracy"
        assert all("Batch Text Generation" in result.extraction_method for result in results)
    
    @patch("transformers.pipeline")
    def test_batch_grouped_inference(self, mock_pipeline, qa_engine: QAInferenceEngine) -> None:
        """
        Test the grouping and processing of questions by shared context.
        
        Args:
            mock_pipeline: Mock for the Hugging Face pipeline
            qa_engine (QAInferenceEngine): The inference engine instance
        """
        # Configure mock pipeline to return formatted answers
        def side_effect(prompt):
            if "Question 1: What is the model?\nQuestion 2: What is the performance?" in prompt:
                return [{
                    "generated_text": "Question 1: BERT\nQuestion 2: 95% accuracy"
                }]
            if "What are the training requirements?" in prompt:
                return [{
                    "generated_text": "8 GPUs required for training"
                }]
            return [{
                "generated_text": "Information not found"
            }]
            
        qa_engine.pipeline = Mock(side_effect=side_effect)
        
        # Test with mixed contexts - 2 questions share context A, 1 has context B
        questions = [
            "What is the model?",
            "What is the performance?",
            "What are the training requirements?"
        ]
        contexts = [
            "Context A",  # First two questions share this context
            "Context A",
            "Context B"   # This question has a different context
        ]
        
        # Mock the single question method too
        qa_engine.answer_single_question = Mock(
            return_value=QAResult(
                answer="8 GPUs required for training",
                extraction_time="2023-01-01_00-00-00"
            )
        )
        
        # Configure batch_questions_single_context mock
        qa_engine.batch_questions_single_context = Mock(
            return_value=[
                QAResult(answer="BERT", extraction_time="2023-01-01_00-00-00"),
                QAResult(answer="95% accuracy", extraction_time="2023-01-01_00-00-00")
            ]
        )
        
        results = qa_engine.batch_grouped_inference(questions, contexts, max_questions_per_group=5)
        
        # Verify results
        assert len(results) == 3
        # Check calls to batch_questions_single_context for the first context group
        qa_engine.batch_questions_single_context.assert_called_once_with(
            ["What is the model?", "What is the performance?"], 
            "Context A"
        )
        # Check calls to answer_single_question for the single question context
        qa_engine.answer_single_question.assert_called_once_with(
            "What are the training requirements?", 
            "Context B"
        )
    
    @patch("transformers.pipeline")
    def test_process_dataset_grouped(self, mock_pipeline, qa_engine: QAInferenceEngine) -> None:
        """
        Test processing a dataset with question grouping.
        
        Args:
            mock_pipeline: Mock for the Hugging Face pipeline
            qa_engine (QAInferenceEngine): The inference engine instance
        """
        # Create a test dataset with some shared contexts
        dataset_dict = {
            "question": [
                "What is the model architecture?",
                "What is the training dataset?",
                "What is the inference speed?",
                "What is the model size?"
            ],
            "context": [
                "Context A",  # These first two share the same context
                "Context A",
                "Context B",  # These next two share a different context
                "Context B"
            ],
            "metadata": ["meta1", "meta2", "meta3", "meta4"]
        }
        dataset = Dataset.from_dict(dataset_dict)
        
        # Mock the batch_grouped_inference method
        qa_engine.batch_grouped_inference = Mock(
            return_value=[
                QAResult(answer="BERT", extraction_time="2023-01-01", extraction_method="Batch 1/2"),
                QAResult(answer="Wikipedia", extraction_time="2023-01-01", extraction_method="Batch 2/2"),
                QAResult(answer="10ms", extraction_time="2023-01-01", extraction_method="Batch 1/2"),
                QAResult(answer="500MB", extraction_time="2023-01-01", extraction_method="Batch 2/2")
            ]
        )
        
        result_dataset = qa_engine.process_dataset_grouped(dataset, max_questions_per_group=2)
        
        # Verify the batch_grouped_inference was called correctly
        qa_engine.batch_grouped_inference.assert_called_once_with(
            questions=dataset["question"],
            contexts=dataset["context"],
            max_questions_per_group=2
        )
        
        # Check the result dataset
        assert isinstance(result_dataset, Dataset)
        assert len(result_dataset) == 4
        assert "answer" in result_dataset.features
        assert "extraction_time" in result_dataset.features
        assert "extraction_method" in result_dataset.features
        assert "metadata" in result_dataset.features
        
        # Verify the metadata was preserved
        assert result_dataset["metadata"] == ["meta1", "meta2", "meta3", "meta4"]
        
        # Verify answers were correctly assigned
        assert result_dataset["answer"] == ["BERT", "Wikipedia", "10ms", "500MB"]

    def test_integration_with_parser(self):
        """Test integration with ModelCardToSchemaParser using batch questions."""
        try:
            # Import required classes only in this test to avoid circular imports
            from mlentory_extract.core.ModelCardToSchemaParser import ModelCardToSchemaParser
            from mlentory_extract.core.QAMatchingEngine import QAMatchingEngine
            
            # Mock the matching engine and QA engine
            mock_matching_engine = MagicMock()
            mock_qa_engine = MagicMock()
            
            # Configure matching engine mock to return sample section matches
            mock_matching_engine.find_relevant_sections.return_value = [
                [(MagicMock(title="Section A", content="Content about model"), 0.9)],
                [(MagicMock(title="Section B", content="Content about training"), 0.8)]
            ]
            
            # Configure QA engine mock to return sample results
            mock_qa_engine.batch_questions_single_context.return_value = [
                QAResult(answer="BERT", extraction_time="2023-01-01", extraction_method="Test"),
                QAResult(answer="10 epochs", extraction_time="2023-01-01", extraction_method="Test")
            ]
            
            # Create parser with mocked components
            parser = ModelCardToSchemaParser()
            parser.matching_engine = mock_matching_engine
            parser.qa_engine = mock_qa_engine
            
            # Create a test DataFrame with one model
            df = pd.DataFrame({
                "modelId": ["test/model"],
                "card": ["This is a test model card with information about the model and training."],
                "tags": [["tag1", "tag2"]],
                "pipeline_tag": ["text-classification"],
                "author": ["Test Author"],
                "createdAt": [pd.Timestamp("2023-01-01")],
                "last_modified": [pd.Timestamp("2023-01-02")]
            })
            
            # Mock schema properties to generate test questions
            parser.schema_properties = {
                "fair4ml:modelArchitecture": {
                    "description": "The architecture of the model",
                    "HF_Readme_Section": "Model Architecture"
                },
                "fair4ml:trainingProcedure": {
                    "description": "How the model was trained",
                    "HF_Readme_Section": "Training"
                }
            }
            parser.processed_properties = []
            
            # Test the grouped extraction method
            result_df = parser.parse_fields_from_text_by_grouping_HF(df, max_questions_per_group=2)
            
            # Verify the matching engine was called
            mock_matching_engine.find_relevant_sections.assert_called_once()
            
            # Verify the QA engine was called with grouped questions
            mock_qa_engine.batch_questions_single_context.assert_called_once()
            
            # Verify results were added to DataFrame
            assert "fair4ml:modelArchitecture" in result_df.columns
            assert "fair4ml:trainingProcedure" in result_df.columns
            
            # Check extracted values
            architecture_data = result_df.loc[0, "fair4ml:modelArchitecture"]
            assert isinstance(architecture_data, list)
            assert architecture_data[0]["data"] == "BERT"
            
            training_data = result_df.loc[0, "fair4ml:trainingProcedure"]
            assert isinstance(training_data, list)
            assert training_data[0]["data"] == "10 epochs"
            
        except ImportError:
            pytest.skip("ModelCardToSchemaParser not available for testing")
