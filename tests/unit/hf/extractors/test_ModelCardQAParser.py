from datasets import load_dataset
from datetime import datetime
import pandas as pd
import pytest
from typing import List
import os

from extractors.core.ModelCardQAParser import ModelCardQAParser


class TestModelCardQAParser:
    """
    A class to test the ModelCardQAParser class
    """

    @pytest.fixture
    def parser_simple(self) -> ModelCardQAParser:
        """
        Create a fixture to create a ModelCardQAParser object

        Returns:
            ModelCardQAParser: A ModelCardQAParser object
        """
        # Create the parser object with test configuration
        parser = ModelCardQAParser(
            qa_model="Intel/dynamic_tinybert",
            questions=["What is the base model?", "Who created this model?"],
            tags_language=["en", "fr", "es"],
            tags_libraries=["pytorch", "tensorflow"],
            tags_other=["other-tag"],
            tags_task=["text-classification", "translation"]
        )
        return parser
    
    @pytest.fixture
    def parser_full(self) -> ModelCardQAParser:
        """
        Create an ModelCardQAParser with real questions and tags

        Returns:
            ModelCardQAParser: An ModelCardQAParser instance
        """
        def load_tsv_file_to_list(path: str) -> List[str]:
            return [val[0] for val in pd.read_csv(path, sep="\t").values.tolist()]
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print("Current directory:")
        print(current_dir)
        
        
        config_path = os.path.join(current_dir, "..", "..", "..", "config", "hf", "extract")  # Navigate up 3 levels and into configuration

        questions = load_tsv_file_to_list(os.path.join(config_path, "questions.tsv"))
        tags_language = load_tsv_file_to_list(os.path.join(config_path, "tags_language.tsv"))
        tags_libraries = load_tsv_file_to_list(os.path.join(config_path, "tags_libraries.tsv"))
        tags_other = load_tsv_file_to_list(os.path.join(config_path, "tags_other.tsv"))
        tags_task = load_tsv_file_to_list(os.path.join(config_path, "tags_task.tsv"))
        
        # Initialize extractor with configuration
        parser = ModelCardQAParser(
            qa_model="Intel/dynamic_tinybert",
            questions=questions,
            tags_language=tags_language,
            tags_libraries=tags_libraries,
            tags_other=tags_other,
            tags_task=tags_task,
        )
        
        return parser

    def add_base_questions(self, df: pd.DataFrame, parser_simple: ModelCardQAParser) -> pd.DataFrame:
        """
        Add base questions to a DataFrame

        Args:
            df (pd.DataFrame): The input DataFrame
            parser (ModelCardQAParser): The ModelCardQAParser object

        Returns:
            pd.DataFrame: The updated DataFrame with added base questions
        """
        new_columns = {
            f"q_id_{idx}": [None for _ in range(len(df))]
            for idx in range(len(parser_simple.questions))
        }
        return df.assign(**new_columns)

    def test_add_default_extraction_info(self, parser_simple: ModelCardQAParser) -> None:
        """
        Test the add_default_extraction_info method

        Args:
            parser (ModelCardQAParser): The ModelCardQAParser object
        """
        data = "This is some data"
        extraction_method = "parser_method"
        confidence = 0.95

        info_dict = parser_simple.add_default_extraction_info(data, extraction_method, confidence)

        assert isinstance(info_dict, dict)
        assert info_dict["data"] == data
        assert info_dict["extraction_method"] == extraction_method
        assert info_dict["confidence"] == confidence
        assert "extraction_time" in info_dict
    
    # Check that all the results are being added the extraction info

    def test_parse_known_fields_HF(self, parser_full: ModelCardQAParser) -> None:
        """
        Test the parse_known_fields_HF method

        Args:
            parser (ModelCardQAParser): The ModelCardQAParser object
        """
        # Mock DataFrame
        data = {
            "modelId": ["model1", "model2"],
            "author": ["author1", "author2"],
            "createdAt": ["2023-01-01", "2023-01-02"],
            "last_modified": ["2023-02-01", "2023-02-02"],
            "card": ["card text 1", "card text 2"],
            "tags": [["pytorch", "en"], ["tensorflow", "fr"]],
            "pipeline_tag": ["text-classification", "translation"]
        }
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser_full)

        # Process the DataFrame
        parsed_df = parser_full.parse_known_fields_HF(HF_df=mock_df.copy())

        # Verify the known fields are parsed correctly
        assert parsed_df.loc[0, "q_id_0"][0]["data"] == "model1"
        assert parsed_df.loc[0, "q_id_1"][0]["data"] == "author1"
        assert parsed_df.loc[0, "q_id_2"][0]["data"] == "2023-01-01"
        assert parsed_df.loc[0, "q_id_26"][0]["data"] == "2023-02-01"
        assert parsed_df.loc[0, "q_id_30"][0]["data"] == "card text 1"
        
    
    def test_parse_known_fields_HF_empty_dataframe_fails(
        self, parser_full: ModelCardQAParser
    ) -> None:
        """
        Test that parse_known_fields_HF raises a KeyError when given an empty DataFrame

        Args:
            parser (ModelCardQAParser): The ModelCardQAParser object
        """
        empty_df = pd.DataFrame()
        mock_df = self.add_base_questions(empty_df, parser_full)

        # Assert that the output is an empty DataFrame
        with pytest.raises(KeyError):
            parsed_df = parser_full.parse_known_fields_HF(HF_df=mock_df.copy())

    def test_parse_known_fields_HF_missing_columns_fails(
        self, parser_full: ModelCardQAParser
    ) -> None:
        """
        Test that parse_known_fields_HF raises a KeyError when given a DataFrame with missing columns

        Args:
            parser (ModelCardQAParser): The ModelCardQAParser object
        """
        data = {"author": ["a1", "a2"]}
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser_full)

        # Assert that the output is an empty DataFrame
        with pytest.raises(KeyError):
            parsed_df = parser_full.parse_known_fields_HF(HF_df=mock_df.copy())

    

    def test_parse_known_fields_HF_default_info(self, parser_full: ModelCardQAParser) -> None:
        """
        Test that parse_known_fields_HF adds the expected default info to the DataFrame

        Args:
            parser (ModelCardQAParser): The ModelCardQAParser object
        """
        data = {
            "modelId": ["nelson2424/gptj6b-FAQ-NelsMarketplace"],
            "author": ["a1"],
            "createdAt": ["c1"],
            "last_modified": ["lm1"],
            "card": ["c1"],
        }
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser_full)
        parsed_df = parser_full.parse_known_fields_HF(HF_df=mock_df.copy())

        # Assert that all new columns have the expected info dictionary
        for col in ["q_id_0", "q_id_1", "q_id_2", "q_id_26", "q_id_29", "q_id_30"]:
            assert (
                parsed_df.loc[0, col][0]["extraction_method"]
                == "Parsed_from_HF_dataset"
            )
            assert parsed_df.loc[0, col][0]["confidence"] == 1.0

    def test_parse_fields_from_tags_HF(self, parser_full: ModelCardQAParser) -> None:
        """
        Test the parse_fields_from_tags_HF method

        Args:
            parser (ModelCardQAParser): The ModelCardQAParser object
        """
        # Mock DataFrame with tags
        data = {
            "modelId": ["model1"],
            "tags": [["pytorch", "en", "dataset:dataset1", "license:MIT", "arxiv:1234.5678"]],
            "pipeline_tag": ["text-classification"],
            "card": ["card text"]
        }
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser_full)

        # Process the DataFrame
        parsed_df = parser_full.parse_fields_from_tags_HF(HF_df=mock_df.copy())

        # Verify tag parsing
        assert "pytorch" in parsed_df.loc[0, "q_id_17"][0]["data"]
        assert "en" in parsed_df.loc[0, "q_id_16"][0]["data"]
        assert "dataset1" in parsed_df.loc[0, "q_id_4"][0]["data"]
        assert "MIT" in parsed_df.loc[0, "q_id_15"][0]["data"]
        assert "1234.5678" in parsed_df.loc[0, "q_id_13"][0]["data"]
        assert "text classification" in parsed_df.loc[0, "q_id_3"][0]["data"]
    
    def test_parse_fields_from_tags_one_model_all_tags(
        self, parser_full: ModelCardQAParser
    ) -> None:
        """
        Test that parse_fields_from_tags_HF handles one model with all tags correctly

        Args:
            parser (ModelCardQAParser): The ModelCardQAParser object
        """
        data = {
            "modelId": ["EleutherAI/gpt-j-6b"],
            "tags": [
                [
                    "Image-Text-to-Text",
                    "dataset:dataset1",
                    "arxiv:paper1",
                    "license:license1",
                    "en",
                    "PyTorch",
                ]
            ],
            "pipeline_tag": "Text-to-Video",
        }

        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser_full)

        # Call the function
        parsed_df = parser_full.parse_fields_from_tags_HF(HF_df=mock_df.copy())

        # Assert the output (check for existence of new columns and data types)
        assert all(
            col in parsed_df.columns
            for col in ["q_id_3", "q_id_4", "q_id_13", "q_id_15", "q_id_16", "q_id_17"]
        )
        assert parsed_df.loc[0, "q_id_3"][0]["data"] == [
            "image text to text",
            "text to video",
        ]
        assert parsed_df.loc[0, "q_id_4"][0]["data"] == ["dataset1"]
        assert parsed_df.loc[0, "q_id_13"][0]["data"] == ["paper1"]
        assert parsed_df.loc[0, "q_id_15"][0]["data"] == ["license1"]
        assert parsed_df.loc[0, "q_id_16"][0]["data"] == ["en"]
        assert parsed_df.loc[0, "q_id_17"][0]["data"] == ["pytorch"]

    def test_parse_fields_from_tags_two_models_all_tags(
        self, parser_full: ModelCardQAParser
    ) -> None:
        """
        Test that parse_fields_from_tags_HF correctly parses tags for two models with all tags

        Args:
            parser (ModelCardQAParser): The ModelCardQAParser object
        """
        data = {
            "modelId": ["m1", "m2"],
            "tags": [
                [
                    "Image-Text-to-Text",
                    "dataset:dataset1",
                    "arxiv:paper1",
                    "license:license1",
                    "en",
                    "PyTorch",
                ],
                [
                    "Image Feature Extraction",
                    "dataset:dataset2",
                    "arxiv:paper2",
                    "license:license2",
                    "sv",
                    "PEFT",
                ],
            ],
            "pipeline_tag": ["Text To Video", "Text-to-Text"],
        }

        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser_full)

        # Call the function to parse tags
        parsed_df = parser_full.parse_fields_from_tags_HF(HF_df=mock_df.copy())

        # Assert the output (check for existence of new columns and data types)
        assert all(
            col in parsed_df.columns
            for col in ["q_id_3", "q_id_4", "q_id_13", "q_id_15", "q_id_16", "q_id_17"]
        )
        assert parsed_df.loc[0, "q_id_3"][0]["data"] == [
            "image text to text",
            "text to video",
        ]
        assert parsed_df.loc[0, "q_id_4"][0]["data"] == ["dataset1"]
        assert parsed_df.loc[0, "q_id_13"][0]["data"] == ["paper1"]
        assert parsed_df.loc[0, "q_id_15"][0]["data"] == ["license1"]
        assert parsed_df.loc[0, "q_id_16"][0]["data"] == ["en"]
        assert parsed_df.loc[0, "q_id_17"][0]["data"] == ["pytorch"]
        assert parsed_df.loc[1, "q_id_3"][0]["data"] == [
            "image feature extraction",
            "text to text",
        ]
        assert parsed_df.loc[1, "q_id_4"][0]["data"] == ["dataset2"]
        assert parsed_df.loc[1, "q_id_13"][0]["data"] == ["paper2"]
        assert parsed_df.loc[1, "q_id_15"][0]["data"] == ["license2"]
        assert parsed_df.loc[1, "q_id_16"][0]["data"] == ["sv"]
        assert parsed_df.loc[1, "q_id_17"][0]["data"] == ["peft"]

    def test_parse_fields_from_tags_no_correct_tags(
        self, parser_full: ModelCardQAParser
    ) -> None:
        """
        Test that parse_fields_from_tags_HF correctly handles no correct tags

        Args:
            parser (ModelCardQAParser): The ModelCardQAParser object
        """
        
        data = {
            "modelId": ["m1"],
            "tags": [
                [
                    "Image-Text-to-Texta",
                    "dataset2:dataset1",
                    "arxiv3:paper1",
                    "license4:license1",
                    "enas",
                    "PyTorch3",
                ]
            ],
            "pipeline_tag": None,
        }

        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser_full)

        # Call the function to parse tags
        parsed_df = parser_full.parse_fields_from_tags_HF(HF_df=mock_df.copy())

        # Assert the output (check for existence of new columns and data types)
        assert all(
            col in parsed_df.columns
            for col in ["q_id_3", "q_id_4", "q_id_13", "q_id_15", "q_id_16", "q_id_17"]
        )
        assert pd.isna(parsed_df.loc[0, "q_id_3"][0]["data"])
        assert pd.isna(parsed_df.loc[0, "q_id_4"][0]["data"])
        assert pd.isna(parsed_df.loc[0, "q_id_13"][0]["data"])
        assert pd.isna(parsed_df.loc[0, "q_id_15"][0]["data"])
        assert pd.isna(parsed_df.loc[0, "q_id_16"][0]["data"])
        assert pd.isna(parsed_df.loc[0, "q_id_17"][0]["data"])

    def test_answer_question(self, parser_full: ModelCardQAParser) -> None:
        """
        Test the answer_question method

        Args:
            parser (ModelCardQAParser): The ModelCardQAParser object
        """
        question = "What is the base model?"
        context = "This model is based on BERT."

        answer = parser_full.answer_question(question, context)

        assert isinstance(answer, list)
        assert len(answer) == 1
        assert isinstance(answer[0], dict)
        assert "data" in answer[0]
        assert "confidence" in answer[0]
        assert "extraction_method" in answer[0]
        assert "extraction_time" in answer[0]
        assert answer[0]["data"] == "BERT"
        assert answer[0]["confidence"] > 0.5
    
    def test_answer_multiple_questions(self, parser_full: ModelCardQAParser):
        """
        Tests the answer_question method of the parser class for various scenarios.

        Args:
            parser: An instance of the parser class.
        """

        # Test case 1: Simple question
        question = "What model is used as the base model?"
        context = "The base model used is BERT."
        expected_answer = {
            "data": "BERT",
            "extraction_method": "Intel/dynamic_tinybert",
            "confidence": 0.9,
        }
        result = parser_full.answer_question(question, context)
        assert result[0]["data"] == expected_answer["data"]
        assert result[0]["confidence"] >= 0.5

        # Test case 2: Question with multiple possible answers
        question = "What evaluation metrics were used?"
        context = "The evaluation metrics used were accuracy and F1 score."
        expected_answer = {
            "data": "accuracy and F1 score",
            "extraction_method": "Intel/dynamic_tinybert",
            "confidence": 0.8,
        }
        result = parser_full.answer_question(question, context)
        assert result[0]["data"] == expected_answer["data"]
        assert result[0]["confidence"] >= 0.5

        # Test case 3: Question with no clear answer
        question = "What is the meaning of life?"
        context = "The lakers won yesterday"
        expected_answer = [
            {
                "data": "",
                "extraction_method": "Intel/dynamic_tinybert",
                "confidence": 0.0,
            }
        ]
        result = parser_full.answer_question(question, context)
        assert result[0]["confidence"] < 0.4

    def test_parse_fields_from_txt_HF(self, parser_full: ModelCardQAParser) -> None:
        """
        Test the correct fucntionality of the parse_fields_from_txt_HF method

        Args:
            parser (ModelCardQAParser): The ModelCardQAParser object
        """
        data = {
            "modelId": ["m1"],
            "card": """
                        The base model used is BERT.The evaluation metrics used were accuracy and F1 score.
                        The hyperparameters optimized during the training process were learning rate and batch size.
                        """,
        }
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser_full)
        parsed_df = parser_full.parse_fields_from_txt_HF(HF_df=mock_df)
        print(parsed_df.loc[0, "q_id_18"])
        assert parsed_df.loc[0, "q_id_8"][0]["data"] == "BERT"
        assert parsed_df.loc[0, "q_id_8"][0]["confidence"] > 0.8
        assert parsed_df.loc[0, "q_id_9"][0]["data"] == "accuracy and F1 score"
        assert parsed_df.loc[0, "q_id_9"][0]["confidence"] > 0.8
        assert parsed_df.loc[0, "q_id_11"][0]["data"] == "learning rate and batch size"
        assert parsed_df.loc[0, "q_id_11"][0]["confidence"] > 0.8
        assert parsed_df.loc[0, "q_id_5"][0]["confidence"] < 0.4
    
    # def test_parse_known_fields_HF_finetuned_model(
    #     self, parser_full: ModelCardQAParser
    # ) -> None:
    #     """
    #     Test that parse_known_fields_HF handles finetuned models correctly

    #     Args:
    #         parser (ModelCardQAParser): The ModelCardQAParser object
    #     """
    #     data = {
    #         "modelId": [
    #             "EleutherAI/gpt-j-6b",
    #             "EleutherAI/gpt-j-6b",
    #             "EleutherAI/gpt-j-6b",
    #         ],
    #         "author": ["a1", "a2", "a3"],
    #         "createdAt": ["c1", "c2", "c3"],
    #         "last_modified": ["lm1", "lm2", "lm3"],
    #         "card": ["c1", "c2", "c3"],
    #     }

    #     mock_df = pd.DataFrame(data)
    #     mock_df = self.add_base_questions(mock_df, parser_full)

    #     # Set up mock data for testing
    #     mock_df.loc[0, "q_id_8"] = "answer"
    #     mock_df.loc[0, "q_id_4"] = "answer"

    #     mock_df.loc[1, "q_id_8"] = "[CLS]"
    #     mock_df.loc[1, "q_id_4"] = "answer"

    #     mock_df.loc[2, "q_id_8"] = None
    #     mock_df.loc[2, "q_id_4"] = "answer"

    #     print(mock_df[["q_id_4", "q_id_6", "q_id_7", "q_id_8"]])

    #     parsed_df = parser_full.parse_known_fields_HF(HF_df=mock_df.copy())

    #     # Assert that q_id_6 and q_id_7 have the value from q_id_4 for the second row only
    #     assert parsed_df.loc[0, "q_id_6"][0]["data"] == "answer"
    #     assert parsed_df.loc[0, "q_id_7"][0]["data"] == "answer"
    #     assert pd.isna(parsed_df.loc[1, "q_id_6"][0]["data"])
    #     assert pd.isna(parsed_df.loc[1, "q_id_7"][0]["data"])
    #     assert pd.isna(parsed_df.loc[2, "q_id_6"][0]["data"])
    #     assert pd.isna(parsed_df.loc[2, "q_id_7"][0]["data"])