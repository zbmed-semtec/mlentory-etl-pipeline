from datasets import load_dataset
from datetime import datetime
import pandas as pd
import os
import pytest
import sys

print(os.getcwd())
sys.path.append('.')
from Extractors.HF_Extractor.Core.MetadataParser import MetadataParser


class TestMetadataParser:
    """
    A class to test the MetadataParser class
    """
    
    @pytest.fixture
    def parser(self) -> MetadataParser:
        """
        Create a fixture to create a MetadataParser object
        
        Returns:
            MetadataParser: A MetadataParser object
        """
        # Create the parser object that will perform the transformations on the raw data
        parser = MetadataParser(qa_model="Intel/dynamic_tinybert", path_to_config_data="./Config_Data")
        return parser
    
    def add_base_questions(self, df: pd.DataFrame, parser: MetadataParser) -> pd.DataFrame:
        """
        Add base questions to a DataFrame
        
        Args:
            df (pd.DataFrame): The input DataFrame
            parser (MetadataParser): The MetadataParser object
            
        Returns:
            pd.DataFrame: The updated DataFrame with added base questions
        """
        new_columns = {}
        
        for idx in range(len(parser.questions)):
            q_id = "q_id_" + str(idx)
            new_columns[q_id] = [None for _ in range(len(df))]
        
        df = df.assign(**new_columns)
        
        return df
    
    def test_add_default_extraction_info(self, parser: MetadataParser) -> None:
        """
        Test the add_default_extraction_info method
        
        Args:
            parser (MetadataParser): The MetadataParser object
        """
        data = "This is some data"
        extraction_method = "parser_method"
        confidence = 0.95
        
        # Call the function
        info_dict = parser.add_default_extraction_info(data, extraction_method, confidence)
        
        # Assert the output
        assert info_dict == {"data": data, "extraction_method": extraction_method, "confidence": confidence}

    def test_parse_known_fields_HF(self, parser: MetadataParser) -> None:
        """
        Test the parse_known_fields_HF method
        
        Args:
            parser (MetadataParser): The MetadataParser object
        """
        # Mock DataFrame
        data = {"modelId": ["nelson2424/gptj6b-FAQ-NelsMarketplace","nelson2424/gptj6b-FAQ-NelsMarketplace"], 
                "author": ["a1","a2"],
                "createdAt": ["c1","c2"],
                "last_modified": ["lm1","lm2"]}
        
        mock_df = pd.DataFrame(data)
        
        mock_df = self.add_base_questions(mock_df, parser)
        
        # Call the function
        parsed_df = parser.parse_known_fields_HF(HF_df=mock_df.copy())
        
        # Assert the output (check for existence of new columns and data types)
        assert all(col in parsed_df.columns for col in ["q_id_0", "q_id_1", "q_id_2", "q_id_26", "q_id_29"])
        assert parsed_df["q_id_0"].dtype == object
        assert parsed_df["q_id_1"].dtype == object
        assert parsed_df["q_id_2"].dtype == object
        assert parsed_df["q_id_26"].dtype == object
        assert parsed_df["q_id_29"].dtype == object
    
    def test_parse_known_fields_HF_empty_dataframe_fails(self, parser: MetadataParser) -> None:
        """
        Test that parse_known_fields_HF raises a KeyError when given an empty DataFrame
        
        Args:
            parser (MetadataParser): The MetadataParser object
        """
        empty_df = pd.DataFrame()
        mock_df = self.add_base_questions(empty_df, parser)
        
        # Assert that the output is an empty DataFrame
        with pytest.raises(KeyError):
            parsed_df = parser.parse_known_fields_HF(HF_df=mock_df.copy())


    def test_parse_known_fields_HF_missing_columns_fails(self, parser: MetadataParser) -> None:
        """
        Test that parse_known_fields_HF raises a KeyError when given a DataFrame with missing columns
        
        Args:
            parser (MetadataParser): The MetadataParser object
        """
        data = {"author": ["a1", "a2"]}
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser)
        
        # Assert that the output is an empty DataFrame
        with pytest.raises(KeyError):
            parsed_df = parser.parse_known_fields_HF(HF_df=mock_df.copy())

    def test_parse_known_fields_HF_finetuned_model(self, parser: MetadataParser) -> None:
        """
        Test that parse_known_fields_HF handles finetuned models correctly
        
        Args:
            parser (MetadataParser): The MetadataParser object
        """
        data = {"modelId": ["EleutherAI/gpt-j-6b", "EleutherAI/gpt-j-6b", "EleutherAI/gpt-j-6b"],
                "author": ["a1", "a2", "a3"],
                "createdAt": ["c1", "c2", "c3"], 
                "last_modified": ["lm1","lm2", "lm3"]}
        
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser)
        
        # Set up mock data for testing
        mock_df.loc[0, "q_id_8"] = "answer"
        mock_df.loc[0, "q_id_4"] = "answer"
        
        mock_df.loc[1, "q_id_8"] = "[CLS]"
        mock_df.loc[1, "q_id_4"] = "answer"
        
        mock_df.loc[2, "q_id_8"] = None
        mock_df.loc[2, "q_id_4"] = "answer"
        
        print(mock_df)
        
        parsed_df = parser.parse_known_fields_HF(HF_df=mock_df.copy())
        
        # Assert that q_id_6 and q_id_7 have the value from q_id_4 for the second row only
        assert parsed_df.loc[0, "q_id_6"][0]["data"] == "answer"
        assert parsed_df.loc[0, "q_id_7"][0]["data"] == "answer"
        assert pd.isna(parsed_df.loc[1, "q_id_6"][0]["data"])
        assert pd.isna(parsed_df.loc[1, "q_id_7"][0]["data"])
        assert pd.isna(parsed_df.loc[2, "q_id_6"][0]["data"]) 
        assert pd.isna(parsed_df.loc[2, "q_id_7"][0]["data"])

    def test_parse_known_fields_HF_default_info(self, parser: MetadataParser) -> None:
        """
        Test that parse_known_fields_HF adds the expected default info to the DataFrame
        
        Args:
            parser (MetadataParser): The MetadataParser object
        """
        data = {"modelId": ["nelson2424/gptj6b-FAQ-NelsMarketplace"], "author": ["a1"], "createdAt": ["c1"], "last_modified": ["lm1"]}
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser)
        parsed_df = parser.parse_known_fields_HF(HF_df=mock_df.copy())
        
        # Assert that all new columns have the expected info dictionary
        for col in ["q_id_0", "q_id_1", "q_id_2", "q_id_26","q_id_29"]:
            assert parsed_df.loc[0, col][0]["extraction_method"] == "Parsed_from_HF_dataset"
            assert parsed_df.loc[0, col][0]["confidence"] == 1.0

    def test_parse_fields_from_tags_one_model_all_tags(self, parser: MetadataParser) -> None:
        """
        Test that parse_fields_from_tags_HF handles one model with all tags correctly
        
        Args:
            parser (MetadataParser): The MetadataParser object
        """
        data = {"modelId": ["EleutherAI/gpt-j-6b"], 
                "tags": [["Image-Text-to-Text",
                        "dataset:dataset1",
                        "arxiv:paper1",
                        "license:license1",
                        "en",
                        "PyTorch"]],
                "pipeline_tag": "Text-to-Video"}
        
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser)
        
        # Call the function
        parsed_df = parser.parse_fields_from_tags_HF(HF_df=mock_df.copy())
        
        # Assert the output (check for existence of new columns and data types)
        assert all(col in parsed_df.columns for col in ["q_id_3", "q_id_4", "q_id_13", "q_id_15", "q_id_16", "q_id_17"])
        assert parsed_df.loc[0, "q_id_3"][0]["data"] == ["image text to text", "text to video"]
        assert parsed_df.loc[0, "q_id_4"][0]["data"] == ["dataset1"]
        assert parsed_df.loc[0, "q_id_13"][0]["data"] == ["paper1"]
        assert parsed_df.loc[0, "q_id_15"][0]["data"] == ["license1"]
        assert parsed_df.loc[0, "q_id_16"][0]["data"] == ["en"]
        assert parsed_df.loc[0, "q_id_17"][0]["data"] == ["pytorch"]
    
    def test_parse_fields_from_tags_two_models_all_tags(self, parser: MetadataParser) -> None:
        """
        Test that parse_fields_from_tags_HF correctly parses tags for two models with all tags
        
        Args:
            parser (MetadataParser): The MetadataParser object
        """
        data = {"modelId": ["m1", "m2"], 
                "tags": [["Image-Text-to-Text",
                        "dataset:dataset1",
                        "arxiv:paper1",
                        "license:license1",
                        "en",
                        "PyTorch"],
                        ["Image Feature Extraction",
                        "dataset:dataset2",
                        "arxiv:paper2",
                        "license:license2",
                        "sv",
                        "PEFT"]],
                "pipeline_tag": ["Text To Video", "Text-to-Text"]}
        
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser)
        
        # Call the function to parse tags
        parsed_df = parser.parse_fields_from_tags_HF(HF_df=mock_df.copy())
        
        # Assert the output (check for existence of new columns and data types)
        assert all(col in parsed_df.columns for col in ["q_id_3", "q_id_4", "q_id_13", "q_id_15", "q_id_16", "q_id_17"])
        assert parsed_df.loc[0, "q_id_3"][0]["data"] == ["image text to text", "text to video"]
        assert parsed_df.loc[0, "q_id_4"][0]["data"] == ["dataset1"]
        assert parsed_df.loc[0, "q_id_13"][0]["data"] == ["paper1"]
        assert parsed_df.loc[0, "q_id_15"][0]["data"] == ["license1"]
        assert parsed_df.loc[0, "q_id_16"][0]["data"] == ["en"]
        assert parsed_df.loc[0, "q_id_17"][0]["data"] == ["pytorch"]
        assert parsed_df.loc[1, "q_id_3"][0]["data"] == ["image feature extraction", "text to text"]
        assert parsed_df.loc[1, "q_id_4"][0]["data"] == ["dataset2"]
        assert parsed_df.loc[1, "q_id_13"][0]["data"] == ["paper2"]
        assert parsed_df.loc[1, "q_id_15"][0]["data"] == ["license2"]
        assert parsed_df.loc[1, "q_id_16"][0]["data"] == ["sv"]
        assert parsed_df.loc[1, "q_id_17"][0]["data"] == ["peft"]


    def test_parse_fields_from_tags_no_correct_tags(self, parser: MetadataParser) -> None:
        """
        Test that parse_fields_from_tags_HF correctly handles no correct tags
        
        Args:
            parser (MetadataParser): The MetadataParser object
        """
        data = {"modelId": ["m1"], 
                "tags": [["Image-Text-to-Texta",
                        "dataset2:dataset1",
                        "arxiv3:paper1",
                        "license4:license1",
                        "enas",
                        "PyTorch3"]],
                "pipeline_tag": None}
        
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser)
        
        # Call the function to parse tags
        parsed_df = parser.parse_fields_from_tags_HF(HF_df=mock_df.copy())
        
        # Assert the output (check for existence of new columns and data types)
        assert all(col in parsed_df.columns for col in ["q_id_3", "q_id_4", "q_id_13", "q_id_15", "q_id_16", "q_id_17"])
        assert pd.isna(parsed_df.loc[0, "q_id_3"][0]["data"])
        assert pd.isna(parsed_df.loc[0, "q_id_4"][0]["data"])
        assert pd.isna(parsed_df.loc[0, "q_id_13"][0]["data"])
        assert pd.isna(parsed_df.loc[0, "q_id_15"][0]["data"])
        assert pd.isna(parsed_df.loc[0, "q_id_16"][0]["data"])
        assert pd.isna(parsed_df.loc[0, "q_id_17"][0]["data"])
        
    def test_answer_question(self, parser: MetadataParser):
        """
        Tests the answer_question method of the parser class for various scenarios.

        Args:
            parser: An instance of the parser class.
        """
        
        # Test case 1: Simple question
        question = "What model is used as the base model?"
        context = "The base model used is BERT."
        expected_answer = {"data": "BERT", "extraction_method": "Intel/dynamic_tinybert", "confidence": 0.9}
        result = parser.answer_question(question, context)
        assert result[0]["data"] == expected_answer["data"]
        assert result[0]["confidence"] >= 0.5

        # Test case 2: Question with multiple possible answers
        question = "What evaluation metrics were used?"
        context = "The evaluation metrics used were accuracy and F1 score."
        expected_answer = {"data": "accuracy and F1 score", "extraction_method": "Intel/dynamic_tinybert", "confidence": 0.8}
        result = parser.answer_question(question, context)
        assert result[0]["data"] == expected_answer["data"]
        assert result[0]["confidence"] >= 0.5

        # Test case 3: Question with no clear answer
        question = "What is the meaning of life?"
        context = "The lakers won yesterday"
        expected_answer = [{"data": "", "extraction_method": "Intel/dynamic_tinybert", "confidence": 0.0}]
        result = parser.answer_question(question, context)
        assert result[0]["confidence"] < 0.4

    
    def test_parse_fields_from_txt_HF_one_model(self, parser: MetadataParser):
        """
        Tests the parse_fields_from_txt_HF method for a single model scenario.

        Args:
            parser: An instance of the parser class.
        """
        data = {"modelId": ["m1"],
                "card": """
                        The base model used is BERT.The evaluation metrics used were accuracy and F1 score.
                        The hyperparameters optimized during the training process were learning rate and batch size.
                        """}
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser)
        parsed_df = parser.parse_fields_from_txt_HF(HF_df = mock_df)
        print(parsed_df.loc[0, "q_id_18"])
        assert parsed_df.loc[0, "q_id_8"][0]["data"] == "BERT"
        assert parsed_df.loc[0, "q_id_8"][0]["confidence"] > 0.8
        assert parsed_df.loc[0, "q_id_9"][0]["data"] == "accuracy and F1 score"
        assert parsed_df.loc[0, "q_id_9"][0]["confidence"] > 0.8
        assert parsed_df.loc[0, "q_id_11"][0]["data"] == "learning rate and batch size"
        assert parsed_df.loc[0, "q_id_11"][0]["confidence"] > 0.8
        assert parsed_df.loc[0, "q_id_5"][0]["confidence"] < 0.4
        # The current model cannot answer question 10,12,14 or 18 correctly
        # Uncomment when a better model is available
        # assert parsed_df.loc[0, "q_id_10"][0]["confidence"] < 0.4
        # assert parsed_df.loc[0, "q_id_12"][0]["confidence"] < 0.4
        # assert parsed_df.loc[0, "q_id_14"][0]["confidence"] < 0.4
        # assert parsed_df.loc[0, "q_id_18"][0]["confidence"] < 0.4
        
        
        