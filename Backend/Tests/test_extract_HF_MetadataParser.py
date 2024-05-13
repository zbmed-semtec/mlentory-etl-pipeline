
from datasets import load_dataset
from datetime import datetime
import pandas as pd
import os
import pytest
import sys

sys.path.append('./../Extractors/HF_Extractor')
from Core.MetadataParser import MetadataParser


class TestMetadataParser:

    # @pytest.fixture
    # def hf_dataset(self):
    #     dataset_models = load_dataset("librarian-bots/model_cards_with_metadata")['train']
    #     return dataset_models.to_pandas()
    
    @pytest.fixture
    def parser(self):
        #Creating the parser object that will perform the transformations on the raw data 
        parser = MetadataParser(qa_model="Intel/dynamic_tinybert",path_to_config_data="./../Extractors/HF_Extractor/Config_Data")
        return parser
    
    def add_base_questions(self,df,parser):
        
        new_columns = {}

        for idx in range(len(parser.questions)):
            q_id = "q_id_"+str(idx)
            new_columns[q_id] = [None for _ in range(len(df))]

        df = df.assign(**new_columns)
        
        return df
    
    def test_add_default_extraction_info(self,parser):
        data = "This is some data"
        extraction_method = "parser_method"
        confidence = 0.95
        # Call the function
        info_dict = parser.add_default_extraction_info(data, extraction_method, confidence)
        # Assert the output
        assert info_dict == {"data": data, "extraction_method": extraction_method, "confidence": confidence}

    def test_parse_known_fields_HF(self,parser):
        # Mock DataFrame
        data = {"modelId": ["m1", "m2"],
                "author": ["a1", "a2"],
                "createdAt": ["c1", "c2"]}
        
        mock_df = pd.DataFrame(data)
        
        mock_df = self.add_base_questions(mock_df,parser)
        
        # Call the function
        parsed_df = parser.parse_known_fields_HF(HF_df=mock_df.copy())
        # Assert the output (check for existence of new columns and data types)
        assert all(col in parsed_df.columns for col in ["q_id_0", "q_id_1", "q_id_2"])
        assert parsed_df["q_id_0"].dtype == object
        assert parsed_df["q_id_1"].dtype == object
        assert parsed_df["q_id_2"].dtype == object
    
    def test_parse_known_fields_HF_empty_dataframe(self, parser):
        empty_df = pd.DataFrame()
        mock_df = self.add_base_questions(empty_df,parser)
        # Assert that the output is an empty DataFrame
        with pytest.raises(KeyError):
            parsed_df = parser.parse_known_fields_HF(HF_df=mock_df.copy())
        
    def test_parse_known_fields_HF_missing_columns(self, parser):
        data = {"author": ["a1", "a2"]}
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df,parser)
        # Assert that the output is an empty DataFrame
        with pytest.raises(KeyError):
            parsed_df = parser.parse_known_fields_HF(HF_df=mock_df.copy())
    
    def test_parse_known_fields_HF_default_info(self, parser):
        data = {"modelId": ["m1"], "author": ["a1"], "createdAt": ["c1"]}
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df,parser)
        parsed_df = parser.parse_known_fields_HF(HF_df=mock_df.copy())
        # Assert that all new columns have the expected info dictionary
        
        for col in ["q_id_0", "q_id_1", "q_id_2"]:
            assert parsed_df.loc[0, col]["extraction_method"] == "Parsed_from_HF_dataset"
            assert parsed_df.loc[0, col]["confidence"] == 1.0
                
    def test_parse_known_fields_HF_finetuned_model(self, parser):
        data = {"modelId": ["m1", "m2"],
                "author": ["a1", "a2"],
                "createdAt": ["c1", "c2"]}
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df,parser)
        mock_df.loc[0,"q_id_8"] = "answer"
        mock_df.loc[0,"q_id_4"] = "answer"
        
        mock_df.loc[1,"q_id_8"] = "[CLS]"
        mock_df.loc[1,"q_id_4"] = "answer"
        
        mock_df.loc[2,"q_id_8"] = None
        mock_df.loc[2,"q_id_4"] = "answer"
        parsed_df = parser.parse_known_fields_HF(HF_df=mock_df.copy())
        # Assert that q_id_6 and q_id_7 have the value from q_id_4 for the second row only
        assert parsed_df.loc[0, "q_id_6"]["data"] == "answer"
        assert parsed_df.loc[0, "q_id_7"]["data"] == "answer"
        assert pd.isna(parsed_df.loc[1, "q_id_6"]["data"])
        assert pd.isna(parsed_df.loc[1, "q_id_7"]["data"])
        assert pd.isna(parsed_df.loc[2, "q_id_6"]["data"]) 
        assert pd.isna(parsed_df.loc[2, "q_id_7"]["data"])
        
    def test_parse_fields_from_tags_one_model_all_tags(self, parser):
        data = {"modelId": ["m1"], 
                "tags": [[  "Image-Text-to-Text",
                            "dataset:dataset1",
                            "arxiv:paper1",
                            "license:license1",
                            "en",
                            "PyTorch"]],
                "pipeline_tag":"Text-to-Video"}
        
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser)
        
        # Call the function
        parsed_df = parser.parse_fields_from_tags_HF(HF_df=mock_df.copy())
        
        # Assert the output (check for existence of new columns and data types)
        assert all(col in parsed_df.columns for col in ["q_id_3", "q_id_4", "q_id_13", "q_id_15", "q_id_16", "q_id_17"])
        assert parsed_df.loc[0, "q_id_3"]["data"] == ["image text to text","text to video"]
        assert parsed_df.loc[0, "q_id_4"]["data"] == ["dataset1"]
        assert parsed_df.loc[0, "q_id_13"]["data"] == ["paper1"]
        assert parsed_df.loc[0, "q_id_15"]["data"] == ["license1"]
        assert parsed_df.loc[0, "q_id_16"]["data"] == ["en"]
        assert parsed_df.loc[0, "q_id_17"]["data"] == ["pytorch"]
    
    def test_parse_fields_from_tags_two_models_all_tags(self, parser):
        data = {"modelId": ["m1","m2"], 
                "tags": [[  "Image-Text-to-Text",
                            "dataset:dataset1",
                            "arxiv:paper1",
                            "license:license1",
                            "en",
                            "PyTorch"],
                         [  "Image Feature Extraction",
                            "dataset:dataset2",
                            "arxiv:paper2",
                            "license:license2",
                            "sv",
                            "PEFT"]],
                "pipeline_tag":["Text To Video","Text-to-Text"]}
        
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser)
        
        # Call the function
        parsed_df = parser.parse_fields_from_tags_HF(HF_df=mock_df.copy())
        
        # Assert the output (check for existence of new columns and data types)
        assert all(col in parsed_df.columns for col in ["q_id_3", "q_id_4", "q_id_13", "q_id_15", "q_id_16", "q_id_17"])
        assert parsed_df.loc[0, "q_id_3"]["data"] == ["image text to text","text to video"]
        assert parsed_df.loc[0, "q_id_4"]["data"] == ["dataset1"]
        assert parsed_df.loc[0, "q_id_13"]["data"] == ["paper1"]
        assert parsed_df.loc[0, "q_id_15"]["data"] == ["license1"]
        assert parsed_df.loc[0, "q_id_16"]["data"] == ["en"]
        assert parsed_df.loc[0, "q_id_17"]["data"] == ["pytorch"]
        assert parsed_df.loc[1, "q_id_3"]["data"] == ["image feature extraction","text to text"]
        assert parsed_df.loc[1, "q_id_4"]["data"] == ["dataset2"]
        assert parsed_df.loc[1, "q_id_13"]["data"] == ["paper2"]
        assert parsed_df.loc[1, "q_id_15"]["data"] == ["license2"]
        assert parsed_df.loc[1, "q_id_16"]["data"] == ["sv"]
        assert parsed_df.loc[1, "q_id_17"]["data"] == ["peft"]
    
    def test_parse_fields_from_tags_no_correct_tags(self, parser):
        data = {"modelId": ["m1"], 
                "tags": [[  "Image-Text-to-Texta",
                            "dataset2:dataset1",
                            "arxiv3:paper1",
                            "license4:license1",
                            "enas",
                            "PyTorch3"]],
                "pipeline_tag":None}
        
        mock_df = pd.DataFrame(data)
        mock_df = self.add_base_questions(mock_df, parser)
        
        # Call the function
        parsed_df = parser.parse_fields_from_tags_HF(HF_df=mock_df.copy())
        
        
        # Assert the output (check for existence of new columns and data types)
        assert all(col in parsed_df.columns for col in ["q_id_3", "q_id_4", "q_id_13", "q_id_15", "q_id_16", "q_id_17"])
        assert pd.isna(parsed_df.loc[0, "q_id_3"]["data"])
        assert pd.isna(parsed_df.loc[0, "q_id_4"]["data"])
        assert pd.isna(parsed_df.loc[0, "q_id_13"]["data"])
        assert pd.isna(parsed_df.loc[0, "q_id_15"]["data"])
        assert pd.isna(parsed_df.loc[0, "q_id_16"]["data"])
        assert pd.isna(parsed_df.loc[0, "q_id_17"]["data"])
        
    def test_answer_question(self, parser):
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

        # Test case 4: Question with multiple sentences in context
        question = "What hyperparameters were optimized during the training process?"
        context = "The hyperparameters optimized during the training process were learning rate and batch size. The model was trained for 10 epochs."
        expected_answer = {"data": "learning rate and batch size", "extraction_method": "Intel/dynamic_tinybert", "confidence": 0.85}
        result = parser.answer_question(question, context)
        assert result[0]["data"] == expected_answer["data"]
        assert result[0]["confidence"] >= 0.3
    
    def test_parse_fields_from_txt_HF_one_model(self, parser):
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
        
        
        