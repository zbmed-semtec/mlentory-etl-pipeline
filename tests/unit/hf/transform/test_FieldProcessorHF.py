import pytest
import sys
import os
import time
import pandas as pd
from datetime import datetime
from typing import List, Tuple
import pytest
from unittest.mock import Mock
import json
from mlentory_transform.hf_transform.FieldProcessorHF import FieldProcessorHF


class TestFieldProcessorHF:
    """
    Test class for FieldProcessorHF
    """

    @pytest.fixture
    def setup_field_processor(self) -> FieldProcessorHF:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up 3 levels and into configuration
        config_path = os.path.join(
            current_dir, "..", "..", "..", "config", "hf", "transform"
        )

        new_schema = pd.read_csv(f"{config_path}/FAIR4ML_schema.tsv", sep="\t")
        transformations = pd.read_csv(
            f"{config_path}/column_transformations.csv", lineterminator="\n", sep=","
        )
        fields_processor_HF = FieldProcessorHF(new_schema, transformations)
        return fields_processor_HF

    def test_process_row(self, setup_field_processor: FieldProcessorHF):
        field_processor = setup_field_processor
        curr_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        row = pd.Series(
            {
                "q_id_0": [
                    {
                        "data": "model_name",
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_1": [
                    {
                        "data": "author",
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_2": [
                    {
                        "data": "2022-01-01",
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_3": [
                    {
                        "data": ["text-to-text"],
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_4": [
                    {
                        "data": ["dataset1"],
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_6": [
                    {
                        "data": ["dataset2"],
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_7": [
                    {
                        "data": ["dataset3"],
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_8": [
                    {
                        "data": "bert",
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_13": [
                    {
                        "data": ["paper1"],
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_15": [
                    {
                        "data": ["license1"],
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_17": [
                    {
                        "data": ["pytorch"],
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_19": [
                    {
                        "data": ["dataset4"],
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_20": [
                    {
                        "data": "Discover new diseases",
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_21": [
                    {
                        "data": "Model Risks",
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_22": [
                    {
                        "data": ["dataset5", "dataset6"],
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_23": [
                    {
                        "data": "Ryzen5",
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_24": [
                    {
                        "data": ["author1"],
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_26": [
                    {
                        "data": "2022-02-01",
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_27": [
                    {
                        "data": ["funding1"],
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_28": [
                    {
                        "data": "1.0.0",
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_29": [
                    {
                        "data": "4GB",
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
                "q_id_30": [
                    {
                        "data": "release notes",
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ],
            }
        )

        processed_row = field_processor.process_row(row)

        # print("HEYYYYYYY", processed_row["fair4ml:ethicalLegalSocial"])

        assert processed_row["fair4ml:ethicalLegalSocial"] == [
            {
                "data": ["license1"],
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["fair4ml:fineTunedFrom"] == [
            {
                "data": "bert",
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        print("HEYYYYYYY", processed_row["fair4ml:hasCO2eEmissions"])

        assert processed_row["fair4ml:hasCO2eEmissions"] == [
            {
                "data": "Not extracted",
                "extraction_method": "None",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["fair4ml:mlTask"] == [
            {
                "data": ["text-to-text"],
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["fair4ml:sharedBy"] == [
            {
                "data": "author",
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["fair4ml:testedOn"] == [
            {
                "data": ["dataset1"],
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["fair4ml:trainedOn"] == [
            {
                "data": ["dataset1"],
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            },
            {
                "data": ["dataset2"],
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            },
            {
                "data": ["dataset3"],
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            },
        ]
        assert processed_row["schema.org:distribution"] == [
            {
                "data": "https://huggingface.co/model_name",
                "extraction_method": "Built in transform stage",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:memoryRequirements"] == [
            {
                "data": "4GB",
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:processorRequirements"] == [
            {
                "confidence": 1.0,
                "data": "Ryzen5",
                "extraction_method": "parsed",
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:releaseNotes"] == [
            {
                "data": "release notes",
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:softwareRequirements"] == [
            {
                "data": ["pytorch"],
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            },
            {
                "data": "Python",
                "extraction_method": "Added in transform stage",
                "confidence": 1.0,
                "extraction_time": curr_date,
            },
        ]
        assert processed_row["schema.org:storageRequirements"] == [
            {
                "data": "4GB",
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["codemeta:issueTracker"] == [
            {
                "data": "https://huggingface.co/model_name/discussions",
                "extraction_method": "Built in transform stage",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["codemeta:readme"] == [
            {
                "data": "https://huggingface.co/model_name/blob/main/README.md",
                "extraction_method": "Built in transform stage",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["codemeta:referencePublication"] == [
            {
                "data": ["paper1"],
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:author"] == [
            {
                "data": ["author1"],
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:dateCreated"] == [
            {
                "data": "2022-01-01",
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:dateModified"] == [
            {
                "data": "2022-02-01",
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:datePublished"] == [
            {
                "data": "2022-01-01",
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:discussionUrl"] == [
            {
                "data": "https://huggingface.co/model_name/discussions",
                "extraction_method": "Built in transform stage",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:funding"] == [
            {
                "data": ["funding1"],
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:license"] == [
            {
                "data": ["license1"],
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:maintainer"] == [
            {
                "data": "author",
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:version"] == [
            {
                "data": "1.0.0",
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:identifier"] == [
            {
                "data": "https://huggingface.co/model_name",
                "extraction_method": "Built in transform stage",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:name"] == [
            {
                "data": "model_name",
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]
        assert processed_row["schema.org:url"] == [
            {
                "data": "https://huggingface.co/model_name",
                "extraction_method": "Built in transform stage",
                "confidence": 1.0,
                "extraction_time": curr_date,
            }
        ]

    def test_process_softwareRequirements(self, setup_field_processor):
        field_processor = setup_field_processor
        curr_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        info_HF = pd.Series(
            {
                "q_id_17": [
                    {
                        "data": ["pytorch", "tensorflow"],
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": curr_date,
                    }
                ]
            }
        )

        field_processor.current_row = info_HF

        processed_value = field_processor.process_softwareRequirements()
        print("Processed value: ", processed_value)
        assert processed_value == [
            {
                "data": ["pytorch", "tensorflow"],
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": curr_date,
            },
            {
                "data": "Python",
                "extraction_method": "Added in transform stage",
                "confidence": 1.0,
                "extraction_time": curr_date,
            },
        ]

    def test_process_trainedOn(self, setup_field_processor):
        field_processor = setup_field_processor
        info_HF = pd.Series(
            {
                "q_id_4": [
                    {
                        "data": ["dataset1"],
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                    }
                ],
                "q_id_6": [
                    {
                        "data": ["dataset2"],
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                    }
                ],
                "q_id_7": [
                    {
                        "data": ["dataset3"],
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                    }
                ],
            }
        )

        field_processor.current_row = info_HF
        processed_value = field_processor.process_trainedOn()

        assert processed_value == [
            {"data": ["dataset1"], "extraction_method": "parsed", "confidence": 1.0},
            {"data": ["dataset2"], "extraction_method": "parsed", "confidence": 1.0},
            {"data": ["dataset3"], "extraction_method": "parsed", "confidence": 1.0},
        ]

    def test_build_HF_link(self, setup_field_processor):
        field_processor = setup_field_processor
        info_HF = pd.Series(
            {
                "q_id_0": [
                    {
                        "data": "model_name",
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": "2022-01-01",
                    }
                ]
            }
        )

        field_processor.current_row = info_HF
        processed_value = field_processor.build_HF_link(tail_info="/discussions")

        assert processed_value == [
            {
                "data": "https://huggingface.co/model_name/discussions",
                "extraction_method": "Built in transform stage",
                "confidence": 1.0,
                "extraction_time": processed_value[0]["extraction_time"],
            }
        ]

    def test_find_value_in_HF(self, setup_field_processor):
        field_processor = setup_field_processor
        info_HF = pd.Series(
            {
                "q_id_0": [
                    {
                        "data": "model_name",
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                    }
                ]
            }
        )

        field_processor.current_row = info_HF
        processed_value = field_processor.find_value_in_HF("q_id_0")

        assert processed_value == [
            {"data": "model_name", "extraction_method": "parsed", "confidence": 1.0}
        ]

    def test_add_default_extraction_info(self, setup_field_processor):
        field_processor = setup_field_processor

        processed_value = field_processor.add_default_extraction_info(
            "test_data", "test_method", 0.8
        )

        assert processed_value == {
            "data": "test_data",
            "extraction_method": "test_method",
            "confidence": 0.8,
            "extraction_time": processed_value["extraction_time"],
        }

    def test_apply_transformation_basic(self, setup_field_processor):
        """Test basic transformation without parameters"""
        field_processor = setup_field_processor

        # Mock input data
        row = pd.Series(
            {
                "q_id_0": [
                    {
                        "data": "test_model",
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": "2024-01-01",
                    }
                ]
            }
        )

        transformation = pd.Series(
            {
                "transformation_function": "find_value_in_HF",
                "parameters": '{"property_name": "q_id_0"}',
                "target_column": "schema.org:name",
            }
        )

        result = field_processor.apply_transformation(row, transformation)

        assert result == [
            {
                "data": "test_model",
                "extraction_method": "parsed",
                "confidence": 1.0,
                "extraction_time": "2024-01-01",
            }
        ]

    def test_apply_transformation_with_link_building(self, setup_field_processor):
        """Test transformation that builds HF links"""
        field_processor = setup_field_processor

        row = pd.Series(
            {
                "q_id_0": [
                    {
                        "data": "test_model",
                        "extraction_method": "parsed",
                        "confidence": 1.0,
                        "extraction_time": "2024-01-01",
                    }
                ]
            }
        )

        transformation = pd.Series(
            {
                "transformation_function": "build_HF_link",
                "parameters": '{"tail_info": "/discussions"}',
                "target_column": "schema.org:discussionUrl",
            }
        )

        result = field_processor.apply_transformation(row, transformation)

        assert result[0]["data"] == "https://huggingface.co/test_model/discussions"
        assert result[0]["extraction_method"] == "Built in transform stage"
        assert result[0]["confidence"] == 1.0

    def test_apply_transformation_not_extracted(self, setup_field_processor):
        """Test transformation for not extracted fields"""
        field_processor = setup_field_processor

        row = pd.Series()
        transformation = pd.Series(
            {
                "transformation_function": "process_not_extracted",
                "parameters": None,
                "target_column": "fair4ml:hasCO2eEmissions",
            }
        )

        result = field_processor.apply_transformation(row, transformation)

        assert result[0]["data"] == "Not extracted"
        assert result[0]["extraction_method"] == "None"
        assert result[0]["confidence"] == 1.0

    def test_apply_transformation_invalid_function(self, setup_field_processor):
        """Test handling of invalid transformation function"""
        field_processor = setup_field_processor

        row = pd.Series()
        transformation = pd.Series(
            {
                "transformation_function": "non_existent_function",
                "parameters": None,
                "target_column": "test",
            }
        )

        with pytest.raises(KeyError):
            field_processor.apply_transformation(row, transformation)

    def test_apply_transformation_invalid_parameters(self, setup_field_processor):
        """Test handling of invalid JSON parameters"""
        field_processor = setup_field_processor

        row = pd.Series()
        transformation = pd.Series(
            {
                "transformation_function": "find_value_in_HF",
                "parameters": "{invalid_json",
                "target_column": "test",
            }
        )

        with pytest.raises(json.JSONDecodeError):
            field_processor.apply_transformation(row, transformation)
