from multiprocessing import Process, Pool, set_start_method, get_context
import pytest
import sys
import os
import time
import pandas as pd
from datetime import datetime
from typing import List, Tuple


sys.path.append(".")
from transform.core.FilesProcessor import FilesProcessor
from transform.core.QueueObserver import QueueObserver, MyQueueEventHandler
from transform.core.FieldProcessorHF import FieldProcessorHF


class TestFieldProcessorHF:
    """
    Test class for FieldProcessorHF
    """

    @classmethod
    def setup_class(self):
        self.hf_example_file = pd.read_csv(
            "./tests/Test_files/hf_extracted_example_file.tsv",
            sep="\t",
            usecols=lambda x: x != "Unnamed: 0",
        )

    @pytest.fixture
    def setup_field_processor(self) -> FieldProcessorHF:
        fields_processor_HF = FieldProcessorHF(path_to_config_data="./config_data")
        return fields_processor_HF

    def test_conversion(
        self, caplog, setup_field_processor: FieldProcessorHF, logger
    ) -> None:
        """
        Test that workers are created on complete batch.

        Args:
            caplog: pytest caplog fixture for capturing logs
            setup_field_processor: fixture for setting up the FieldProcessorHF instance
        """
        df = self.hf_example_file
        field_processor = setup_field_processor
        print("TYPEEEEEEEEEEE", type(setup_field_processor))

        manager = get_context("spawn").Manager()
        model_list = manager.list()
        for index, row in df.iterrows():
            # print(row)
            m4ml_model_data = field_processor.process_row(row)
            model_list.append(m4ml_model_data)
            print("m4ml new row: \n", m4ml_model_data)

        models_m4ml_df = pd.DataFrame(list(model_list))

        print(models_m4ml_df.head())

        # Assert the each file got processed
        # assert self.check_files_got_processed(file_paths,file_processor)

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

        print("HEYYYYYYY", processed_row["fair4ml:ethicalLegalSocial"])

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

    def test_process_property_invalid_property_name(self, setup_field_processor):
        field_processor = setup_field_processor
        property_description_M4ML = pd.Series(
            {"Property": "invalid_property", "Source": "invalid_source"}
        )
        info_HF = pd.Series()

        processed_value = field_processor.process_property(
            property_description_M4ML, info_HF
        )

        assert processed_value == ""

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

        processed_value = field_processor.process_softwareRequirements(info_HF)
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

        processed_value = field_processor.process_trainedOn(info_HF)

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

        processed_value = field_processor.build_HF_link(
            info_HF, tail_info="/discussions"
        )

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

        processed_value = field_processor.find_value_in_HF(info_HF, "q_id_0")

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
