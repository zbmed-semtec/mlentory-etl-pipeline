import pytest
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime
import json
import os
from openml_extract.OpenMLExtractor import OpenMLExtractor  # Assuming the class is in openml_extractor.py

@pytest.fixture
def schema_file(tmp_path):
    schema = {
        "run": {
            "schema.org:name": "Run_{run.id}",
            "schema.org:url": "run.openml_url",
            "schema.org:version": "flow.version",
            "schema.org:description": "flow.description",
            "fair4ml:intendedUse": "flow.description",
            "fair4ml:mlTask": "task.task_type",
            "fair4ml:modelCategory": "run.flow_name",
            "fair4ml:evaluatedOn": "dataset.name",
            "schema.org:dateCreated": "flow.upload_date",
            "schema.org:dateModified": "flow.upload_date",
            "schema.org:datePublished": "flow.upload_date",
            "schema.org:inLanguage": "flow.language"
        },
        "dataset": {
            "schema.org:identifier": "dataset.id",
            "schema.org:name": "dataset.name",
            "schema.org:WebPage": "dataset.openml_url",
            "schema.org:fileFormat": "dataset.format",
            "schema.org:licence": "dataset.licence",
            "schema.org:citation": "dataset.citation",
            "schema.org:inLanguage": "dataset.language",
            "schema.org:uploadDate": "dataset.upload_date",
            "schema.org:version": "dataset.version",
            "schema.org:distribution": "dataset.url",
            "schema.org:sameAs": "dataset.paper_url",
            "schema.org:description": "dataset.description",
            "schema.org:status": "NA",
            "likes": "NA",
            "downloads": "NA",
            "issues": "NA"
        }
    }
    schema_path = tmp_path / "schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f)
    return str(schema_path)

@pytest.fixture
def extractor(schema_file):
    return OpenMLExtractor(schema_file)

def test_init_valid_schema(extractor, schema_file):
    assert extractor.schema is not None
    assert extractor.schema == json.load(open(schema_file))

def test_init_invalid_schema(tmp_path):
    invalid_schema = tmp_path / "invalid.json"
    with open(invalid_schema, "w") as f:
        f.write("invalid json")
    with pytest.raises(ValueError):
        OpenMLExtractor(str(invalid_schema))

def test_load_schema_file_not_found():
    with pytest.raises(FileNotFoundError):
        OpenMLExtractor("non_existent.json")

@patch("openml.runs.get_run")
@patch("openml.datasets.get_dataset")
@patch("openml.flows.get_flow")
@patch("openml.tasks.get_task")
def test_get_run_metadata(mock_task, mock_flow, mock_dataset, mock_run, extractor):
    mock_run.return_value = Mock(id=7, openml_url="https://www.openml.org/r/7", flow_name="weka.SMO_PolyKernel(1)")
    mock_dataset.return_value = Mock(name="credit-approval")
    mock_flow.return_value = Mock(
        version="1",
        description="J. Platt: Fast Training of Support Vector Machines...",
        upload_date="2014-04-04T14:39:43",
        language="English"
    )
    mock_task.return_value = Mock(task_type="Learning Curve")

    metadata = extractor.get_run_metadata(7)

    assert metadata["schema.org:name"][0]["data"] == "Run_7"
    assert metadata["schema.org:url"][0]["data"] == "https://www.openml.org/r/7"
    assert metadata["schema.org:version"][0]["data"] == "1"
    assert metadata["fair4ml:mlTask"][0]["data"] == "Learning Curve"
    assert metadata["fair4ml:modelCategory"][0]["data"] == "weka.SMO_PolyKernel(1)"
    assert metadata["fair4ml:evaluatedOn"][0]["data"] == "credit-approval"
    assert metadata["schema.org:dateCreated"][0]["data"] == "2014-04-04T14:39:43"
    assert metadata["schema.org:inLanguage"][0]["data"] == "English"
    assert metadata["schema.org:name"][0]["extraction_method"] == "openml_python_package"

@patch("openml.datasets.get_dataset")
@patch.object(OpenMLExtractor, "_scrape_openml_stats")
def test_get_dataset_metadata(mock_scrape, mock_dataset, extractor):
    mock_dataset.return_value = Mock(
        id=3,
        name="kr-vs-kp",
        openml_url="https://www.openml.org/d/3",
        format="ARFF",
        licence="Public",
        citation="https://archive.ics.uci.edu/ml/citation_policy.html",
        language="English",
        upload_date="2014-04-06T23:19:28",
        version=1,
        url="https://api.openml.org/data/v1/download/3/kr-vs-kp.arff",
        paper_url="https://dl.acm.org/doi/abs/10.5555/32231",
        description="Author: Alen Shapiro..."
    )
    mock_scrape.return_value = {"status": "verified", "downloads": 0, "likes": 0, "issues": 0}

    datasets_df = pd.DataFrame({"did": [3], "status": ["active"]})
    metadata = extractor.get_dataset_metadata(3, datasets_df)

    assert metadata["schema.org:identifier"][0]["data"] == 3
    assert metadata["schema.org:name"][0]["data"] == "kr-vs-kp"
    assert metadata["schema.org:WebPage"][0]["data"] == "https://www.openml.org/d/3"
    assert metadata["schema.org:fileFormat"][0]["data"] == "ARFF"
    assert metadata["schema.org:licence"][0]["data"] == "Public"
    assert metadata["schema.org:inLanguage"][0]["data"] == "English"
    assert metadata["schema.org:version"][0]["data"] == "1"
    assert metadata["schema.org:status"][0]["data"] == "verified"
    assert metadata["downloads"][0]["data"] == 0
    assert metadata["schema.org:status"][0]["extraction_method"] == "web_scraping"

@patch("openml.runs.list_runs")
@patch.object(OpenMLExtractor, "get_run_metadata")
def test_get_multiple_runs_metadata(mock_get_metadata, mock_list_runs, extractor, tmp_path):
    mock_list_runs.return_value = pd.DataFrame({"run_id": [7, 8]})
    mock_get_metadata.side_effect = [
        {
            "schema.org:name": [{"data": "Run_7"}],
            "fair4ml:mlTask": [{"data": "Learning Curve"}]
        },
        {
            "schema.org:name": [{"data": "Run_8"}],
            "fair4ml:mlTask": [{"data": "Classification"}]
        }
    ]

    result = extractor.get_multiple_runs_metadata(2, 0, 2, str(tmp_path), True)

    assert len(result) == 2
    assert result["schema.org:name"].tolist() == [[{"data": "Run_7"}], [{"data": "Run_8"}]]
    assert os.path.exists(tmp_path / "openml_run_metadata_")

@patch("openml.datasets.list_datasets")
@patch.object(OpenMLExtractor, "get_dataset_metadata")
def test_get_multiple_datasets_metadata(mock_get_metadata, mock_list_datasets, extractor, tmp_path):
    mock_list_datasets.return_value = pd.DataFrame({"did": [3, 4], "status": ["active", "active"]})
    mock_get_metadata.side_effect = [
        {
            "schema.org:identifier": [{"data": 3}],
            "schema.org:name": [{"data": "kr-vs-kp"}]
        },
        {
            "schema.org:identifier": [{"data": 4}],
            "schema.org:name": [{"data": "other_dataset"}]
        }
    ]

    result = extractor.get_multiple_datasets_metadata(2, 0, 2, str(tmp_path), True)

    assert len(result) == 2
    assert result["schema.org:identifier"].tolist() == [[{"data": 3}], [{"data": 4}]]
    assert os.path.exists(tmp_path / "openml_dataset_metadata_")

@patch.object(OpenMLExtractor, "get_multiple_runs_metadata")
@patch.object(OpenMLExtractor, "get_multiple_datasets_metadata")
def test_extract_run_info_with_additional_entities(mock_datasets, mock_runs, extractor):
    mock_runs.return_value = pd.DataFrame({
        "schema.org:name": [[{"data": "Run_7"}]]
    })
    mock_datasets.return_value = pd.DataFrame({
        "schema.org:identifier": [[{"data": 3}]]
    })

    result = extractor.extract_run_info_with_additional_entities(1, 0, 2, "./outputs", False, ["dataset"])

    assert "run" in result
    assert "dataset" in result
    assert len(result["run"]) == 1
    assert len(result["dataset"]) == 1