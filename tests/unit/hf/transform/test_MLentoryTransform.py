import pytest
import pandas as pd
import os
from datetime import datetime
from mlentory_transform.core.MlentoryTransform import MlentoryTransform


@pytest.fixture
def sample_schema() -> pd.DataFrame:
    """
    Create a sample schema DataFrame for testing.

    Returns:
        pd.DataFrame: Sample schema with required columns
    """
    schema_data = pd.read_csv("tests/unit/hf/transform/FAIR4ML_schema.tsv", sep="\t")
    return schema_data


@pytest.fixture
def sample_transformations() -> pd.DataFrame:
    """
    Create sample transformation rules DataFrame for testing.

    Returns:
        pd.DataFrame: Sample transformation rules
    """
    transformations_data = pd.read_csv(
        "tests/unit/hf/transform/column_transformations.csv"
    )
    return transformations_data


@pytest.fixture
def sample_extracted_data() -> pd.DataFrame:
    """
    Create sample extracted model data for testing.

    Returns:
        pd.DataFrame: Sample model data
    """
    extracted_data = {
        "modelId": ["bert-base", "gpt2-small"],
        "taskType": ["text-classification", "text-generation"],
        "downloadCount": [1000, 2000],
        "extra_field": ["ignore1", "ignore2"],
    }
    return pd.DataFrame(extracted_data)


@pytest.fixture
def temp_output_dir(tmp_path) -> str:
    """
    Create a temporary directory for test outputs.

    Args:
        tmp_path: pytest fixture for temporary directory

    Returns:
        str: Path to temporary directory
    """
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return str(output_dir)


class TestMlentoryTransform:
    """Test suite for MlentoryTransform class."""

    def test_transform_hf_models_basic(
        self,
        sample_schema: pd.DataFrame,
        sample_transformations: pd.DataFrame,
        sample_extracted_data: pd.DataFrame,
    ):
        """
        Test basic transformation functionality without saving output.

        Args:
            sample_schema: Sample schema DataFrame
            sample_transformations: Sample transformation rules
            sample_extracted_data: Sample input data
        """
        transformer = MlentoryTransform()
        result_df = transformer.transform_HF_models(
            sample_schema, sample_transformations, sample_extracted_data
        )

        # Verify the transformed DataFrame structure
        assert isinstance(result_df, pd.DataFrame)
        assert "model_name" in result_df.columns
        assert "task" in result_df.columns
        assert "downloads" in result_df.columns

        # Verify data transformation
        assert result_df["model_name"].tolist() == ["bert-base", "gpt2-small"]
        assert result_df["downloads"].tolist() == [1000, 2000]

    def test_transform_hf_models_with_save(
        self,
        sample_schema: pd.DataFrame,
        sample_transformations: pd.DataFrame,
        sample_extracted_data: pd.DataFrame,
        temp_output_dir: str,
    ):
        """
        Test transformation with output saving functionality.

        Args:
            sample_schema: Sample schema DataFrame
            sample_transformations: Sample transformation rules
            sample_extracted_data: Sample input data
            temp_output_dir: Temporary directory path
        """
        transformer = MlentoryTransform()
        result_df = transformer.transform_HF_models(
            sample_schema,
            sample_transformations,
            sample_extracted_data,
            save_output_in_json=True,
            output_dir=temp_output_dir,
        )

        # Verify knowledge graph creation
        assert "HF_models" in transformer.current_sources
        assert transformer.current_sources["HF_models"] is not None

    def test_transform_hf_models_invalid_save(
        self,
        sample_schema: pd.DataFrame,
        sample_transformations: pd.DataFrame,
        sample_extracted_data: pd.DataFrame,
    ):
        """
        Test transformation with invalid save parameters.

        Args:
            sample_schema: Sample schema DataFrame
            sample_transformations: Sample transformation rules
            sample_extracted_data: Sample input data
        """
        transformer = MlentoryTransform()

        with pytest.raises(ValueError):
            transformer.transform_HF_models(
                sample_schema,
                sample_transformations,
                sample_extracted_data,
                save_output_in_json=True,
                output_dir=None,
            )

    @pytest.mark.integration
    def test_end_to_end_transformation(
        self,
        sample_schema: pd.DataFrame,
        sample_transformations: pd.DataFrame,
        sample_extracted_data: pd.DataFrame,
        temp_output_dir: str,
    ):
        """
        Integration test for complete transformation workflow.

        Args:
            sample_schema: Sample schema DataFrame
            sample_transformations: Sample transformation rules
            sample_extracted_data: Sample input data
            temp_output_dir: Temporary directory path
        """
        transformer = MlentoryTransform()

        # Transform the data
        result_df = transformer.transform_HF_models(
            sample_schema,
            sample_transformations,
            sample_extracted_data,
            save_output_in_json=True,
            output_dir=temp_output_dir,
        )

        # Save individual sources
        transformer.save_indiviual_sources(temp_output_dir)

        # Verify files were created
        files = os.listdir(temp_output_dir)
        assert any(
            file.endswith("HF_models_transformation_result.json") for file in files
        )
