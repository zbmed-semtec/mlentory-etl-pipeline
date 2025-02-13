import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import os
from datetime import datetime
from ..utils.enums import Platform, SchemasURL

from .FieldProcessorHF import FieldProcessorHF


class TransformHF:
    """
    A class for transforming HuggingFace model metadata into a standardized schema.

    This class provides functionality to:
    - Transform extracted model data into a target schema
    - Apply field-level transformations
    - Save transformed data in various formats
    - Track transformation progress

    Attributes:
        new_schema (pd.DataFrame): Target schema for the transformed data
        transformations (pd.DataFrame): Transformation rules to apply
        fields_processor (FieldProcessorHF): Processor for field-level transformations
    """

    def __init__(self, new_schema: pd.DataFrame, transformations: pd.DataFrame):
        """
        Initialize the HuggingFace transformer with schema and transformations.

        Args:
            new_schema (pd.DataFrame): DataFrame containing the target schema
            transformations (pd.DataFrame): DataFrame containing column transformations
        """
        self.new_schema = new_schema
        self.transformations = transformations
        self.fields_processor = FieldProcessorHF(self.new_schema, self.transformations)

    def transform_models(
        self,
        extracted_df: pd.DataFrame,
        save_output_in_json: bool = False,
        output_dir: str = None,
    ) -> pd.DataFrame:
        """
        Transform the extracted data into the target schema.

        This method:
        1. Processes each row of the input DataFrame
        2. Applies the specified transformations
        3. Optionally saves the results to a file
        4. Shows progress using tqdm

        Args:
            extracted_df (pd.DataFrame): DataFrame containing extracted model data
            save_output_in_json (bool, optional): Whether to save the transformed data.
                Defaults to False.
            output_dir (str, optional): Directory to save the transformed data.
                Required if save_output_in_json is True.

        Returns:
            pd.DataFrame: Transformed DataFrame conforming to the target schema

        Raises:
            ValueError: If save_output_in_json is True but output_dir is not provided
        """
        if (save_output_in_json == True) and (output_dir is None):
            raise ValueError("output_dir must be provided when save_output is True")

        processed_models = []

        for row_num, row in tqdm(
            extracted_df.iterrows(),
            total=len(extracted_df),
            desc="Transforming progress",
        ):
            model_data = self.fields_processor.process_row(row)
            processed_models.append(model_data)

        transformed_df = pd.DataFrame(list(processed_models))

        if save_output_in_json:
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = os.path.join(
                output_dir, f"{current_date}_transformation_results.csv"
            )
            transformed_df.to_json(output_path, index=False)
            print(f"Saved transformation results to {output_path}")

        return transformed_df

    def transform_dataset_croissant(
        self, dataset_id: str, croissant_metadata: dict, extraction_metadata: dict
    ) -> List[dict]:
        """
        Transform Croissant metadata into rows compatible with FAIR4ML schema.

        Args:
            dataset_id (str): The HuggingFace dataset ID
            croissant_metadata (dict): Dataset metadata in Croissant format
            extraction_metadata (dict): Metadata about the extraction process

        Returns:
            List[dict]: List of dictionaries containing transformed metadata
        """
        rows = []
        extraction_time = extraction_metadata.get("extraction_time")
        extraction_method = extraction_metadata.get("extraction_method")
        confidence = extraction_metadata.get("confidence", 1.0)

        # Base dataset information
        base_info = {
            "schema.org:name": [
                {
                    "data": dataset_id,
                    "extraction_method": extraction_method,
                    "confidence": confidence,
                    "extraction_time": extraction_time,
                }
            ],
            "schema.org:description": [
                {
                    "data": croissant_metadata.get("description", ""),
                    "extraction_method": extraction_method,
                    "confidence": confidence,
                    "extraction_time": extraction_time,
                }
            ],
            "schema.org:url": [
                {
                    "data": croissant_metadata.get("url", ""),
                    "extraction_method": extraction_method,
                    "confidence": confidence,
                    "extraction_time": extraction_time,
                }
            ],
        }

        # Add creator information
        creator = croissant_metadata.get("creator", {})
        if creator:
            base_info["schema.org:creator"] = [
                {
                    "data": creator.get("name", ""),
                    "extraction_method": extraction_method,
                    "confidence": confidence,
                    "extraction_time": extraction_time,
                }
            ]
            if "url" in creator:
                base_info["schema.org:creatorUrl"] = [
                    {
                        "data": creator["url"],
                        "extraction_method": extraction_method,
                        "confidence": confidence,
                        "extraction_time": extraction_time,
                    }
                ]

        # Add keywords
        keywords = croissant_metadata.get("keywords", [])
        if keywords:
            base_info["schema.org:keywords"] = [
                {
                    "data": keyword,
                    "extraction_method": extraction_method,
                    "confidence": confidence,
                    "extraction_time": extraction_time,
                }
                for keyword in keywords
            ]

        # Add license information
        if "license" in croissant_metadata:
            base_info["schema.org:license"] = [
                {
                    "data": croissant_metadata["license"],
                    "extraction_method": extraction_method,
                    "confidence": confidence,
                    "extraction_time": extraction_time,
                }
            ]

        # Process distributions
        for dist in croissant_metadata.get("distribution", []):
            dist_row = base_info.copy()

            if "contentUrl" in dist:
                dist_row["schema.org:contentUrl"] = [
                    {
                        "data": dist["contentUrl"],
                        "extraction_method": extraction_method,
                        "confidence": confidence,
                        "extraction_time": extraction_time,
                    }
                ]

            if "encodingFormat" in dist:
                dist_row["schema.org:encodingFormat"] = [
                    {
                        "data": dist["encodingFormat"],
                        "extraction_method": extraction_method,
                        "confidence": confidence,
                        "extraction_time": extraction_time,
                    }
                ]

            if "name" in dist:
                dist_row["cr:distributionName"] = [
                    {
                        "data": dist["name"],
                        "extraction_method": extraction_method,
                        "confidence": confidence,
                        "extraction_time": extraction_time,
                    }
                ]

            rows.append(dist_row)

        # Process record sets
        for record_set in croissant_metadata.get("recordSet", []):
            record_row = base_info.copy()

            if "name" in record_set:
                record_row["cr:recordSetName"] = [
                    {
                        "data": record_set["name"],
                        "extraction_method": extraction_method,
                        "confidence": confidence,
                        "extraction_time": extraction_time,
                    }
                ]

            if "description" in record_set:
                record_row["cr:recordSetDescription"] = [
                    {
                        "data": record_set["description"],
                        "extraction_method": extraction_method,
                        "confidence": confidence,
                        "extraction_time": extraction_time,
                    }
                ]

            # Process fields in record set
            for field in record_set.get("field", []):
                if "@id" in field:
                    record_row["cr:field"] = [
                        {
                            "data": field["@id"],
                            "extraction_method": extraction_method,
                            "confidence": confidence,
                            "extraction_time": extraction_time,
                        }
                    ]

                if "dataType" in field:
                    record_row["cr:dataType"] = [
                        {
                            "data": field["dataType"],
                            "extraction_method": extraction_method,
                            "confidence": confidence,
                            "extraction_time": extraction_time,
                        }
                    ]

            rows.append(record_row)

        # If no distributions or record sets, add base info as a row
        if not rows:
            rows.append(base_info)

        return rows
