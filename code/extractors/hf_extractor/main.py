from core.MetadataParser import MetadataParser
from datasets import load_dataset
from datetime import datetime
import pandas as pd
import os

if __name__ == "__main__":
    dataset_models = load_dataset("librarian-bots/model_cards_with_metadata")["train"]
    original_HF_df = dataset_models.to_pandas()

    # Creating the parser object that will perform the transformations on the raw data
    parser = MetadataParser(qa_model="Intel/dynamic_tinybert")

    # Create new columns to answer each question in the dataframe
    HF_df = original_HF_df.iloc[0:15]

    new_columns = {}

    for idx in range(len(parser.questions)):
        q_id = "q_id_" + str(idx)
        new_columns[q_id] = [None for _ in range(len(HF_df))]

    HF_df = HF_df.assign(**new_columns)

    # Fill the new columns
    HF_df = parser.parse_fields_from_tags_HF(HF_df=HF_df)
    HF_df = parser.parse_known_fields_HF(HF_df=HF_df)
    HF_df = parser.parse_fields_from_txt_HF(HF_df=HF_df)

    # Remove unnecessary columns, the information is contained int the
    HF_df = HF_df.drop(
        columns=[
            "modelId",
            "author",
            "last_modified",
            "downloads",
            "likes",
            "library_name",
            "tags",
            "pipeline_tag",
            "createdAt",
            "card",
        ]
    )

    # Improve column naming
    for q in parser.questions:
        print(q)

    def augment_column_name(name: str) -> str:
        if "q_id" in name:
            num_id = int(name.split("_")[2])
            return name + "_" + parser.questions[num_id]
        else:
            return name

    HF_df.columns = HF_df.columns.map(augment_column_name)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Get current date and time

    modified_hf_filename = f"./../transform_queue/{now}_Parsed_HF_Dataframe.json"
    original_hf_filename = f"./../datasets/{now}_Parsed_HF_Dataframe.json"

    original_HF_df.to_csv(original_hf_filename, sep="\t")
    HF_df.to_json(path_or_buf=modified_hf_filename, orient="records", indent=4)
    print(HF_df.head())
