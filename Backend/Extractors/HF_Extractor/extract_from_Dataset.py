from transformers import pipeline
from datasets import load_dataset
from Utils.MetadataParser import MetadataParser
import pandas as pd
import os

def load_config_file(path):
    config_info = [val[0].lower() for val in pd.read_csv(path,sep='\t').values.tolist()]
    config_info = set(config_info)
    return config_info


dataset_models = load_dataset("librarian-bots/model_cards_with_metadata")['train']
HF_df = dataset_models.to_pandas()

qa_pipeline = pipeline("question-answering", model="Intel/dynamic_tinybert",device=0)
parser = MetadataParser(qa_pipeline)

    
# Getting the tags
tags_language = load_config_file("./Config_Data/tags_language.tsv")
tags_libraries = load_config_file("./Config_Data/tags_libraries.tsv")
tags_other = load_config_file("./Config_Data/tags_other.tsv")
tags_task = load_config_file("./Config_Data/tags_task.tsv")

#Getting the questions
questions = load_config_file("./Config_Data/questions.tsv")

#Create new columns to answer each question in the dataframe
HF_df_small = HF_df.iloc[0:10] 

new_columns = {}

for idx in range(len(questions)):
    q_id = "q_id_"+str(idx)
    new_columns[q_id] = [None for _ in range(len(HF_df_small))]

HF_df_small = HF_df_small.assign(**new_columns)

print(HF_df_small.head())




