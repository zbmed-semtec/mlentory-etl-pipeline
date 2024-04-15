from datasets import load_dataset
from Utils.MetadataParser import MetadataParser
import pandas as pd
import os

dataset_models = load_dataset("librarian-bots/model_cards_with_metadata")['train']
HF_df = dataset_models.to_pandas()

parser = MetadataParser(qa_model="Intel/dynamic_tinybert")

#Create new columns to answer each question in the dataframe
HF_df = HF_df.iloc[0:10] 

new_columns = {}

for idx in range(len(parser.questions)):
    q_id = "q_id_"+str(idx)
    new_columns[q_id] = [None for _ in range(len(HF_df))]

HF_df = HF_df.assign(**new_columns)

#Fill the new columns
HF_df = parser.parse_fields_from_txt_HF(HF_df=HF_df)
HF_df = parser.parse_fields_from_tags_HF(HF_df=HF_df)
HF_df = parser.parse_known_fields_HF(HF_df=HF_df)

#Remove unecessary columns
HF_df = HF_df.drop(columns=['modelId', 'author', 'last_modified', 'downloads', 'likes',
       'library_name', 'tags', 'pipeline_tag', 'createdAt', 'card'])

#Improve column naming

for q in parser.questions:
    print(q)

def augment_column_name(name:str)->str:
    if("q_id" in name):
        num_id = int(name.split('_')[2])
        return name+"_"+parser.questions[num_id]
    else:
        return name
    
    
HF_df.columns = HF_df.columns.map(augment_column_name)

HF_df.to_csv("Parsed_HF_Dataframe.tsv",sep="\t")
print(HF_df.head())