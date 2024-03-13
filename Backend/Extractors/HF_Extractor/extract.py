from huggingface_hub import hf_hub_download
from huggingface_hub import ModelCard
from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError
import matplotlib.pyplot as plt
import requests
import datetime
from collections import defaultdict

api = HfApi()

model_retreival_limit = 200
model_list = list(api.list_models(limit=model_retreival_limit,sort="last_modified",direction=-1))

cnt_models_to_process = 3
found_metadata_list = []

for model in model_list:
    
    if(cnt_models_to_process==0):
        break
    
    model_card_data = None
    try:
        model_card_data = ModelCard.load(model.id)
    except EntryNotFoundError:
        print(f"Error loading model card, no model card found for {model.id}")
    
    if(model_card_data!=None):
        print("\n##############################################")
        print(model.last_modified)
        print(model.id)
        print(card.data.to_dict())
        print("\n##############################################")
        
        cnt_models_to_process-=1