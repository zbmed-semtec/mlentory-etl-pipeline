from huggingface_hub import hf_hub_download
from huggingface_hub import ModelCard
from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError
import matplotlib.pyplot as plt
import requests
import datetime
from collections import defaultdict

api = HfApi()
        
model_retreival_limit = 500
model_list = list(api.list_models(limit=model_retreival_limit,sort="last_modified",direction=-1))

cnt_models_to_process = 3
found_metadata_list = []

def http_request_to_dict(url):
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for non-2xx status codes

        # Handle various response content types
        if response.headers.get('Content-Type', '').startswith('application/json'):
            return response.json()  # Parse as JSON

        # Handle other content types or return raw content if parsing fails
        return response.text

    except requests.exceptions.RequestException as e:
        print(f"Error making HTTP request: {e}")
        return None

for model in model_list:
    
    if(cnt_models_to_process==0):
        break
    
    response_dict = None
    try:
        model_id_split = model.id.split("/")
        user_tag = model_id_split[0]
        model_name = model_id_split[1]
        query_downloads_url = f"https://huggingface.co/api/models/{user_tag}/{model_name}?expand[]=downloads&expand[]=downloadsAllTime" 
        response_dict = http_request_to_dict(query_downloads_url)
    except EntryNotFoundError:
        print(f"No model card found for {model.id}")
    except Exception as e:
        print(f"An unexpected error occurred: \n {e}")
    
    if(response_dict!=None):
        print("\n##############################################")
        print(user_tag)
        print(model_name)
        print(response_dict)
        print("\n##############################################")
        
        cnt_models_to_process-=1