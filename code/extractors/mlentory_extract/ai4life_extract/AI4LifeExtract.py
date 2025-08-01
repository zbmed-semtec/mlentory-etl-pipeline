import json
import requests
import os
from datetime import datetime

class AI4LifeExtractor:
    def __init__(self):
       
        self.base_url = "https://hypha.aicell.io"
        self.parent_id = "bioimage-io/bioimage.io"
        
    def fetch_reords(self, num_models:int):  
         
        # List available records
        response = requests.get(
            f"{self.base_url}/public/services/artifact-manager/list",
            params={
                "parent_id": self.parent_id,
                "limit": num_models
            }
        )
        return response.json()
    
    def save_records(self, records, output_dir):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(output_dir, 'ai4life_metadata_record_{timestamp}.json')
         # 2. Write to a JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(records,            # the Python object to serialize
                f,               # file handle
                indent=4,        # pretty‑print with 4‑space indentation
                ensure_ascii=False) #allow non‑ASCII characters if needed 
        
        
              
    def download_modelfiles_with_additional_entities(self, num_models:int, output_dir:str = "./output"):
        """Example: Download model files"""
        records = self.fetch_reords(num_models)
        self.save_records(records,output_dir)
        # print("Available models:", json.dumps(records, indent=2))
       
            