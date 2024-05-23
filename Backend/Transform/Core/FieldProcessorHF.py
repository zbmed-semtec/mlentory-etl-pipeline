from typing import Callable, List
import logging
import time
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FieldProcessorHF:
    """
    This class processes the fields of the incoming tsv files and maps it to the M4ML schema.
    """
    def __init__(self, path_to_config_data: str = "Backend/Transform/Config_Data/M4ML_schema.tsv"):
        self.M4ML_schema = pd.read_csv(path_to_config_data, sep="\t")
    
    def process_row(self, row):
        """
        This method processes a row of the incoming tsv file and maps it to the M4ML schema.
        """
        df_M4ML = pd.DataFrame(columns=self.M4ML_schema['Property'].tolist())
                
        #Go through each row of the M4ML_schema
        for index, row_M4ML in self.M4ML_schema.iterrows():
            #Get the property source from the 
            property_source = row_M4ML['Source']
            #Get the column type in the M4ML_schema
            property_source = row_M4ML['Property']
            #Get the column type in the M4ML_schema
            property_source = row_M4ML['Range']
        
        print("Data line: \n", row)
        
        return data_line
    
        