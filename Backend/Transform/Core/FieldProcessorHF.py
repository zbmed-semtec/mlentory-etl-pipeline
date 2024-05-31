from typing import Callable, List, Dict
import logging
import time
import pandas as pd
import json
import ast

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FieldProcessorHF:
    """
    This class processes the fields of the incoming tsv files and maps it to the M4ML schema.
    """
    def __init__(self, path_to_config_data: str = "./../Config_Data"):
        self.M4ML_schema = pd.read_csv(path_to_config_data+"/M4ML_schema.tsv", sep="\t")
        # print(self.M4ML_schema.head())
    
    def process_row(self, row : pd.Series) -> pd.DataFrame:
        """
        This method processes a row of the incoming tsv file and maps it to the M4ML schema.
        """
        df_M4ML = pd.DataFrame(columns=self.M4ML_schema['Property'].tolist())
        row = row.apply(lambda x: [print(x),ast.literal_eval(x)] if isinstance(x, str) else x)
                
        #Go through each row of the M4ML_schema
        for index, row_M4ML in self.M4ML_schema.iterrows():
            #Get the property source from the 
            property_source = row_M4ML['Source']
            #Get the column type in the M4ML_schema
            property_name = row_M4ML['Property']
            #Get the column type in the M4ML_schema
            property_range = row_M4ML['Range']
            new_property = self.process_property(property_description_M4ML = row_M4ML, info_HF = row)
            df_M4ML[property_name] = new_property
        
        # print("Data line: \n", row)
        
        return df_M4ML
        
    def process_property(self,property_description_M4ML: pd.Series, info_HF: pd.Series) -> str:
        """
        This function takes the property description from M4ML and
        the information from HuggingFace and creates a new property value in the M4ML schema.

        Args:
            property_description_M4ML (pd.Series): The description of the property in the M4ML schema.
            info_HF (pd.Series): The information about the property in the HuggingFace schema.

        Returns:
            str: The processed property value.
        """
        # Get the property name and source from the M4ML schema
        property_name = property_description_M4ML['Property']
        # property_source = property_description_M4ML['Source']

        processed_value = ""
        # print(info_HF)
        # Depending on the property name, there will be different processing logic
        if property_name == 'fair4ml:ethicalLegalSocial':
            processed_value = self.find_value_in_HF(info_HF, "q_id_15")
        elif property_name == 'fair4ml:evaluatedOn':
            processed_value = self.find_value_in_HF(info_HF, "q_id_19")
        elif property_name == 'fair4ml:fineTunedFrom':
            processed_value = self.find_value_in_HF(info_HF, "q_id_8")
        elif property_name == 'fair4ml:hasCO2eEmissions':
            processed_value = self.add_default_extraction_info(data="Not extracted",
                                                               extraction_method="None",
                                                               confidence=1.0)
        elif property_name == 'fair4ml:intendedUse':
             processed_value = self.find_value_in_HF(info_HF, "q_id_20")
        elif property_name == 'fair4ml:mlTask':
            processed_value = self.find_value_in_HF(info_HF, "q_id_3")
        # elif property_name == 'fair4ml:modelCategory':
        #     processed_value = process_model_category(info_HF[property_source])
        elif property_name == 'fair4ml:modelRisks':
            processed_value = self.find_value_in_HF(info_HF, "q_id_21")
        elif property_name == 'fair4ml:sharedBy':
            processed_value = self.find_value_in_HF(info_HF, "q_id_1")
        elif property_name == 'fair4ml:testedOn':
            processed_value = self.find_value_in_HF(info_HF, "q_id_4")
        elif property_name == 'fair4ml:trainedOn':
            processed_value = self.process_trainedOn(info_HF)
        elif property_name == 'fair4ml:usageInstructions':
            processed_value = self.find_value_in_HF(info_HF, "q_id_22")
        elif property_name == 'schema.org:distribution':
            processed_value = self.build_distribution_link(info_HF)
        # elif property_name == 'fair4ml:validatedOn':
        #     processed_value = process_dataset(info_HF[property_source])
        # elif property_name == 'schema.org:memoryRequirements':
        #     processed_value = process_text_or_url(info_HF[property_source])
        # elif property_name == 'schema.org:operatingSystem':
        #     processed_value = info_HF[property_source]
        # elif property_name == 'schema.org:processorRequirements':
        #     processed_value = info_HF[property_source]
        # elif property_name == 'schema.org:releaseNotes':
        #     processed_value = process_text_or_url(info_HF[property_source])
        # elif property_name == 'schema.org:softwareHelp':
        #     processed_value = process_creative_work(info_HF[property_source])
        # elif property_name == 'schema.org:softwareRequirements':
        #     processed_value = process_text_or_url(info_HF[property_source])
        # elif property_name == 'schema.org:storageRequirements':
        #     processed_value = info_HF[property_source]
        # elif property_name == 'codemeta:buildInstructions':
        #     processed_value = process_url(info_HF[property_source])
        # elif property_name == 'codemeta:developmentStatus':
        #     processed_value = info_HF[property_source]
        # elif property_name == 'codemeta:issueTracker':
        #     processed_value = process_url(info_HF[property_source])
        # elif property_name == 'codemeta:readme':
        #     processed_value = process_url(info_HF[property_source])
        # elif property_name == 'codemeta:referencePublication':
        #     processed_value = process_scholarly_article(info_HF[property_source])
        # elif property_name == 'schema.org:archivedAt':
        #     processed_value = process_url_or_webpage(info_HF[property_source])
        # elif property_name == 'schema.org:author':
        #     processed_value = process_person_or_org(info_HF[property_source])
        # elif property_name == 'schema.org:citation':
        #     processed_value = process_creative_work_or_text(info_HF[property_source])
        # elif property_name == 'schema.org:conditionsOfAccess':
        #     processed_value = info_HF[property_source]
        # elif property_name == 'schema.org:contributor':
        #     processed_value = process_person_or_org(info_HF[property_source])
        # elif property_name == 'schema.org:copyrightHolder':
        #     processed_value = process_person_or_org(info_HF[property_source])
        # elif property_name == 'schema.org:dateCreated':
        #     processed_value = process_date_or_datetime(info_HF[property_source])
        # elif property_name == 'schema.org:dateModified':
        #     processed_value = process_date_or_datetime(info_HF[property_source])
        # elif property_name == 'schema.org:datePublished':
        #     processed_value = process_date_or_datetime(info_HF[property_source])
        # elif property_name == 'schema.org:discussionUrl':
        #     processed_value = process_url(info_HF[property_source])
        # elif property_name == 'schema.org:funding':
        #     processed_value = process_grant(info_HF[property_source])
        # elif property_name == 'schema.org:inLanguage':
        #     processed_value = process_language_or_text(info_HF[property_source])
        # elif property_name == 'schema.org:isAccessibleForFree':
        #     processed_value = process_boolean(info_HF[property_source])
        # elif property_name == 'schema.org:keywords':
        #     processed_value = process_defined_term_or_text_or_url(info_HF[property_source])
        # elif property_name == 'schema.org:license':
        #     processed_value = process_creative_work_or_url(info_HF[property_source])
        # elif property_name == 'schema.org:maintainer':
        #     processed_value = process_person_or_org(info_HF[property_source])
        # elif property_name == 'schema.org:version':
        #     processed_value = info_HF[property_source]
        # elif property_name == 'schema.org:description':
        #     processed_value = process_text_or_text_object(info_HF[property_source])
        # elif property_name == 'schema.org:identifier':
        #     processed_value = process_property_value_or_text_or_url(info_HF[property_source])
        # elif property_name == 'schema.org:name':
        #     processed_value = info_HF[property_source]
        # elif property_name == 'schema.org:url':
        #     processed_value = process_url(info_HF[property_source])
        # print("Processed value: ",processed_value)
        return processed_value
    
    def process_trainedOn(self,info_HF: pd.DataFrame) -> List:
        """
        Process the trainedOn property of a HF object.
        To process this proper we take into account 3 different values.
        1. Q4 What datasets was the model trained on?
        2. Q6 What datasets were used to finetune the model?
        3. Q7 What datasets were used to retrain the model?
        
        Args:
            info_HF -- The HF object to process

        Return:
            str -- A string representing the list of datasets used to train the model.
        """
        q4_values = self.find_value_in_HF(info_HF,"q_id_4")
        q6_values = self.find_value_in_HF(info_HF,"q_id_6")
        q7_values = self.find_value_in_HF(info_HF,"q_id_7")
        
        processed_values = []
        
        processed_values.extend(q4_values)
        processed_values.extend(q6_values)
        processed_values.extend(q7_values)
        
        print(processed_values)
        
        return processed_values

    def build_distribution_link(self,info_HF: pd.DataFrame) -> str:
        """
        Build the distribution link of a HF object.
        """
        
        return "LOL"
    
    def find_value_in_HF (self,info_HF,property_name):
        """
        Find the value of a property in a HF object.
        """
        
        prefix = property_name
        column_with_prefix = list(filter(lambda x: x.startswith(prefix), info_HF.index))
        processed_value = info_HF[column_with_prefix[0]]
        return processed_value
    
    def add_default_extraction_info(self,data:str,extraction_method:str,confidence:float) -> Dict:
        return str({"data":data,
                    "extraction_method":extraction_method,
                    "confidence":confidence})