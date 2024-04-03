import pandas as pd
from typing import Any, Dict, List, Set, Union

class MetadataParser:

    def __init__(self, qa_pipeline: Any) -> None:
        # Getting the tags
        self.tags_language = self.load_config_file("./Config_Data/tags_language.tsv")
        self.tags_libraries = self.load_config_file("./Config_Data/tags_libraries.tsv")
        self.tags_other = self.load_config_file("./Config_Data/tags_other.tsv")
        self.tags_task = self.load_config_file("./Config_Data/tags_task.tsv")
        #Getting the questions
        self.questions = self.load_config_file("./Config_Data/questions.tsv")
        #Assigning the question answering pipeline
        self.qa_pipeline = qa_pipeline
        
    def load_config_file(self, path: str) -> Set[str]:
        config_info = [val[0].lower() for val in pd.read_csv(path,sep='\t').values.tolist()]
        config_info = set(config_info)
        return config_info

    def answer_question(self, question: str, context: str) -> str:
        answer = self.qa_pipeline({"question": question, "context": context})
        return answer['answer']+''
    
    def parse_known_fields_HF(self, HF_df: pd.DataFrame) -> pd.DataFrame:
        HF_df.loc[:,"q_id_0"] = HF_df.loc[:, ("modelId")]
        HF_df.loc[:,"q_id_1"] = HF_df.loc[:, ("author")]
        HF_df.loc[:,"q_id_2"] = HF_df.loc[:, ("createdAt")]
        # Check if the model was finetuned or retrained
        for index, row in HF_df.iterrows():
            if(row['q_id_8'] != '[CLS]' and row['q_id_8'] != None):
                q_4_answer = HF_df.loc[index:index,"q_id_4"]
                HF_df.loc[index:index,"q_id_6"] = q_4_answer
                HF_df.loc[index:index,"q_id_7"] = q_4_answer
                
        return HF_df
    
    def parse_fields_from_tags_HF(self, HF_df: pd.DataFrame) -> pd.DataFrame:
        for index, row in HF_df.iterrows():
            for tag in row['tags']:
                #Check question 3
                tag_for_q3 = tag.replace('-'," ")
                if tag_for_q3 in self.tags_task:
                    if(HF_df.loc[index, "q_id_3"] == None):
                        HF_df.loc[index, "q_id_3"] = [tag_for_q3]
                    else:
                        HF_df.loc[index, "q_id_3"].append(tag_for_q3)
                #Check question 4
                if "dataset:" in tag:
                    tag_for_q4 = tag.replace("dataset:","")
                    if(HF_df["q_id_4"][index] == None):
                        HF_df.loc[index, "q_id_4"] = [tag_for_q4]
                    else:
                        HF_df["q_id_4"][index].append(tag_for_q4)
                #Check question 13
                if "arxiv:" in tag:
                    tag_for_q13 = tag.replace("arxiv:","")
                    if(HF_df["q_id_13"][index] == None):
                        HF_df.loc[index, "q_id_13"] = [tag_for_q13]
                    else:
                        HF_df.loc[index, "q_id_13"].append(tag_for_q13)
                #Check question 15
                if "license:" in tag:
                    tag_for_q15 = tag.replace("license:","")
                    if(HF_df["q_id_15"][index] == None):
                        HF_df.loc[index, "q_id_15"] = [tag_for_q15]
                    else:
                        HF_df.loc[index, "q_id_15"].append(tag_for_q15)
                #Check question 16
                if tag in self.tags_language:
                    if(HF_df["q_id_16"][index] == None):
                        HF_df.loc[index, "q_id_16"] = [tag]
                    else:
                        HF_df.loc[index, "q_id_16"].append(tag)
                #Check question 17
                if tag in self.tags_libraries:
                    if(HF_df["q_id_17"][index] == None):
                        HF_df.loc[index, "q_id_17"] = [tag]
                    else:
                        HF_df.loc[index, "q_id_17"].append(tag)
            #Check question 3 again
            if(row['pipeline_tag'] != None):
                tag_for_q3 = row['pipeline_tag'].replace("-"," ")
                if(HF_df["q_id_3"][index] == None):
                    HF_df.loc[index, "q_id_3"] = [tag_for_q3]
                else:
                    if(tag_for_q3 not in HF_df["q_id_3"][index]):
                        HF_df.loc[index, "q_id_3"].append(tag_for_q3)
        return HF_df
    
    def parse_fields_from_txt_HF(self, HF_df: pd.DataFrame) -> pd.DataFrame:
        questions_to_process = {5,8,9,10,11,12,14,18}

        for index, row in HF_df.iterrows():
            context = row["card"]  # Getting the context from the "card" column

            # Create an empty dictionary to store answers for each question
            answers = {}
            q_cnt = -1
            for question in self.questions:
                q_cnt+=1
                if(q_cnt not in questions_to_process): continue
                answer = self.answer_question(question, context)
                q_id = "q_id_"+str(q_cnt)
                answers[q_id] = answer  # Store answer for each question
            
            # Add a new column for each question and populate with answers
            for question, answer in answers.items():
                HF_df.loc[index, question] = answer
        
        return HF_df
    
    # def chunker(self, iterable, chunksize):
    #     """Yields chunks of size chunksize from an iterable."""
    #     chunk = []
    #     for item in iterable:
    #         chunk.append(item)
    #         if len(chunk) == chunksize:
    #             yield chunk
    #             chunk = []
    #     if chunk:
    #         yield chunk
    
    # def parse_fields_from_txt_HF(self, HF_df: pd.DataFrame,chunk_size: int) -> pd.DataFrame:
    #     questions_to_process = {5, 8, 9, 10, 11, 12, 14, 18}

    #     # Iterate through the dataframe in chunks
    #     for chunk in self.chunker(HF_df.iterrows(), chunk_size):
    #         batch_df = pd.DataFrame(chunk, columns=["index", "card"])
    #         batch_context = batch_df["card"].tolist()  # Extract contexts in a list

    #         # Create an empty dictionary to store answers for each batch
    #         batch_answers = {}

    #         # Process the questions in a batch using self.answer_question
    #         for question in self.questions:
    #             if question not in questions_to_process:
    #                 continue
    #             # Assuming answer_question can handle a list of contexts
    #             batch_answers[question] = self.answer_question(question, batch_context)

    #         # Update the original dataframe with answers for each chunk
    #         for index, row in chunk:
    #             for question, answer in batch_answers.items():
    #                 q_id = "q_id_" + str(question)
    #                 HF_df.loc[index, q_id] = answer[row["index"]]  # Access answer by index

    #     return HF_df

    