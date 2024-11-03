import pandas as pd
import transformers
from transformers import pipeline
from typing import Any, Dict, List, Set, Union

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile, RepoFolder
import math
from datetime import datetime


class MetadataParser:

    def __init__(
        self, qa_model: str, path_to_config_data: str = "./../config_data"
    ) -> None:
        # Check for GPU availability
        try:
            import torch  # Import torch to check for cuda availability

            if torch.cuda.is_available():
                self.device = 0  # Use GPU if available
            else:
                self.device = None  # Don't specify device if no GPU
        except ModuleNotFoundError:
            # If transformers.torch is not available, assume no GPU
            self.device = None

        # Getting the tags
        self.tags_language = set(
            self.load_config_file(f"{path_to_config_data}/tags_language.tsv")
        )
        self.tags_libraries = set(
            self.load_config_file(f"{path_to_config_data}/tags_libraries.tsv")
        )
        self.tags_other = set(
            self.load_config_file(f"{path_to_config_data}/tags_other.tsv")
        )
        self.tags_task = set(
            self.load_config_file(f"{path_to_config_data}/tags_task.tsv")
        )
        # Getting the questions
        self.questions = self.load_config_file(f"{path_to_config_data}/questions.tsv")
        self.available_questions = set()
        for id in range(len(self.questions)):
            self.available_questions.add("q_id_" + str(id))

        # Initializing HF API
        self.hf_api = HfApi()

        # Assigning the question answering pipeline
        self.qa_model = qa_model
        if self.device != None:
            self.qa_pipeline = pipeline(
                "question-answering", model=qa_model, device=self.device
            )
        else:
            self.qa_pipeline = pipeline("question-answering", model=qa_model)

    def load_config_file(self, path: str) -> Set[str]:
        config_info = [
            val[0].lower() for val in pd.read_csv(path, sep="\t").values.tolist()
        ]
        # config_info = set(config_info)
        return config_info

    def answer_question(self, question: str, context: str) -> str:
        answer = self.qa_pipeline({"question": question, "context": context})
        return [
            {
                "data": answer["answer"] + "",
                "extraction_method": self.qa_model,
                "confidence": answer["score"],
                "extraction_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            }
        ]

    def add_default_extraction_info(
        self, data: str, extraction_method: str, confidence: float
    ) -> Dict:
        return {
            "data": data,
            "extraction_method": extraction_method,
            "confidence": confidence,
            "extraction_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }

    def get_repository_weight_HF(self, model_name: str) -> str:
        model_repo_weight = 0
        model_tree_file_information = self.hf_api.list_repo_tree(
            f"{model_name}", recursive=True
        )
        for x in list(model_tree_file_information):
            if isinstance(x, RepoFile):
                # The weight of each file is in Bytes.
                model_repo_weight += x.size
        return f"{model_repo_weight/(math.pow(10,9)):.3f} Gbytes"

    def parse_known_fields_HF(self, HF_df: pd.DataFrame) -> pd.DataFrame:
        HF_df.loc[:, "q_id_0"] = HF_df.loc[:, ("modelId")]
        HF_df.loc[:, "q_id_1"] = HF_df.loc[:, ("author")]
        HF_df.loc[:, "q_id_2"] = HF_df.loc[:, ("createdAt")].apply(lambda x: str(x))
        HF_df.loc[:, "q_id_26"] = HF_df.loc[:, ("last_modified")].apply(
            lambda x: str(x)
        )
        HF_df.loc[:, "q_id_30"] = HF_df.loc[:, ("card")]

        # Check if the model was finetuned or retrained
        # q_id_8 asks What model is used as the base model?
        # q_id_4 asks What datasets was the model trained on?
        # q_id_6 asks What datasets were used to finetune the model?
        # q_id_7 asks What datasets were used to retrain the model?
        for index, row in HF_df.iterrows():
            if row["q_id_8"] != "[CLS]" and row["q_id_8"] != None:
                q_4_answer = HF_df.loc[index:index, "q_id_4"]
                HF_df.loc[index:index, "q_id_6"] = q_4_answer
                HF_df.loc[index:index, "q_id_7"] = q_4_answer

        for index, row in HF_df.iterrows():
            HF_df.loc[index, "q_id_29"] = self.get_repository_weight_HF(
                HF_df.loc[index, "q_id_0"]
            )

        for index in range(len(HF_df)):
            for id in [
                "q_id_0",
                "q_id_1",
                "q_id_2",
                "q_id_6",
                "q_id_7",
                "q_id_26",
                "q_id_29",
                "q_id_30",
            ]:
                HF_df.loc[index, id] = [
                    self.add_default_extraction_info(
                        HF_df.loc[index, id], "Parsed_from_HF_dataset", 1.0
                    )
                ]

        for id in [
            "q_id_0",
            "q_id_1",
            "q_id_2",
            "q_id_6",
            "q_id_7",
            "q_id_26",
            "q_id_29",
            "q_id_30",
        ]:
            self.available_questions.discard(id)

        return HF_df

    def parse_fields_from_tags_HF(self, HF_df: pd.DataFrame) -> pd.DataFrame:

        for index, row in HF_df.iterrows():
            for tag in row["tags"]:
                # Check question 3
                tag_for_q3 = tag.replace("-", " ").lower()
                if tag_for_q3 in self.tags_task:
                    if HF_df.loc[index, "q_id_3"] == None:
                        HF_df.loc[index, "q_id_3"] = [tag_for_q3]
                    else:
                        HF_df.loc[index, "q_id_3"].append(tag_for_q3)
                # Check question 4
                if "dataset:" in tag:
                    tag_for_q4 = tag.replace("dataset:", "")
                    if HF_df["q_id_4"][index] == None:
                        HF_df.loc[index, "q_id_4"] = [tag_for_q4]
                    else:
                        HF_df["q_id_4"][index].append(tag_for_q4)
                # Check question 13
                if "arxiv:" in tag:
                    tag_for_q13 = tag.replace("arxiv:", "")
                    if HF_df["q_id_13"][index] == None:
                        HF_df.loc[index, "q_id_13"] = [tag_for_q13]
                    else:
                        HF_df.loc[index, "q_id_13"].append(tag_for_q13)
                # Check question 15
                if "license:" in tag:
                    tag_for_q15 = tag.replace("license:", "")
                    if HF_df["q_id_15"][index] == None:
                        HF_df.loc[index, "q_id_15"] = [tag_for_q15]
                    else:
                        HF_df.loc[index, "q_id_15"].append(tag_for_q15)
                # Check question 16
                if tag in self.tags_language:
                    if HF_df["q_id_16"][index] == None:
                        HF_df.loc[index, "q_id_16"] = [tag]
                    else:
                        HF_df.loc[index, "q_id_16"].append(tag)
                # Check question 17
                tag_for_q17 = tag.lower()
                if tag_for_q17 in self.tags_libraries:
                    if HF_df["q_id_17"][index] == None:
                        HF_df.loc[index, "q_id_17"] = [tag_for_q17]
                    else:
                        HF_df.loc[index, "q_id_17"].append(tag_for_q17)
            # Check question 3 in pipeline_tags
            if row["pipeline_tag"] != None:
                tag_for_q3 = row["pipeline_tag"].replace("-", " ").lower()
                if HF_df["q_id_3"][index] == None:
                    HF_df.loc[index, "q_id_3"] = [tag_for_q3]
                else:
                    if tag_for_q3 not in HF_df["q_id_3"][index]:
                        HF_df.loc[index, "q_id_3"].append(tag_for_q3)

        for index in range(len(HF_df)):
            for id in ["q_id_3", "q_id_4", "q_id_13", "q_id_15", "q_id_16", "q_id_17"]:
                HF_df.loc[index, id] = [
                    self.add_default_extraction_info(
                        HF_df.loc[index, id], "Parsed_from_HF_dataset", 1.0
                    )
                ]

        for id in ["q_id_3", "q_id_4", "q_id_13", "q_id_15", "q_id_16", "q_id_17"]:
            self.available_questions.discard(id)

        return HF_df

    def parse_fields_from_txt_HF(self, HF_df: pd.DataFrame) -> pd.DataFrame:
        questions_to_process = set()
        for q in self.available_questions:
            print(q.split("_")[2])
            questions_to_process.add(int(q.split("_")[2]))

        for index, row in HF_df.iterrows():
            context = row["card"]  # Getting the context from the "card" column
            # Create an empty dictionary to store answers for each question
            answers = {}

            q_cnt = -1
            for question in self.questions:
                q_cnt += 1
                if q_cnt not in questions_to_process:
                    continue
                answer = self.answer_question(question, context)
                q_id = "q_id_" + str(q_cnt)
                answers[q_id] = answer  # Store answer for each question

            # Add a new column for each question and populate with answers
            for question, answer in answers.items():
                # print(answer)
                HF_df.loc[index, question] = [answer]

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
