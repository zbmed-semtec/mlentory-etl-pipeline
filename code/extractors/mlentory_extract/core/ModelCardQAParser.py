import pandas as pd
import transformers
from transformers import pipeline
from datasets import Dataset

from typing import Any, Dict, List, Set, Union

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile, RepoFolder


import math
from datetime import datetime
from tqdm import tqdm


class ModelCardQAParser:
    """
    A parser for extracting structured information from model cards using question-answering techniques.

    This class provides functionality to:
    - Extract information using a QA model
    - Parse known fields from HuggingFace datasets
    - Process tags and metadata
    - Handle batch processing of questions
    
    Attributes:
        device: GPU device ID if available, None otherwise
        tags_language (set): Set of supported language tags
        tags_libraries (set): Set of supported ML library tags
        tags_other (set): Set of miscellaneous tags
        tags_task (set): Set of supported ML task tags
        questions (list): List of questions for information extraction
        available_questions (set): Set of question IDs that haven't been processed
        hf_api: HuggingFace API client
        qa_model (str): Name of the QA model being used
        qa_pipeline: Transformer pipeline for question answering
    """

    def __init__(
        self,
        qa_model: str,
        questions: List[str],
        tags_language: List[str],
        tags_libraries: List[str],
        tags_other: List[str],
        tags_task: List[str],
    ) -> None:
        """
        Initialize the Model Card QA Parser

        Args:
            qa_model (str): Model to use for QA-based information extraction
            questions (list[str]): List of questions to extract information
            tags_language (list[str]): List of language tags
            tags_libraries (list[str]): List of library tags
            tags_other (list[str]): List of other tags
            tags_task (list[str]): List of task tags
        """
        # Check for GPU availability
        try:
            import torch

            if torch.cuda.is_available():
                self.device = 0
                print("\nUSING GPU\n")
            else:
                self.device = None
                print("\nNOT USING GPU\n")
        except ModuleNotFoundError:
            # If transformers.torch is not available, assume no GPU
            self.device = None

        # Store configuration data
        self.tags_language = set(tag.lower() for tag in tags_language)
        self.tags_libraries = set(tag.lower() for tag in tags_libraries)
        self.tags_other = set(tag.lower() for tag in tags_other)
        self.tags_task = set(tag.lower() for tag in tags_task)
        self.questions = questions

        self.available_questions = {f"q_id_{id}" for id in range(len(self.questions))}

        # Initializing HF API
        self.hf_api = HfApi()

        # Assigning the question answering pipeline
        self.qa_model = qa_model
        if self.device is not None:
            self.qa_pipeline = pipeline(
                "question-answering", model=qa_model, device=self.device
            )
        else:
            self.qa_pipeline = pipeline("question-answering", model=qa_model)

    def load_tsv_file_to_list(self, path: str) -> Set[str]:
        """
        Load a TSV file and return its contents as a set of lowercase strings.

        Args:
            path (str): Path to the TSV file

        Returns:
            Set[str]: Set of lowercase strings from the first column of the TSV
        """
        config_info = [
            val[0].lower() for val in pd.read_csv(path, sep="\t").values.tolist()
        ]
        # config_info = set(config_info)
        return config_info

    def answer_question(self, question: str, context: str) -> str:
        """
        Extract answer for a given question from the provided context.

        Args:
            question (str): Question to be answered
            context (str): Text context to extract answer from

        Returns:
            list: List containing a dictionary with answer details:
                - data: The extracted answer
                - extraction_method: Name of the QA model used
                - confidence: Confidence score of the answer
                - extraction_time: Timestamp of extraction
        """
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
        """
        Create a standardized dictionary for extraction metadata.

        Args:
            data (str): The extracted information
            extraction_method (str): Method used for extraction
            confidence (float): Confidence score of the extraction

        Returns:
            dict: Dictionary containing extraction metadata
        """
        return {
            "data": data,
            "extraction_method": extraction_method,
            "confidence": confidence,
            "extraction_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }

    def get_repository_weight_HF(self, model_name: str) -> str:
        try:
            model_repo_weight = 0
            model_tree_file_information = self.hf_api.list_repo_tree(
                f"{model_name}", recursive=True
            )
            for x in list(model_tree_file_information):
                if isinstance(x, RepoFile):
                    # The weight of each file is in Bytes.
                    model_repo_weight += x.size
            return f"{model_repo_weight/(math.pow(10,9)):.3f} Gbytes"
        except:
            return "Not available"

    def parse_known_fields_HF(self, HF_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse known fields from HuggingFace dataset into standardized format.

        Args:
            HF_df (pd.DataFrame): DataFrame containing HuggingFace model information

        Returns:
            pd.DataFrame: DataFrame with parsed known fields
        """
        HF_df.loc[:, "q_id_0"] = HF_df.loc[:, ("modelId")]
        HF_df.loc[:, "q_id_1"] = HF_df.loc[:, ("author")]
        HF_df.loc[:, "q_id_2"] = HF_df.loc[:, ("createdAt")].apply(lambda x: str(x))
        HF_df.loc[:, "q_id_26"] = HF_df.loc[:, ("last_modified")].apply(
            lambda x: str(x)
        )
        HF_df.loc[:, "q_id_30"] = HF_df.loc[:, ("card")]

        # Iterate with a progress bar
        for index, row in tqdm(
            HF_df.iterrows(), total=len(HF_df), desc="Processing repository weights"
        ):
            HF_df.loc[index, "q_id_29"] = self.get_repository_weight_HF(
                HF_df.loc[index, "q_id_0"]
            )

        for index in tqdm(range(len(HF_df)), desc="Adding default extraction info"):
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
        """
        Extract information from HuggingFace model tags.

        Args:
            HF_df (pd.DataFrame): DataFrame containing HuggingFace model information

        Returns:
            pd.DataFrame: DataFrame with parsed tag information
        """

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

    def batched_parse_fields_from_txt_HF(self, HF_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process text fields in batches using the QA pipeline.

        Args:
            HF_df (pd.DataFrame): DataFrame containing model information

        Returns:
            pd.DataFrame: DataFrame with extracted information from text fields
        """
        # Create a set of questions to process
        questions_to_process = sorted(
            int(q.split("_")[2]) for q in self.available_questions
        )

        # Create a dataset with questions and contexts
        question_context_dataset = self.create_question_context_dataset(
            HF_df, self.questions, questions_to_process
        )

        # Process the dataset in batches using the QA pipeline
        processed_dataset = self.process_question_context_dataset(
            question_context_dataset, self.qa_pipeline, batch_size=8
        )

        # Merge the results back into the original DataFrame
        HF_df = self.merge_answers_into_dataframe(HF_df, processed_dataset)

        return HF_df

    def create_question_context_dataset(
        self, HF_df: pd.DataFrame, questions: List[str], questions_to_process: Set[int]
    ) -> Dataset:
        """
        Create a dataset of question-context pairs for batch processing.

        Args:
            HF_df (pd.DataFrame): DataFrame containing model information
            questions (List[str]): List of questions to process
            questions_to_process (Set[int]): Set of question IDs to process

        Returns:
            Dataset: HuggingFace dataset containing question-context pairs
        """
        data = []
        for index, row in HF_df.iterrows():
            context = row["card"]
            for q_cnt, question in enumerate(questions):
                if q_cnt in questions_to_process:
                    data.append(
                        {
                            "context": context,
                            "question": question,
                            "row_index": index,
                            "question_id": f"q_id_{q_cnt}",
                        }
                    )

        return Dataset.from_list(data)

    def process_question_context_dataset(
        self, dataset: Dataset, qa_pipeline, batch_size: int = 16
    ) -> Dataset:
        """
        Process a dataset of questions and contexts using the QA pipeline.

        Args:
            dataset (Dataset): Dataset containing question-context pairs
            qa_pipeline: HuggingFace pipeline for question answering
            batch_size (int, optional): Size of batches for processing. Defaults to 16.

        Returns:
            Dataset: Dataset with added answer information
        """
        extraction_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        def process_batch(batch):
            # Run the pipeline in batch mode
            answers = qa_pipeline(
                [
                    {"question": q, "context": c}
                    for q, c in zip(batch["question"], batch["context"])
                ]
            )

            # Extract and structure answers
            batch["answer"] = [a["answer"] for a in answers]
            batch["score"] = [a["score"] for a in answers]
            batch["extraction_time"] = extraction_time
            return batch

        return dataset.map(process_batch, batched=True, batch_size=batch_size)

    def merge_answers_into_dataframe(
        self, HF_df: pd.DataFrame, processed_dataset: Dataset
    ) -> pd.DataFrame:
        """
        Merge processed answers back into the original DataFrame.

        Args:
            HF_df (pd.DataFrame): Original DataFrame
            processed_dataset (Dataset): Dataset containing processed answers

        Returns:
            pd.DataFrame: DataFrame with merged answer information
        """
        # Group answers by row_index and question_id
        grouped_answers = processed_dataset.to_pandas().groupby("row_index")

        for row_index, group in grouped_answers:
            for _, row in group.iterrows():
                question_id = row["question_id"]
                HF_df.at[row_index, question_id] = [
                    {
                        "data": row["answer"],
                        "confidence": row["score"],
                        "extraction_method": "Pipeline",
                        "extraction_time": row["extraction_time"],
                    }
                ]

        return HF_df

    def parse_fields_from_txt_HF(self, HF_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract information from text fields using QA model.

        Args:
            HF_df (pd.DataFrame): DataFrame containing model information

        Returns:
            pd.DataFrame: DataFrame with extracted information from text fields
        """
        questions_to_process = set()
        for q in self.available_questions:
            # print(q.split("_")[2])
            questions_to_process.add(int(q.split("_")[2]))

        for index, row in tqdm(
            HF_df.iterrows(), total=len(HF_df), desc="Parsing text fields"
        ):
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
