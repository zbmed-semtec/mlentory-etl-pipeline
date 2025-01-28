import torch

torch.backends.cuda.matmul.allow_tf32 = True
from transformers import pipeline
from datasets import Dataset
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass
import transformers
from transformers import AutoTokenizer
from tqdm import tqdm


@dataclass
class QAResult:
    """Stores the result of a QA inference"""

    answer: str
    confidence: float
    extraction_time: str
    extraction_method: str = "Pipeline"


class QAInferenceEngine:
    """Handles model loading and inference for question answering tasks"""

    def __init__(self, model_name: str, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = 0 if torch.cuda.is_available() else None
        self._setup_pipeline()

    def _setup_pipeline(self) -> None:
        """Initialize the QA pipeline with optimizations"""
        if self.device is not None:
            if self.model_name == "mosaicml/mpt-7b":

                model = transformers.AutoModelForQuestionAnswering.from_pretrained(
                    "mosaicml/mpt-7b", trust_remote_code=True
                )

                tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

                self.pipeline = pipeline(
                    "question-answering",
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device,
                    batch_size=self.batch_size,
                    model_kwargs={
                        "torch_dtype": torch.float16
                    },  # Use FP16 for efficiency
                )

            else:
                model = transformers.AutoModelForQuestionAnswering.from_pretrained(
                    self.model_name
                )
                model.half()
                model.to(self.device)

                tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                self.pipeline = pipeline(
                    "question-answering",
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device,
                    batch_size=self.batch_size,
                    model_kwargs={
                        "torch_dtype": torch.float16
                    },  # Use FP16 for efficiency
                )

        else:
            self.pipeline = pipeline(
                "question-answering", model=self.model_name, batch_size=self.batch_size
            )

    @torch.inference_mode()
    def batch_inference(
        self, questions: List[str], contexts: List[str]
    ) -> List[QAResult]:
        """Process multiple questions and contexts in batches"""
        qa_inputs = [{"question": q, "context": c} for q, c in zip(questions, contexts)]

        results = []
        extraction_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Process in batches with progress bar
        total_batches = (len(qa_inputs) + self.batch_size - 1) // self.batch_size
        for i in tqdm(
            range(0, len(qa_inputs), self.batch_size),
            total=total_batches,
            desc="Processing QA batches",
        ):
            batch = qa_inputs[i : i + self.batch_size]
            answers = self.pipeline(batch)

            # Handle single or batch outputs
            if not isinstance(answers, list):
                answers = [
                    QAResult(
                        answer=answers["answer"],
                        confidence=answers["score"],
                        extraction_time=extraction_time,
                    )
                ]

            for answer in answers:
                results.append(
                    QAResult(
                        answer=answer["answer"],
                        confidence=answer["score"],
                        extraction_time=extraction_time,
                    )
                )

        return results

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process a HuggingFace dataset containing questions and contexts"""
        # Convert dataset to lists for batch processing
        questions = dataset["question"]
        contexts = dataset["context"]

        # Use our existing batch_inference method which handles batching properly
        results = self.batch_inference(questions, contexts)

        # Convert results back to dataset format
        return Dataset.from_dict(
            {
                "answer": [r.answer for r in results],
                "score": [r.confidence for r in results],
                "extraction_time": [r.extraction_time for r in results],
                "question": dataset["question"],
                "context": dataset["context"],
                "row_index": dataset["row_index"],
                "question_id": dataset["question_id"],
            }
        )

    @torch.inference_mode()
    def answer_single_question(self, question: str, context: str) -> QAResult:
        """Process a single question-context pair"""
        with torch.inference_mode():
            answer = self.pipeline({"question": question, "context": context})

        return QAResult(
            answer=answer["answer"],
            confidence=answer["score"],
            extraction_time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
