import torch
from transformers import pipeline
from datasets import Dataset
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass

@dataclass
class QAResult:
    """Stores the result of a QA inference"""
    answer: str
    confidence: float
    extraction_time: str
    extraction_method: str = "Pipeline"

class QAInferenceEngine:
    """Handles model loading and inference for question answering tasks"""
    
    def __init__(self, model_name: str, batch_size: int = 8):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = 0 if torch.cuda.is_available() else None
        self._setup_pipeline()

    def _setup_pipeline(self) -> None:
        """Initialize the QA pipeline with optimizations"""
        if self.device is not None:
            self.pipeline = pipeline(
                "question-answering",
                model=self.model_name,
                device=self.device,
                model_kwargs={"torch_dtype": torch.float16}  # Use FP16 for efficiency
            )
        else:
            self.pipeline = pipeline("question-answering", model=self.model_name)

    @torch.inference_mode()
    def batch_inference(self, questions: List[str], contexts: List[str]) -> List[QAResult]:
        """Process multiple questions and contexts in batches"""
        qa_inputs = [
            {"question": q, "context": c} 
            for q, c in zip(questions, contexts)
        ]
        
        results = []
        extraction_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Process in batches
        for i in range(0, len(qa_inputs), self.batch_size):
            batch = qa_inputs[i:i + self.batch_size]
            answers = self.pipeline(batch)
            
            # Handle single or batch outputs
            if not isinstance(answers, list):
                answers = [answers]
            
            for answer in answers:
                results.append(QAResult(
                    answer=answer["answer"],
                    confidence=answer["score"],
                    extraction_time=extraction_time
                ))
        
        return results

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process a HuggingFace dataset containing questions and contexts"""
        def process_batch(batch):
            answers = self.pipeline(
                [
                    {"question": q, "context": c}
                    for q, c in zip(batch["question"], batch["context"])
                ]
            )
            
            batch["answer"] = [a["answer"] for a in answers]
            batch["score"] = [a["score"] for a in answers]
            batch["extraction_time"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            return batch

        return dataset.map(
            process_batch,
            batched=True,
            batch_size=self.batch_size
        )

    def answer_single_question(self, question: str, context: str) -> QAResult:
        """Process a single question-context pair"""
        with torch.inference_mode():
            answer = self.pipeline({"question": question, "context": context})
            
        return QAResult(
            answer=answer["answer"],
            confidence=answer["score"],
            extraction_time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ) 