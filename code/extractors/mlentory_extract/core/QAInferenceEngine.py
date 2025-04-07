import torch
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasets import Dataset

torch.backends.cuda.matmul.allow_tf32 = True
from transformers import pipeline


# Define the prompt template
PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Based *only* on the context provided above, answer the question. If the context does not contain the information needed to answer the question, respond with the exact phrase "Information not found".

Answer:"""

# Define the phrase indicating information wasn't found
INFO_NOT_FOUND_PHRASE = "Information not found"


@dataclass
class QAResult:
    """Stores the result of a QA inference using text generation."""

    answer: str
    extraction_time: str
    confidence: Optional[float] = None
    extraction_method: str = "Text Generation"


class QAInferenceEngine:
    """Handles model loading and inference for question answering tasks using text generation."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 4,
        max_new_tokens: int = 2000,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.device = 0 if torch.cuda.is_available() else -1
        self._setup_pipeline()

    def _setup_pipeline(self) -> None:
        """Initialize the text-generation pipeline with optimizations"""
        print(f"Setting up text-generation pipeline for model: {self.model_name}")
        # Load tokenizer and model for causal LM
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Set padding token if not already set (common for models like Llama, GPT-NeoX)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Ensure padding side is left for generation
        tokenizer.padding_side = "left"
            
        model_kwargs = {}
        if self.device == 0: # Only specify torch_dtype if using GPU
            model_kwargs["torch_dtype"] = torch.float16
            # Optional: Add device_map="auto" for multi-GPU or large models if needed
            # model_kwargs["device_map"] = "auto"

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                **model_kwargs
            )

            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                batch_size=self.batch_size,
                # Generation parameters
                max_new_tokens=self.max_new_tokens,
                # Common parameters to control generation quality/determinism if needed:
                # temperature=0.7,
                # top_p=0.9,
                # do_sample=True, # Set to False for deterministic output if preferred
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            print("Text-generation pipeline setup complete.")
        except Exception as e:
            print(f"Error setting up text-generation pipeline: {e}")
            raise e

    @torch.inference_mode()
    def batch_inference(
        self, questions: List[str], contexts: List[str]
    ) -> List[QAResult]:
        """Process multiple questions and contexts in batches using text generation."""
        if len(questions) != len(contexts):
            raise ValueError("Number of questions and contexts must be the same.")

        # Prepare prompts for the batch
        prompts = [
            PROMPT_TEMPLATE.format(context=c, question=q)
            for q, c in zip(questions, contexts)
        ]

        results = []
        extraction_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Process in batches with progress bar
        total_steps = (len(prompts) + self.batch_size - 1) // self.batch_size
        print(f"Starting batch inference with {len(prompts)} items, batch size {self.batch_size}...")

        pipeline_outputs = []
        try:
            for output in tqdm(self.pipeline(prompts, batch_size=self.batch_size), total=len(prompts), desc="Processing Generation Batches"):
                 pipeline_outputs.append(output)

        except Exception as e:
             print(f"Error during pipeline execution: {e}")
             # Return partial results or raise error depending on desired behavior
             # For now, return empty list if error occurs mid-batch
             return []

        print("Parsing generated outputs...")
        # Parse the generated outputs
        for i, output_list in enumerate(pipeline_outputs):
            # Handle cases where pipeline returns a list per item or a single dict
            if isinstance(output_list, list):
                 # Take the first generated sequence if multiple are returned
                 output = output_list[0] if output_list else {"generated_text": ""}
            elif isinstance(output_list, dict):
                 output = output_list
            else:
                 print(f"Warning: Unexpected output format at index {i}: {type(output_list)}")
                 output = {"generated_text": ""}

            generated_text = output.get("generated_text", "")

            # Extract the answer part (text after the prompt)
            # Find the "Answer:" marker in the original prompt to split accurately
            prompt_used = prompts[i]
            answer_marker = "Answer:"
            answer_start_index = prompt_used.rfind(answer_marker)

            if answer_start_index != -1:
                # Check if the generated text includes the prompt
                if generated_text.startswith(prompt_used):
                     final_answer = generated_text[len(prompt_used):].strip()
                else:
                     # Sometimes models only generate the part after the marker
                     # Find the marker in the *generated* text if the prompt isn't repeated
                     gen_answer_start = generated_text.rfind(answer_marker)
                     if gen_answer_start != -1:
                          final_answer = generated_text[gen_answer_start + len(answer_marker):].strip()
                     else:
                           # Fallback: assume the whole generation is the answer if marker isn't found
                           final_answer = generated_text.strip()
            else:
                 # Fallback if marker logic fails
                 final_answer = generated_text.strip() # Use the full generated text if marker fails

            # Check if the model indicated information not found
            if final_answer.strip() == INFO_NOT_FOUND_PHRASE:
                answer_to_store = INFO_NOT_FOUND_PHRASE
            else:
                answer_to_store = final_answer # Store the extracted answer

            results.append(
                QAResult(
                    answer=answer_to_store,
                    extraction_time=extraction_time,
                    # Confidence is None for text-generation
                )
            )
            
        print(f"Batch inference complete. Processed {len(results)} items.")
        return results

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process a HuggingFace dataset containing questions and contexts"""
        if not all(col in dataset.column_names for col in ["question", "context"]):
             raise ValueError("Dataset must contain 'question' and 'context' columns.")

        questions = dataset["question"]
        contexts = dataset["context"]

        results = self.batch_inference(questions, contexts)

        # Prepare data for the new dataset
        result_data = {
            "answer": [r.answer for r in results],
            "extraction_time": [r.extraction_time for r in results],
        }

        # Keep original columns, ensure alignment if results mismatch input length
        original_cols_to_keep = [col for col in dataset.column_names if col not in result_data]
        num_results = len(results)

        for col in original_cols_to_keep:
             # Truncate original data if results are fewer (e.g., due to errors)
             result_data[col] = dataset[col][:num_results]

        if num_results < len(questions):
             print(f"Warning: Number of results ({num_results}) is less than input items ({len(questions)}). Dataset truncated.")

        return Dataset.from_dict(result_data)

    @torch.inference_mode()
    def answer_single_question(self, question: str, context: str) -> QAResult:
        """Process a single question-context pair using text generation."""
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        try:
             output_list = self.pipeline(prompt)
        except Exception as e:
             print(f"Error during single inference: {e}")
             return QAResult(answer="Error during generation", extraction_time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        # Handle output format variations
        if isinstance(output_list, list):
            output = output_list[0] if output_list else {"generated_text": ""}
        elif isinstance(output_list, dict):
            output = output_list
        else:
            print(f"Warning: Unexpected output format for single inference: {type(output_list)}")
            output = {"generated_text": ""}

        generated_text = output.get("generated_text", "")

        # Extract the answer part
        answer_marker = "Answer:"
        answer_start_index = prompt.rfind(answer_marker)

        if answer_start_index != -1:
             if generated_text.startswith(prompt):
                  final_answer = generated_text[len(prompt):].strip()
             else:
                  gen_answer_start = generated_text.rfind(answer_marker)
                  if gen_answer_start != -1:
                       final_answer = generated_text[gen_answer_start + len(answer_marker):].strip()
                  else:
                        final_answer = generated_text.strip()
        else:
              final_answer = generated_text.strip()

        # Check for "Information not found"
        answer_to_store = INFO_NOT_FOUND_PHRASE if final_answer.strip() == INFO_NOT_FOUND_PHRASE else final_answer

        return QAResult(
            answer=answer_to_store,
            extraction_time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
