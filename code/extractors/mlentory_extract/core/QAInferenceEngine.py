import torch
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasets import Dataset
import re

# Add vLLM imports
from vllm import LLM, SamplingParams

torch.backends.cuda.matmul.allow_tf32 = True


# Define the prompt template
PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Based *only* on the context provided above, answer the question. If the context does not contain the information needed to answer the question, respond with the exact phrase "Information not found".

Answer:"""

# Define the multi-question prompt template
MULTI_QUESTION_TEMPLATE = """Context:
{context}

I have multiple questions about this context. For each question, provide a concise answer based ONLY on the information in the context above.
If the context does not contain the information needed to answer a question, respond with "Information not found" for that question.

{questions}

For each question, provide your answer in the format:
Question [number]: [Your answer]
"""

# Define the phrase indicating information wasn't found
INFO_NOT_FOUND_PHRASE = "Information not found"


@dataclass
class QAResult:
    """Stores the result of a QA inference using text generation."""

    answer: str
    extraction_time: str
    confidence: Optional[float] = None
    extraction_method: str = "Text Generation"


@dataclass
class QABatchResult:
    """Stores the results of a batch QA inference with multiple questions for the same context."""
    
    question_answers: Dict[str, str]  # Maps questions to their answers
    extraction_time: str
    context: str
    confidence: Optional[float] = None
    extraction_method: str = "Batch Text Generation"


class QAInferenceEngine:
    """Handles model loading and inference for question answering tasks using text generation with vLLM."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 4, # vLLM handles batching internally, this might be less relevant or used differently
        max_new_tokens: int = 2000,
        tensor_parallel_size: int = 1, # vLLM specific
        dtype: str = "auto", # vLLM specific, e.g., "half" for float16
        trust_remote_code: bool = True, # For tokenizer and model loading
        # Common sampling parameters for vLLM
        temperature: float = 0.0, # For deterministic output by default
        top_p: float = 1.0,
        # stop_sequences: Optional[List[str]] = None, # Can be added if needed
    ):
        self.model_name = model_name
        # self.batch_size = batch_size # vLLM handles its own batching.
        self.max_new_tokens = max_new_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        
        # Store sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=self.max_new_tokens, # vLLM uses max_tokens
            # stop=stop_sequences if stop_sequences else [] 
        )

        
        self._setup_engine() # Renamed from _setup_pipeline

    def _setup_engine(self) -> None:
        """Initialize the vLLM engine."""
        print(f"Setting up vLLM engine for model: {self.model_name}")
        try:

            self.llm = LLM(
                model=self.model_name,
                tokenizer=self.model_name, # vLLM can load tokenizer by name
                tensor_parallel_size=self.tensor_parallel_size,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                # Add other vLLM specific options if needed, e.g., quantization
            )
            print("vLLM engine setup complete.")
        except Exception as e:
            print(f"Error setting up vLLM engine: {e}")
            raise e

    @torch.inference_mode() # Keep for consistency, though vLLM manages its own inference mode
    def batch_inference(
        self, questions: List[str], contexts: List[str]
    ) -> List[QAResult]:
        """Process multiple questions and contexts in batches using vLLM."""
        if len(questions) != len(contexts):
            raise ValueError("Number of questions and contexts must be the same.")

        # Prepare prompts for the batch
        prompts = [
            PROMPT_TEMPLATE.format(context=c, question=q)
            for q, c in zip(questions, contexts)
        ]

        results = []
        extraction_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if not prompts:
            return []

        print(f"Starting vLLM batch inference with {len(prompts)} items...")
        
        try:
            # vLLM's generate method takes a list of prompts
            outputs = self.llm.generate(prompts, self.sampling_params)
        except Exception as e:
             print(f"Error during vLLM generation: {e}. Returning empty list.")
             return []

        print("Parsing generated outputs from vLLM...")

        # We typically take the first CompletionOutput.
        for i, request_output in enumerate(outputs):
            # Assuming we take the first completion if multiple are generated per prompt (n > 1)
            if request_output.outputs:
                generated_text = request_output.outputs[0].text
            else:
                generated_text = "" # Should not happen if generation was successful

            # The prompt is NOT included in vLLM's generated_text by default.
            # The text is only the newly generated tokens.
            final_answer = generated_text.strip()

            # Check if the model indicated information not found
            if final_answer == INFO_NOT_FOUND_PHRASE:
                answer_to_store = INFO_NOT_FOUND_PHRASE
            else:
                answer_to_store = final_answer

            results.append(
                QAResult(
                    answer=answer_to_store,
                    extraction_time=extraction_time,
                    # Confidence is None for text-generation
                )
            )
            
        print(f"vLLM Batch inference complete. Processed {len(results)} items.")
        return results

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process a HuggingFace dataset containing questions and contexts using vLLM"""
        if not all(col in dataset.column_names for col in ["question", "context"]):
             raise ValueError("Dataset must contain 'question' and 'context' columns.")

        questions = dataset["question"]
        contexts = dataset["context"]

        results = self.batch_inference(questions, contexts) # Uses the new vLLM batch_inference

        # Prepare data for the new dataset
        result_data = {
            "answer": [r.answer for r in results],
            "extraction_time": [r.extraction_time for r in results],
            "extraction_method": [r.extraction_method for r in results], # Add this
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
        """Process a single question-context pair using vLLM."""
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        extraction_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        try:
             # vLLM generate expects a list of prompts, even for a single one.
             outputs = self.llm.generate([prompt], self.sampling_params)
        except Exception as e:
             print(f"Error during single vLLM inference: {e}")
             return QAResult(
                 answer="Error during generation", 
                 extraction_time=extraction_time,
                 extraction_method="Text Generation (vLLM Error)"
            )

        if outputs and outputs[0].outputs:
            generated_text = outputs[0].outputs[0].text
        else:
            generated_text = ""
        
        final_answer = generated_text.strip()

        # Check for "Information not found"
        answer_to_store = INFO_NOT_FOUND_PHRASE if final_answer == INFO_NOT_FOUND_PHRASE else final_answer

        return QAResult(
            answer=answer_to_store,
            extraction_time=extraction_time,
            extraction_method="Text Generation (vLLM)"
        )

    @torch.inference_mode()
    def batch_questions_single_context(
        self, questions: List[str], context: str
    ) -> List[QAResult]:
        """
        Process multiple questions with a single shared context in one prompt using vLLM.
        
        Args:
            questions (List[str]): List of questions to answer
            context (str): The shared context for all questions
            
        Returns:
            List[QAResult]: List of QA results, one for each question
        """
        if not questions:
            return []
            
        if not context or not isinstance(context, str) or context.strip() == "":
            # Create error results for all questions if context is invalid
            error_results = []
            extraction_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for q_idx in range(len(questions)):
                error_results.append(
                    QAResult(
                        answer="Error: Invalid context provided",
                        extraction_time=extraction_time,
                        extraction_method=f"Batch Text Generation (vLLM Error, Q{q_idx+1}/{len(questions)})"
                    )
                )
            return error_results
            
        # Format the questions with numbers
        formatted_questions = "\n".join([f"Question {i+1}: {q}" for i, q in enumerate(questions)])
        
        # Create the prompt with all questions
        prompt = MULTI_QUESTION_TEMPLATE.format(
            context=context,
            questions=formatted_questions
        )
        
        print(f"\n\nPrompt for vLLM multi-question: {prompt}\n\n")
        extraction_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Generate response
        try:
            print(f"Generating response for multi-question prompt using vLLM:")
            # vLLM generate expects a list of prompts
            outputs = self.llm.generate([prompt], self.sampling_params)
        except Exception as e:
            print(f"Error during multi-question vLLM inference: {e}")
            # Return error results for all questions
            error_results = []
            for q_idx in range(len(questions)):
                error_results.append(
                    QAResult(
                        answer="Error during generation",
                        extraction_time=extraction_time,
                        extraction_method=f"Batch Text Generation (vLLM Error, Q{q_idx+1}/{len(questions)})"
                    )
                )
            return error_results
            
        if outputs and outputs[0].outputs:
            generated_text_blob = outputs[0].outputs[0].text.strip()
        else:
            generated_text_blob = ""
            
        # Parse the answers for each question from the single text blob
        parsed_answers = self._parse_multi_question_response(generated_text_blob, len(questions))
        
        # Create QAResult objects
        results = []
        for q_idx, answer in enumerate(parsed_answers):
            results.append(
                QAResult(
                    answer=answer,
                    extraction_time=extraction_time,
                    extraction_method=f"Batch Text Generation (Q{q_idx+1}/{len(questions)})"
                )
            )
            
        return results
        
    def _parse_multi_question_response(
        self, response_text: str, num_questions: int
    ) -> List[str]:
        """
        Parse a response containing multiple answers to extract individual answers.
        (This method can likely remain the same as its logic is based on string parsing patterns)
        
        Args:
            response_text (str): The text response from the model
            num_questions (int): The number of questions that were asked
            
        Returns:
            List[str]: List of extracted answers, one per question
        """
        # Initialize answers with default value
        answers = [INFO_NOT_FOUND_PHRASE] * num_questions
        
        # Try to parse answers using patterns
        
        # Pattern 1: Look for "Question N: Answer"
        # Ensure regex handles newlines within an answer correctly.
        question_pattern = re.compile(r"Question\s+(\d+):\s*(.*?)(?=Question\s+\d+:|$)", re.DOTALL | re.IGNORECASE)
        matches = question_pattern.findall(response_text)
        
        found_by_pattern1 = False
        if matches:
            for match in matches:
                try:
                    q_num = int(match[0])
                    if 1 <= q_num <= num_questions:
                        answers[q_num-1] = match[1].strip()
                        found_by_pattern1 = True
                except (ValueError, IndexError):
                    continue # Malformed question number
        
        # If pattern 1 didn't find all answers, or found none, try a line-based approach.
        # This is a fallback or complementary strategy.
        if not found_by_pattern1 or any(ans == INFO_NOT_FOUND_PHRASE for ans in answers):
            lines = response_text.split('\n')
            current_question_index = -1 # Use index directly
            
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                # More flexible matching for "Question X:" or "X."
                # Covers "Question 1: Answer", "1. Answer", "Answer to Question 1: ..."
                q_match = re.match(r"^(?:Answer\s+to\s+)?(?:Question\s+)?(\d+)[:.]?\s*(.*)", line_stripped, re.IGNORECASE)
                
                if q_match:
                    q_num = int(q_match.group(1))
                    if 1 <= q_num <= num_questions:
                        current_question_index = q_num - 1
                        answer_on_same_line = q_match.group(2).strip()
                        if answer_on_same_line:
                            # If this question was already answered by pattern 1, append if new, else overwrite.
                            # For simplicity here, we'll overwrite if it's the default.
                            if answers[current_question_index] == INFO_NOT_FOUND_PHRASE or not found_by_pattern1:
                                answers[current_question_index] = answer_on_same_line
                            elif answer_on_same_line != INFO_NOT_FOUND_PHRASE: # Append if already has content
                                answers[current_question_index] += " " + answer_on_same_line
                        # If the line was just "Question X:", the next lines are the answer
                        elif answers[current_question_index] == INFO_NOT_FOUND_PHRASE and not found_by_pattern1 :
                             answers[current_question_index] = "" # Prepare for multi-line answer
                
                elif current_question_index != -1: # We are in the answer part of a question
                    # Append to the current question's answer, if it was initialized
                    if answers[current_question_index] == INFO_NOT_FOUND_PHRASE and not found_by_pattern1:
                        answers[current_question_index] = line_stripped # Start the answer
                    elif answers[current_question_index] != INFO_NOT_FOUND_PHRASE: # Append to existing
                         answers[current_question_index] += " " + line_stripped
            
            # Clean up any leading/trailing spaces from concatenated parts
            for i in range(num_questions):
                if isinstance(answers[i], str):
                    answers[i] = answers[i].strip()
                    if not answers[i]: # If after stripping it's empty, revert to not found
                        answers[i] = INFO_NOT_FOUND_PHRASE
        
        return answers
        
    def batch_grouped_inference(
        self, 
        questions: List[str], 
        contexts: List[str], 
        max_questions_per_group: int = 5 # This might be less critical with vLLM's efficiency
    ) -> List[QAResult]:
        """
        Group questions by context and process each group together using vLLM.
        
        Args:
            questions (List[str]): List of questions to answer
            contexts (List[str]): List of contexts, aligned with questions
            max_questions_per_group (int, optional): Maximum number of questions 
                to include in a single prompt. Defaults to 5.
                
        Returns:
            List[QAResult]: List of QA results aligned with the input questions and contexts
        """
        if len(questions) != len(contexts):
            raise ValueError("Number of questions and contexts must be the same.")
            
        if not questions:
            return []
            
        # Group questions by context
        context_to_questions_map: Dict[str, List[str]] = {}
        context_to_indices_map: Dict[str, List[int]] = {}
        
        for i, (question, context) in enumerate(zip(questions, contexts)):
            if context not in context_to_questions_map:
                context_to_questions_map[context] = []
                context_to_indices_map[context] = []
            
            context_to_questions_map[context].append(question)
            context_to_indices_map[context].append(i)
        
        all_results: List[Optional[QAResult]] = [None] * len(questions) # Preallocate results list
        
        print(f"Processing {len(context_to_questions_map)} unique contexts with grouped questions using vLLM")
        
        # Prepare a list of all prompts and corresponding original indices for batching
        # This approach aims to leverage vLLM's internal batching more effectively
        # by submitting all prompts from all groups at once, if possible, or in fewer, larger batches.
        # However, batch_questions_single_context is designed to combine multiple questions into *one* prompt.
        # So we still need to iterate per context.

        for context, grouped_questions in tqdm(context_to_questions_map.items(), desc="Processing context groups with vLLM"):
            original_indices_for_context = context_to_indices_map[context]

            # Process questions for the current context in sub-batches if necessary
            for i in range(0, len(grouped_questions), max_questions_per_group):
                current_batch_questions = grouped_questions[i : i + max_questions_per_group]
                current_batch_indices = original_indices_for_context[i : i + max_questions_per_group]
                
                # Skip empty contexts
                extraction_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                if not context or not isinstance(context, str) or context.strip() == "":
                    for original_idx in current_batch_indices:
                        all_results[original_idx] = QAResult(
                            answer="Error: Empty context",
                            extraction_time=extraction_time,
                            extraction_method="Batch Text Generation (vLLM Grouped Error)"
                        )
                    continue
                
                # If only one question in this sub-batch for the context, use answer_single_question
                if len(current_batch_questions) == 1:
                    result = self.answer_single_question(current_batch_questions[0], context)
                    # result.extraction_method += " (Grouped)" # Optionally indicate it was part of grouping
                    all_results[current_batch_indices[0]] = result
                else:
                    # Use the batch_questions_single_context for multiple questions in this sub-batch
                    batch_qa_results = self.batch_questions_single_context(current_batch_questions, context)
                    
                    # Place results in the correct positions
                    for j, original_idx in enumerate(current_batch_indices):
                        if j < len(batch_qa_results):
                            # batch_qa_results[j].extraction_method += " (Grouped)" # Optionally indicate
                            all_results[original_idx] = batch_qa_results[j]
                        else: # Should not happen if batch_questions_single_context returns one result per question
                             all_results[original_idx] = QAResult(
                                answer="Error: Mismatch in grouped results",
                                extraction_time=extraction_time,
                                extraction_method="Batch Text Generation (vLLM Grouped Error)"
                            )
        
        # Final check for any None results (should be filled)
        final_results_checked: List[QAResult] = []
        for i, res_opt in enumerate(all_results):
            if res_opt is None:
                final_results_checked.append(QAResult(
                    answer="Error: Processing failed for this item",
                    extraction_time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    extraction_method="Batch Text Generation (vLLM Grouped Error)"
                ))
            else:
                final_results_checked.append(res_opt)
                
        return final_results_checked
        
    def process_dataset_grouped(
        self, 
        dataset: Dataset, 
        max_questions_per_group: int = 5
    ) -> Dataset:
        """
        Process a HuggingFace dataset using grouped question processing with vLLM.
        
        Args:
            dataset (Dataset): Dataset containing 'question' and 'context' columns
            max_questions_per_group (int, optional): Maximum questions per group.
                Defaults to 5.
                
        Returns:
            Dataset: Dataset with added 'answer' and 'extraction_time' columns
        """
        if not all(col in dataset.column_names for col in ["question", "context"]):
            raise ValueError("Dataset must contain 'question' and 'context' columns.")
            
        questions = dataset["question"]
        contexts = dataset["context"]
        
        results = self.batch_grouped_inference( # Uses the new vLLM batch_grouped_inference
            questions=questions,
            contexts=contexts,
            max_questions_per_group=max_questions_per_group
        )
        
        # Prepare data for the new dataset
        answers = [r.answer for r in results]
        extraction_times = [r.extraction_time for r in results]
        extraction_methods = [r.extraction_method for r in results]

        num_results = len(results)
        
        # Ensure all lists have the same length for Dataset.from_dict
        if num_results < len(questions):
            print(f"Warning: Number of results ({num_results}) is less than input items ({len(questions)}). Dataset will be truncated.")
            # Truncate original data to match result length
            for key in dataset.column_names:
                dataset = dataset.select(range(num_results))
            questions = questions[:num_results] # This might not be necessary if dataset is already sliced
            contexts = contexts[:num_results] # Same as above

        result_data = {
            "question": questions[:num_results], # Ensure alignment
            "context": contexts[:num_results],   # Ensure alignment
            "answer": answers,
            "extraction_time": extraction_times,
            "extraction_method": extraction_methods,
        }
        
        # Add other original columns, ensuring they are also aligned/truncated
        for col in dataset.column_names:
            if col not in result_data: # Avoid overwriting q, c, or new columns
                result_data[col] = dataset[col][:num_results] # Ensure aligned
                            
        return Dataset.from_dict(result_data)
