import torch
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasets import Dataset
import re

torch.backends.cuda.matmul.allow_tf32 = True
from transformers import pipeline


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
            for batch_start in tqdm(range(0, len(prompts), self.batch_size), total=len(prompts), desc="Processing QA batches"):
                batch_prompts = prompts[batch_start:batch_start + self.batch_size]
                pipeline_outputs.extend(self.pipeline(batch_prompts, batch_size=self.batch_size))
        except Exception as e:
             print(f"Error during pipeline execution, retuning empty list: {e}")
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

    @torch.inference_mode()
    def batch_questions_single_context(
        self, questions: List[str], context: str
    ) -> List[QAResult]:
        """
        Process multiple questions with a single shared context in one prompt.
        
        Args:
            questions (List[str]): List of questions to answer
            context (str): The shared context for all questions
            
        Returns:
            List[QAResult]: List of QA results, one for each question
        """
        if not questions:
            return []
            
        if not context or not isinstance(context, str) or context.strip() == "":
            raise ValueError("Context must be a non-empty string")
            
        # Format the questions with numbers
        formatted_questions = "\n".join([f"Question {i+1}: {q}" for i, q in enumerate(questions)])
        
        # Create the prompt with all questions
        prompt = MULTI_QUESTION_TEMPLATE.format(
            context=context,
            questions=formatted_questions
        )
        
        print(f"\n\nPrompt: {prompt}\n\n")
        
        # Generate response
        try:
            print(f"Generating response for prompt:")
            output_list = self.pipeline(prompt)
        except Exception as e:
            print(f"Error during multi-question inference: {e}")
            # Return error results for all questions
            error_results = []
            extraction_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for _ in questions:
                error_results.append(
                    QAResult(
                        answer="Error during generation",
                        extraction_time=extraction_time
                    )
                )
            return error_results
            
        # Parse the output
        if isinstance(output_list, list):
            output = output_list[0] if output_list else {"generated_text": ""}
        elif isinstance(output_list, dict):
            output = output_list
        else:
            print(f"Warning: Unexpected output format for multi-question inference: {type(output_list)}")
            output = {"generated_text": ""}
            
        generated_text = output.get("generated_text", "")
        
        # Extract just the generated part (after the prompt)
        if generated_text.startswith(prompt):
            generated_answers = generated_text[len(prompt):].strip()
        else:
            # If the model doesn't return the prompt, use the full text
            generated_answers = generated_text.strip()
            
        # Parse the answers for each question
        answers = self._parse_multi_question_response(generated_answers, len(questions))
        
        # Create QAResult objects
        results = []
        extraction_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        for q_idx, answer in enumerate(answers):
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
        question_pattern = re.compile(r"Question\s+(\d+):\s*(.*?)(?=Question\s+\d+:|$)", re.DOTALL)
        matches = question_pattern.findall(response_text)
        
        if matches:
            for match in matches:
                try:
                    q_num = int(match[0])
                    if 1 <= q_num <= num_questions:
                        answers[q_num-1] = match[1].strip()
                except (ValueError, IndexError):
                    continue
                    
        # If no matches found or some questions still have default answers,
        # try an alternative approach: split by lines and look for patterns
        if all(answer == INFO_NOT_FOUND_PHRASE for answer in answers):
            lines = response_text.split('\n')
            current_question = None
            current_answer = []
            
            for line in lines:
                # Check if line contains a question number
                q_match = re.match(r"^(?:Answer\s+to\s+)?(?:Question\s+)?(\d+)[:.]\s*(.*)", line)
                if q_match:
                    # If we were building an answer, save it
                    if current_question is not None and current_answer:
                        answers[current_question-1] = ' '.join(current_answer).strip()
                        current_answer = []
                    
                    # Start new question
                    q_num = int(q_match.group(1))
                    if 1 <= q_num <= num_questions:
                        current_question = q_num
                        # If there's answer content on the same line
                        if q_match.group(2).strip():
                            current_answer.append(q_match.group(2).strip())
                else:
                    # Add to current answer if we're inside a question
                    if current_question is not None and line.strip():
                        current_answer.append(line.strip())
            
            # Save the last answer if any
            if current_question is not None and current_answer:
                answers[current_question-1] = ' '.join(current_answer).strip()
        
        return answers
        
    def batch_grouped_inference(
        self, 
        questions: List[str], 
        contexts: List[str], 
        max_questions_per_group: int = 5
    ) -> List[QAResult]:
        """
        Group questions by context and process each group together.
        
        This method identifies questions that share the same context and processes them
        together to reduce the number of prompts sent to the model.
        
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
        context_to_questions = {}
        context_to_indices = {}
        
        for i, (question, context) in enumerate(zip(questions, contexts)):
            if context not in context_to_questions:
                context_to_questions[context] = []
                context_to_indices[context] = []
            
            context_to_questions[context].append(question)
            context_to_indices[context].append(i)
        
        # Process each context group
        all_results = [None] * len(questions)  # Preallocate results list
        
        print(f"Processing {len(context_to_questions)} unique contexts with grouped questions")
        for context, grouped_questions in tqdm(context_to_questions.items(), desc="Processing context groups"):
            # Process questions in groups of max_questions_per_group
            for i in range(0, len(grouped_questions), max_questions_per_group):
                batch_questions = grouped_questions[i:i + max_questions_per_group]
                batch_indices = context_to_indices[context][i:i + max_questions_per_group]
                
                # Skip empty contexts
                if not context or not isinstance(context, str) or context.strip() == "":
                    for idx in batch_indices:
                        all_results[idx] = QAResult(
                            answer="Error: Empty context",
                            extraction_time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        )
                    continue
                
                # If only one question, use the single question method
                if len(batch_questions) == 1:
                    result = self.answer_single_question(batch_questions[0], context)
                    all_results[batch_indices[0]] = result
                else:
                    # Use the batch method for multiple questions
                    batch_results = self.batch_questions_single_context(batch_questions, context)
                    
                    # Place results in the correct positions
                    for j, idx in enumerate(batch_indices):
                        if j < len(batch_results):  # Safety check
                            all_results[idx] = batch_results[j]
        
        # Check if any results are None (shouldn't happen, but just in case)
        for i, result in enumerate(all_results):
            if result is None:
                all_results[i] = QAResult(
                    answer="Error: Processing failed",
                    extraction_time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                )
                
        return all_results
        
    def process_dataset_grouped(
        self, 
        dataset: Dataset, 
        max_questions_per_group: int = 5
    ) -> Dataset:
        """
        Process a HuggingFace dataset using grouped question processing.
        
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
        
        results = self.batch_grouped_inference(
            questions=questions,
            contexts=contexts,
            max_questions_per_group=max_questions_per_group
        )
        
        # Prepare data for the new dataset
        result_data = {
            "answer": [r.answer for r in results],
            "extraction_time": [r.extraction_time for r in results],
            "extraction_method": [r.extraction_method for r in results],
        }
        
        # Keep original columns
        for col in dataset.column_names:
            if col not in result_data:
                result_data[col] = dataset[col][:len(results)]
                
        if len(results) < len(questions):
            print(f"Warning: Number of results ({len(results)}) is less than input items ({len(questions)}). Dataset truncated.")
            
        return Dataset.from_dict(result_data)
