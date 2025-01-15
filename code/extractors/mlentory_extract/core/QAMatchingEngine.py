import torch
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np

@dataclass
class Section:
    """Represents a section of text with its title and content"""
    title: str
    content: str
    start_idx: int
    end_idx: int

class QAMatchingEngine:
    """
    Engine for matching questions to relevant sections of text using semantic similarity.
    Uses section titles and content embeddings to find the most relevant context for each question.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the matching engine.
        
        Args:
            embedding_model (str): Name of the model to use for embeddings
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = embedding_model
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model).to(self.device)
        self.model.eval()
    
    def _extract_sections(self, text: str) -> List[Section]:
        """
        Extract sections from text based on markdown-style headers.
        
        Args:
            text (str): The text to segment
            
        Returns:
            List[Section]: List of extracted sections with titles and content
        """
        # Split text into lines
        lines = text.split('\n')
        sections = []
        current_title = ""
        current_content = []
        start_idx = 0
        
        for i, line in enumerate(lines):
            # Check for markdown headers (# Title, ## Subtitle, etc)
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            
            if header_match or i == len(lines) - 1:
                # Save previous section if exists
                if current_title and current_content:
                    sections.append(Section(
                        title=current_title,
                        content=current_title+'\n'+'\n'.join(current_content),
                        start_idx=start_idx,
                        end_idx=i
                    ))
                
                if header_match:
                    current_title = header_match.group(2)
                    current_content = []
                    start_idx = i
            else:
                current_content.append(line)
        
        # If no sections found, create one section with entire text
        if not sections:
            sections = [Section(
                title="",
                content=text,
                start_idx=0,
                end_idx=len(lines)
            )]
            
        return sections
    
    @torch.no_grad()
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            torch.Tensor: Tensor of embeddings
        """
        # Tokenize and get model outputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model(**inputs)
        
        # Use mean pooling to get text embeddings
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return embeddings
    
    def _compute_similarity(self, query_embedding: torch.Tensor, section_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between query and sections.
        
        Args:
            query_embedding (torch.Tensor): Query embedding
            section_embeddings (torch.Tensor): Section embeddings
            
        Returns:
            torch.Tensor: Similarity scores
        """
        return torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            section_embeddings
        )
    
    def find_relevant_sections(self, questions: List[str], context: str, top_k: int = 2) -> List[List[Tuple[Section, float]]]:
        """
        Find the most relevant sections for each question.
        
        Args:
            questions (List[str]): List of questions
            context (str): Text context to search in
            top_k (int): Number of top sections to return per question
            
        Returns:
            List[List[Tuple[Section, float]]]: For each question, list of (section, score) tuples
        """
        # Extract sections
        sections = self._extract_sections(context)
        if not sections:
            return [[(Section("", context, 0, len(context.split('\n'))), 1.0)] for _ in questions]
        
        # Get embeddings for sections (both title and content)
        section_texts = [f"{s.title} {s.content}" for s in sections]
        section_embeddings = self._get_embeddings(section_texts)
        
        # Get embeddings for questions
        question_embeddings = self._get_embeddings(questions)
        
        results = []
        for q_emb in question_embeddings:
            # Compute similarities
            similarities = self._compute_similarity(q_emb, section_embeddings)
            
            # Get top-k sections
            top_k_values, top_k_indices = torch.topk(similarities, min(top_k, len(sections)))
            
            # Create result list for this question
            question_results = [
                (sections[idx], score.item())
                for idx, score in zip(top_k_indices, top_k_values)
            ]
            results.append(question_results)
        
        return results
    
    def get_best_context(self, question: str, context: str) -> str:
        """
        Get the best context for a single question.
        
        Args:
            question (str): The question to find context for
            context (str): The full text context
            
        Returns:
            str: The most relevant context for the question
        """
        relevant_sections = self.find_relevant_sections([question], context, top_k=2)[0]
        
        # Combine the top sections
        combined_context = "\n".join(
            section.content for section, score in relevant_sections
        )
        
        return combined_context 