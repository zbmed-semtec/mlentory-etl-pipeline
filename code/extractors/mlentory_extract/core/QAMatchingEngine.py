import os
import torch
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .MarkdownParser import MarkdownParser, Section


@dataclass
class RelevantSectionMatch:
    """Represents a section of text relevant to a query, along with a similarity score."""
    section: Section
    score: float


@dataclass
class GroupedRelevantSectionMatch:
    """Represents a group of question indices and their shared relevant sections."""
    question_indices: List[int]
    relevant_sections: List[RelevantSectionMatch]


class QAMatchingEngine:
    """
    Engine for matching questions to relevant sections of text using semantic similarity.
    Uses section titles and content embeddings to find the most relevant context for each question.
    """

    def __init__(
        self, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        """
        Initialize the matching engine.

        Args:
            embedding_model (str): Name of the model to use for embeddings
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = embedding_model

        # Enable TF32 for better performance on Ampere GPUs
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # print(os.getenv("HF_TOKEN"))
        # Load model in half precision for CUDA, regular precision for CPU
        self.model = AutoModel.from_pretrained(
            embedding_model,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")
        ).to(self.device)

        # Load tokenizer with caching
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model, use_fast=True)

        # Set model to eval mode
        self.model.eval()
        
        # Initialize markdown parser
        self.markdown_parser = MarkdownParser()

        # Cache for question embeddings
        self.last_questions = []
        self.last_question_embeddings: Optional[torch.Tensor] = None
        
    @torch.no_grad()
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings for a list of texts.

        Args:
            texts (List[str]): List of texts to embed

        Returns:
            torch.Tensor: Tensor of embeddings
        """
        # Process in smaller batches to avoid OOM
        batch_size = 8  # Reduced batch size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                # Tokenize with efficient padding
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                ).to(self.device)

                # Get model outputs efficiently
                outputs = self.model(**inputs)

                # Efficient mean pooling
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state

                # Use broadcasting for faster computation
                mask_expanded = attention_mask.unsqueeze(-1)
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

                # Move to CPU to free GPU memory
                all_embeddings.append(embeddings.cpu())

            except RuntimeError as e:
                print(f"Warning: Error processing batch {i}-{i+batch_size}: {str(e)}")
                # Try processing one by one if batch fails
                for text in batch_texts:
                    try:
                        inputs = self.tokenizer(
                            [text],
                            padding=True,
                            truncation=True,
                            max_length=2048,
                            return_tensors="pt",
                        ).to(self.device)

                        outputs = self.model(**inputs)
                        attention_mask = inputs["attention_mask"]
                        token_embeddings = outputs.last_hidden_state
                        mask_expanded = attention_mask.unsqueeze(-1)
                        sum_embeddings = torch.sum(
                            token_embeddings * mask_expanded, dim=1
                        )
                        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                        embeddings = sum_embeddings / sum_mask
                        all_embeddings.append(embeddings.cpu())
                    except Exception as e2:
                        print(f"Warning: Could not process text: {str(e2)}")
                        # Add zero embedding as fallback
                        all_embeddings.append(
                            torch.zeros((1, outputs.last_hidden_state.size(-1)))
                        )

            # Clear GPU cache after each batch
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Concatenate all embeddings
        return torch.cat(all_embeddings, dim=0).to(self.device)

    def _compute_similarity(
        self, query_embedding: torch.Tensor, section_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between query and sections.

        Args:
            query_embedding (torch.Tensor): Query embedding
            section_embeddings (torch.Tensor): Section embeddings

        Returns:
            torch.Tensor: Similarity scores
        """
        # Normalize embeddings for more efficient cosine similarity
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)
        section_embeddings = torch.nn.functional.normalize(
            section_embeddings, p=2, dim=-1
        )

        return torch.mm(query_embedding.unsqueeze(0), section_embeddings.t()).squeeze(0)

    def find_relevant_sections(
        self, questions: List[str], context: str, top_k: int = 2, max_section_length: int = 2000
    ) -> List[List[RelevantSectionMatch]]:
        """
        Find the most relevant sections for each question.

        Args:
            questions (List[str]): List of questions
            context (str): Text context to search in
            top_k (int): Number of top sections to return per question
            max_section_length (int): Maximum length for fine-grained sections

        Returns:
            List[List[RelevantSectionMatch]]: For each question, list of RelevantSectionMatch objects
        """
        
        # Extract sections using the markdown parser
        sections = self.markdown_parser.extract_hierarchical_sections(
            context, max_section_length
        )
        
        print(f"\n \n Printing all sections!!!!!!!! \n \n", len(sections))
        for s in sections:
            print(f"Section:{s.content} \n")
        
        # print(f"\n \n Sections: {sections} \n \n")
        if not sections:
            return [
                [RelevantSectionMatch(section=Section("", context, 0, len(context.split("\n"))), score=1.0)]
                for _ in questions
            ]

        # Get embeddings for sections
        section_texts = [f"{s.content}" for s in sections]
        section_embeddings = self._get_embeddings(section_texts)

        # Get embeddings for questions, using cache if possible
        if self.last_questions == questions and self.last_question_embeddings is not None:
            question_embeddings = self.last_question_embeddings
            # Ensure embeddings are on the correct device (might have been moved to CPU)
            question_embeddings = question_embeddings.to(self.device)
        else:
            question_embeddings = self._get_embeddings(questions)
            self.last_question_embeddings = question_embeddings
            self.last_questions = questions
        results = []
        for q_emb in question_embeddings:
            # Compute similarities
            similarities = self._compute_similarity(q_emb, section_embeddings)

            # Get top-k sections
            top_k_values, top_k_indices = torch.topk(
                similarities, min(top_k, len(sections))
            )

            # Create result list for this question
            question_results = [
                RelevantSectionMatch(section=sections[idx], score=score.item())
                for idx, score in zip(top_k_indices, top_k_values)
            ]
            results.append(question_results)

        return results
    
    def find_grouped_relevant_sections(
        self, questions: List[str], context: str, top_k: int = 2, max_section_length: int = 2000, max_questions_per_group: int = 5
    ) -> List[GroupedRelevantSectionMatch]:
        """
        Find the most relevant sections for a group of similar questions.

        Args:
            questions (List[str]): List of questions
            context (str): Text context to search in
            top_k (int): Number of top sections to return per question
            max_section_length (int): Maximum length for fine-grained sections
            max_questions_per_group (int): Maximum number of questions to include in a group

        Returns:
            List[GroupedRelevantSectionMatch]: A list where each item contains 
            question_indices and their relevant_sections (list of RelevantSectionMatch objects)
        """
        
        # Extract sections using the markdown parser
        sections = self.markdown_parser.extract_chunk_sections(
            context, max_section_length, max_list_table_lines=5
        )
        
        if not sections:
            return [
                GroupedRelevantSectionMatch(
                    question_indices=[i],
                    relevant_sections=[RelevantSectionMatch(section=Section("", context, 0, len(context.split("\n"))), score=1.0)]
                )
                for i in range(len(questions))
            ]

        # Get embeddings for sections
        section_texts = [f"{s.title} \n {s.content}" for s in sections]
        section_embeddings = self._get_embeddings(section_texts)

        # Get embeddings for questions, using cache if possible
        if self.last_questions == questions and self.last_question_embeddings is not None:
            question_embeddings = self.last_question_embeddings
            # Ensure embeddings are on the correct device (might have been moved to CPU)
            question_embeddings = question_embeddings.to(self.device)
        else:
            question_embeddings = self._get_embeddings(questions)
            self.last_question_embeddings = question_embeddings
            self.last_questions = questions
            
        # Group questions by similarity
        question_groups = self._group_questions_by_similarity(question_embeddings, n_clusters=question_embeddings.shape[0]//max_questions_per_group)
        
        results = []
        for q_indices in question_groups:
            # If the group is larger than max_questions_per_group, split it into smaller groups
            for i in range(0, len(q_indices), max_questions_per_group):
                sub_group = q_indices[i:i + max_questions_per_group]
                
                # Use the first question embedding as representative for the group
                # Could be improved by using average embedding or other strategies
                representative_embedding = question_embeddings[sub_group[0]]
                
                # Compute similarities
                similarities = self._compute_similarity(representative_embedding, section_embeddings)

                # Get top-k sections
                top_k_values, top_k_indices = torch.topk(
                    similarities, min(top_k, len(sections))
                )

                # Create result list for this question group
                question_results = [
                    RelevantSectionMatch(section=sections[idx], score=score.item())
                    for idx, score in zip(top_k_indices, top_k_values)
                ]
                results.append(GroupedRelevantSectionMatch(question_indices=sub_group, relevant_sections=question_results))

        return results

    def _group_questions_by_similarity(
        self, question_embeddings: torch.Tensor, n_clusters: int = 10
    ) -> List[List[int]]:
        """
        Group questions by similarity using agglomerative clustering.

        Args:
            question_embeddings (torch.Tensor): Embeddings for the questions (shape: [N, D])
            n_clusters (int): Number of clusters to create

        Returns:
            List[List[int]]: List of groups of question indices

        Raises:
            ValueError: If question_embeddings is not 2D

        Example:
            >>> groups = self._group_questions_by_similarity(embeddings, distance_threshold=0.3)
            >>> for group in groups:
            ...     print([questions[i] for i in group])
        """
        if question_embeddings.ndim != 2:
            raise ValueError("question_embeddings must be a 2D tensor")

        # Normalize embeddings for cosine similarity
        embeddings = torch.nn.functional.normalize(question_embeddings, p=2, dim=-1).cpu().numpy()

        # Compute cosine distance matrix (1 - cosine similarity)
        cosine_distances = 1 - np.dot(embeddings, embeddings.T)
        np.fill_diagonal(cosine_distances, 0.0)

        # Perform clustering
        clustering = AgglomerativeClustering(
            # affinity="precomputed",
            linkage="average",
            n_clusters=n_clusters
        )
        labels = clustering.fit_predict(cosine_distances)

        # Group indices by cluster label
        groups = []
        for label in np.unique(labels):
            group = np.where(labels == label)[0].tolist()
            groups.append(group)

        return groups
        