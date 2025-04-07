import os
import sys
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from pathlib import Path

# Add the parent directory to the Python path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from core.QAMatchingEngine import QAMatchingEngine
from core.MarkdownParser import MarkdownParser, Section


class TestQAMatchingEngine:
    """
    Test suite for QAMatchingEngine class.
    
    Tests the initialization, embedding generation, section finding,
    and context retrieval functionalities using real models and parser.
    """

    @pytest.fixture(scope="module")
    def engine(self):
        """
        Fixture for QAMatchingEngine with real components.
        
        Returns:
            QAMatchingEngine: Engine instance with real components
        """
        # Use a small and fast model for testing
        engine = QAMatchingEngine(embedding_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
        return engine

    @pytest.fixture
    def sample_context(self):
        """
        Fixture providing more complex markdown content for challenging tests.
        
        Returns:
            str: Sample markdown content with various sections
        """
        return """
# Introduction to AI

Artificial Intelligence (AI) is a broad field encompassing various techniques. It aims to create systems that can perform tasks typically requiring human intelligence.

## Machine Learning

Machine Learning (ML) is a subset of AI focused on algorithms that learn from data without being explicitly programmed.
It includes supervised, unsupervised, and reinforcement learning paradigms.

### Supervised Learning

Algorithms learn from labeled data (input-output pairs). Examples include linear regression for predicting values and Support Vector Machines (SVM) for classification.

### Unsupervised Learning

Algorithms find patterns in unlabeled data. Clustering (like K-Means) and dimensionality reduction (like PCA) are common techniques.

## Deep Learning

Deep Learning (DL) is a subfield of ML utilizing artificial neural networks with multiple layers (deep architectures) to model complex patterns.

### Neural Networks

Inspired by the biological brain, these networks consist of interconnected nodes or neurons organized in layers. Key components:
- Input Layer: Receives raw data.
- Hidden Layers: Perform intermediate computations. Multiple hidden layers define depth.
- Output Layer: Produces the final result (e.g., prediction, classification).

#### Activation Functions

Functions like ReLU (Rectified Linear Unit), Sigmoid, and Tanh introduce non-linearity, allowing networks to learn complex relationships. Without them, a deep network would behave like a single linear transformation.

## Natural Language Processing

The goal of NLP is to enable computers to understand, interpret, and generate human language. Common tasks include.

Common tasks include:
- Text classification
- Machine translation
- Sentiment analysis
- Named entity recognition
- Question answering
- Text generation
- Speech recognition
- Text-to-speech
- Image captioning
- Image segmentation

Techniques often involve tokenization, embeddings (like Word2Vec or BERT), and sequence models (like LSTMs or Transformers).

# Conclusion

AI, ML, DL, and NLP are interconnected and rapidly evolving fields driving innovation across various domains. Understanding their fundamentals is crucial in the modern technological landscape.
"""

    @pytest.fixture
    def sample_sections(self):
        """
        Fixture providing sample sections for testing.
        
        Returns:
            list: List of Section objects
        """
        return [
            Section("Machine Learning", "Machine learning is a subset of artificial intelligence.", 1, 3),
            Section("Neural Networks", "Neural networks are a key component in deep learning.", 5, 7),
            Section("Backpropagation", "Backpropagation is an algorithm used to train neural networks.", 9, 11)
        ]

    def test_initialization(self):
        """
        Test initialization of QAMatchingEngine with actual components.
        """
        engine = QAMatchingEngine(embedding_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
        
        assert engine.model_name == "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        assert engine.last_question_embeddings is None
        assert engine.model is not None
        assert engine.tokenizer is not None
        assert isinstance(engine.device, torch.device)
        # Check if markdown_parser is an instance of MarkdownParser
        assert isinstance(engine.markdown_parser, MarkdownParser)

    def test_compute_similarity(self, engine):
        """
        Test similarity computation between query and section embeddings.
        
        Args:
            engine: QAMatchingEngine fixture
        """
        # Generate real embeddings
        texts = ["What is machine learning?", "What are neural networks?", "How does backpropagation work?"]
        with torch.no_grad():
            embeddings = engine._get_embeddings(texts)
        
        query_embedding = embeddings[0]
        section_embeddings = embeddings[1:]
        
        # Compute similarity
        similarities = engine._compute_similarity(query_embedding, section_embeddings)
        
        # Verify output shape and values
        assert similarities.shape == torch.Size([2])
        assert all(0 <= score <= 1 for score in similarities.tolist())

    def test_get_embeddings(self, engine):
        """
        Test embedding generation with real model.
        
        Args:
            engine: QAMatchingEngine fixture
        """
        texts = ["Text 1", "Text 2"]
        
        # Generate embeddings
        embeddings = engine._get_embeddings(texts)
        
        # Verify output
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == len(texts)
        # Check embedding dimension based on the actual model used
        assert embeddings.shape[1] > 0

    def test_find_relevant_sections(self, engine, sample_context):
        """
        Test finding relevant sections for questions with real model and parser.
        
        Args:
            engine: QAMatchingEngine fixture
            sample_context: Sample markdown content
        """
        # More challenging questions requiring understanding of relationships/details
        questions = [
            "What is the relationship between AI and Machine Learning?", 
            "How do algorithms learn from labeled data?",
            "Explain the role of activation functions in deep networks.",
            "What techniques are used in NLP?",
            "Summarize the core concepts discussed."
        ]
        
        top_k = 2
        results = engine.find_relevant_sections(questions, sample_context, top_k=top_k)
            
        # Verify results structure
        assert len(results) == len(questions)
        idx = 0
        for question_results in results:
            print(f"\n \n Question results for question: {questions[idx]} \n \n")
            idx += 1
            assert len(question_results) <= top_k  # Max top_k results per question
            assert len(question_results) > 0      # Should find at least one section
            for section_result in question_results:
                print(f"\n \n Section result: {section_result[0].title} {section_result[0].content} \n \n")
                assert isinstance(section_result, tuple)
                assert len(section_result) == 2
                assert isinstance(section_result[0], Section)
                assert isinstance(section_result[1], float)
                assert 0 <= section_result[1] <= 1 # Score should be between 0 and 1

        # Verify logical relevance (spot checks)
        # Check that any of the top sections are relevant to the question
        found_relevant_section = False
        
        for section_result in results[0]:
            print(f"\n \n Section result: {section_result} \n \n")
            if "introduction to ai" in section_result[0].title.lower() or "machine learning" in section_result[0].title.lower():
                found_relevant_section = True
       
        assert found_relevant_section
        
        # Q2: Labeled data -> Expect 'Supervised Learning'
        found_relevant_section = False
        for section_result in results[1]:
            if "supervised learning" in section_result[0].title.lower():
                    found_relevant_section = True
        assert found_relevant_section

        # Q3: Activation functions -> Expect 'Activation Functions' or 'Neural Networks'
        found_relevant_section = False
        for section_result in results[2]:
            if "activation functions" in section_result[0].title.lower() or "neural networks" in section_result[0].title.lower():
                    found_relevant_section = True
        assert found_relevant_section

        # Q4: NLP Techniques -> Expect 'Natural Language Processing'
        found_relevant_section = False
        for section_result in results[3]:
            if "natural language processing" in section_result[0].title.lower():
                    found_relevant_section = True
        assert found_relevant_section

    def test_get_best_context(self, engine, sample_context):
        """
        Test getting best context for a question with real model and parser.
        
        Args:
            engine: QAMatchingEngine fixture
            sample_context: Sample markdown content (now more complex)
        """
        # A more nuanced question
        question = "How are patterns found in data without labels?"
        
        # Get the best context using the real parser
        best_context_sections = engine.find_relevant_sections([question], sample_context, top_k=3)[0]
        best_context = "\n".join([section.content for section, score in best_context_sections])
        print(f"\n \n Best context: {best_context} \n \n")
        
        # Verify result
        assert isinstance(best_context, str)
        assert len(best_context) > 0
        # The content should relate to unsupervised learning
        assert "unsupervised learning" in best_context.lower() or "unlabeled data" in best_context.lower()
        assert "machine translation" not in best_context.lower()
        assert " supervised " not in best_context.lower()

    def test_batch_processing(self, engine):
        """
        Test processing multiple texts in batches.
        
        Args:
            engine: QAMatchingEngine fixture
        """
        # Create a larger batch of texts to test batch processing
        texts = [f"Sample text {i}" for i in range(10)]
        
        # Generate embeddings
        embeddings = engine._get_embeddings(texts)
        
        # Verify output
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0

    def test_empty_sections_handling(self, engine):
        """
        Test handling of context that yields minimal sections from the real parser.
        
        Args:
            engine: QAMatchingEngine fixture
        """
        questions = ["What is this about?"]
        # Context that might not have standard markdown headers
        context_without_headers = "This is a simple paragraph. It does not contain any markdown headers. Just plain text content."
        
        # Use the real markdown parser
        results = engine.find_relevant_sections(questions, context_without_headers)
        
        print(f"\n \n LEN of engine.last_question_embeddings: {len(engine.last_question_embeddings)} \n \n")
        
        print(f"\n \n Results: {results} \n \n")
            
        # Check assumptions based on MarkdownParser behavior:
        # If no headers, it might return the whole text as one section.
        assert len(results) == 1 # Results for the one question
        assert len(results[0]) == 1 # Should find one matching section (the default one)
        
        # Verify the section content and score
        found_section = results[0][0][0]
        found_score = results[0][0][1]
        
        # The title might be None or default, content should be the whole text
        assert found_section.title is None or found_section.title == "" # Adjust if MarkdownParser has different default
        assert found_section.content.strip() == context_without_headers.strip()
