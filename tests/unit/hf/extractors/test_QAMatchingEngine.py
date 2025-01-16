import pytest
import torch
from mlentory_extract.core.QAMatchingEngine import QAMatchingEngine, Section

@pytest.fixture
def qa_engine():
    # We use an optimized model for running tests in the cpu
    return QAMatchingEngine(embedding_model="sentence-transformers/all-mpnet-base-v2")

@pytest.fixture
def sample_markdown_text():
    return """# Introduction

                This is an introduction paragraph about machine learning.

                ## Model Architecture
                The model uses a transformer-based architecture.
                It has multiple attention layers.

                ## Training Details
                The model was trained for 10 epochs.
                Learning rate was set to 0.001.

                ### Hardware Requirements
                Requires at least 8GB GPU memory.
"""

@pytest.fixture
def sample_questions():
    return [
        "What is the model architecture?",
        "What are the training details?",
        "What hardware is needed?"
    ]

def test_extract_sections(qa_engine, sample_markdown_text):
    sections = qa_engine._extract_sections(sample_markdown_text)
    
    assert len(sections) == 4  # Should find 4 sections
    assert sections[0].title == "Introduction"
    assert "introduction paragraph" in sections[0].content
    assert sections[1].title == "Model Architecture"
    assert "transformer-based" in sections[1].content
    assert sections[2].title == "Training Details"
    assert "10 epochs" in sections[2].content

def test_get_embeddings(qa_engine):
    texts = ["This is a test sentence.", "Another test sentence."]
    embeddings = qa_engine._get_embeddings(texts)
    
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == 2  # Two embeddings
    assert embeddings.shape[1] > 0  # Non-empty embedding dimension

def test_find_relevant_sections(qa_engine, sample_markdown_text, sample_questions):
    results = qa_engine.find_relevant_sections(sample_questions, sample_markdown_text, top_k=2)
    
    assert len(results) == len(sample_questions)
    for question_results in results:
        assert len(question_results) <= 2  # Should return at most top_k=2 results
        for section, score in question_results:
            assert isinstance(section, Section)
            assert isinstance(score, float)
            assert 0 <= score <= 1  # Similarity score should be between 0 and 1

def test_get_best_context(qa_engine, sample_markdown_text):
    question = "What are the hardware requirements?"
    context = qa_engine.get_best_context(question, sample_markdown_text)
    
    assert isinstance(context, str)
    assert "GPU memory" in context  # Should find relevant context

def test_empty_context(qa_engine):
    empty_text = ""
    question = "What is this about?"
    results = qa_engine.find_relevant_sections([question], empty_text)
    
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0][0].content == ""

def test_no_sections_found(qa_engine):
    text_without_headers = "This is just a plain text without any markdown headers."
    question = "What is this about?"
    results = qa_engine.find_relevant_sections([question], text_without_headers)
    
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0][0].content == text_without_headers

def test_compute_similarity(qa_engine):
    # Create sample embeddings
    query_embedding = torch.randn(768)  # Assuming 768-dim embeddings
    section_embeddings = torch.randn(3, 768)  # 3 sections
    
    similarities = qa_engine._compute_similarity(query_embedding, section_embeddings)
    
    assert isinstance(similarities, torch.Tensor)
    assert similarities.shape[0] == 3  # Should have similarity score for each section
    assert all(-1 <= score <= 1 for score in similarities)  # Cosine similarity range 