import numpy as np
np.float_ = np.float64
import pytest
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, XSD, FOAF, RDFS, SKOS
from unittest.mock import MagicMock

# Assuming GraphHandlerForKG is in the correct path relative to tests
# Adjust the import path if necessary
from mlentory_load.core.GraphHandlerForKG import GraphHandlerForKG, _NAME_PREDICATES_N3, _PREDICATES_TO_RESOLVE_N3
from mlentory_transform.utils.enums import SchemasURL # Import the enum

# Define namespaces used in sample data
NS2 = Namespace("https://mlentory.zbmed.de/mlentory_graph/ns2#") # Example namespace for ns2
NS1 = Namespace("https://mlentory.zbmed.de/mlentory_graph/ns1#") # Example namespace for ns1
SCHEMA = Namespace(SchemasURL.SCHEMA.value) # Define SCHEMA namespace using enum


@pytest.fixture
def sample_entities_lookup():
    """Provides a pre-populated entities_in_kg dictionary for testing."""
    entities = {
        # Keys should be N3 URIs
        '<https://mlentory.zbmed.de/mlentory_graph/model1>': {
            RDF.type.n3(): [NS2.ML_Model.n3()],
            SCHEMA.name.n3(): ['"Test Model One"^^<https://www.w3.org/2001/XMLSchema#string>'],
            SCHEMA.keywords.n3(): [
                # Values should be N3 URIs or literals
                '<https://mlentory.zbmed.de/mlentory_graph/keyword1>',
                '<https://mlentory.zbmed.de/mlentory_graph/keyword_no_name>',
                '"literal_keyword"^^<https://www.w3.org/2001/XMLSchema#string>'
            ],
            SCHEMA.mlTask.n3(): [
                '<https://mlentory.zbmed.de/mlentory_graph/task1>',
                '<https://mlentory.zbmed.de/mlentory_graph/task_no_name>',
            ],
            SCHEMA.fineTunedFrom.n3(): [
                '<https://mlentory.zbmed.de/mlentory_graph/base_model1>',
                '"Unknown Base Model"^^<https://www.w3.org/2001/XMLSchema#string>',
                '<https://mlentory.zbmed.de/mlentory_graph/base_model_no_name>'
            ],
            SCHEMA.trainedOn.n3(): [
                '<https://mlentory.zbmed.de/mlentory_graph/dataset1>',
                '<https://mlentory.zbmed.de/mlentory_graph/dataset_no_name>',
            ],
            SCHEMA.testedOn.n3(): [
                 '<https://mlentory.zbmed.de/mlentory_graph/dataset1>',
                 '<https://mlentory.zbmed.de/mlentory_graph/dataset2>',
                 '<https://mlentory.zbmed.de/mlentory_graph/dataset_not_in_lookup>'
            ],
        },
        # --- Resolvable Entities ---
        '<https://mlentory.zbmed.de/mlentory_graph/dataset1>': {
            RDF.type.n3(): [SCHEMA.Dataset.n3()],
            SCHEMA.name.n3(): ['"Awesome Dataset One"^^<https://www.w3.org/2001/XMLSchema#string>'],
        },
         '<https://mlentory.zbmed.de/mlentory_graph/dataset2>': {
            RDF.type.n3(): [SCHEMA.Dataset.n3()],
            SCHEMA.name.n3(): ['"Spectacular Dataset Two"^^<https://www.w3.org/2001/XMLSchema#string>'],
        },
         '<https://mlentory.zbmed.de/mlentory_graph/keyword1>': {
             RDF.type.n3(): [SCHEMA.DefinedTerm.n3()],
             SCHEMA.name.n3(): ['"Cool Keyword"^^<https://www.w3.org/2001/XMLSchema#string>']
         },
         '<https://mlentory.zbmed.de/mlentory_graph/task1>': {
             RDF.type.n3(): [SCHEMA.DefinedTerm.n3()],
             SCHEMA.name.n3(): ['"Text Generation"^^<https://www.w3.org/2001/XMLSchema#string>']
         },
         '<https://mlentory.zbmed.de/mlentory_graph/base_model1>': {
             RDF.type.n3(): [NS2.ML_Model.n3()],
             SCHEMA.name.n3(): ['"Solid Foundation Model"^^<https://www.w3.org/2001/XMLSchema#string>']
         },
         # --- Entities Existing but Without Names in this Sample ---
         '<https://mlentory.zbmed.de/mlentory_graph/keyword_no_name>': {
            RDF.type.n3(): [SCHEMA.DefinedTerm.n3()]
         },
         '<https://mlentory.zbmed.de/mlentory_graph/task_no_name>': {
            RDF.type.n3(): [SCHEMA.DefinedTerm.n3()]
         },
        '<https://mlentory.zbmed.de/mlentory_graph/base_model_no_name>': {
            RDF.type.n3(): [NS2.ML_Model.n3()]
         },
        '<https://mlentory.zbmed.de/mlentory_graph/dataset_no_name>': {
             RDF.type.n3(): [SCHEMA.Dataset.n3()]
         },
    }
    return entities


@pytest.fixture
def graph_handler_instance():
    """Provides a GraphHandlerForKG instance with mocked dependencies."""
    mock_sql = MagicMock()
    mock_rdf = MagicMock()
    mock_index = MagicMock()
    # Pass mock objects instead of real handlers
    handler = GraphHandlerForKG(
        SQLHandler=mock_sql,
        RDFHandler=mock_rdf,
        IndexHandler=mock_index,
        kg_files_directory="./dummy_kg_files",
    )
    return handler


# --- Tests for _resolve_identifier ---

def test_resolve_identifier_success(graph_handler_instance, sample_entities_lookup):
    """Test resolving a known identifier with a name."""
    # Pass plain URI to the function
    identifier_plain = 'https://mlentory.zbmed.de/mlentory_graph/dataset1'
    # Expect resolved name (literal)
    expected_name = '"Awesome Dataset One"^^<https://www.w3.org/2001/XMLSchema#string>'
    resolved_name = graph_handler_instance._resolve_identifier(identifier_plain, sample_entities_lookup)
    assert resolved_name == expected_name

def test_resolve_identifier_defined_term(graph_handler_instance, sample_entities_lookup):
    """Test resolving an identifier that represents a defined term (keyword)."""
    # Pass plain URI
    identifier_plain = 'https://mlentory.zbmed.de/mlentory_graph/keyword1'
    # Expect resolved name
    expected_name = '"Cool Keyword"^^<https://www.w3.org/2001/XMLSchema#string>'
    resolved_name = graph_handler_instance._resolve_identifier(identifier_plain, sample_entities_lookup)
    assert resolved_name == expected_name

def test_resolve_identifier_no_name(graph_handler_instance, sample_entities_lookup):
    """Test resolving an identifier that exists but has no matching name predicate."""
    # Pass plain URI
    identifier_plain = 'https://mlentory.zbmed.de/mlentory_graph/keyword_no_name'
    # Expect plain URI back (function adds <> for lookup, fails, returns stripped URI)
    expected_output = identifier_plain
    resolved_output = graph_handler_instance._resolve_identifier(identifier_plain, sample_entities_lookup)
    assert resolved_output == expected_output

def test_resolve_identifier_not_found(graph_handler_instance, sample_entities_lookup):
    """Test resolving an identifier that does not exist in the lookup."""
    # Pass plain URI
    identifier_plain = 'https://example.com/non_existent_uri'
    # Expect plain URI back
    expected_output = identifier_plain
    resolved_output = graph_handler_instance._resolve_identifier(identifier_plain, sample_entities_lookup)
    assert resolved_output == expected_output

def test_resolve_identifier_literal(graph_handler_instance, sample_entities_lookup):
    """Test resolving something that is not a URI (a literal string)."""
    # Pass literal
    identifier_literal = '"literal_keyword"^^<https://www.w3.org/2001/XMLSchema#string>'
    # Expect literal back (adding <> makes it invalid for lookup, returns stripped original)
    # Note: Stripping <> from a literal does nothing.
    expected_output = identifier_literal
    resolved_output = graph_handler_instance._resolve_identifier(identifier_literal, sample_entities_lookup)
    assert resolved_output == expected_output

# --- Tests for _resolve_identifier_list ---

def test_resolve_identifier_list_mixed(graph_handler_instance, sample_entities_lookup):
    """Test resolving a list with a mix of resolvable, unresolvable, and literal identifiers."""
    # Pass list of plain URIs and literals
    identifiers_list = [
        'https://mlentory.zbmed.de/mlentory_graph/dataset1',
        'https://mlentory.zbmed.de/mlentory_graph/keyword_no_name',
        'https://example.com/non_existent_uri',
        '"literal_keyword"^^<https://www.w3.org/2001/XMLSchema#string>',
        'https://mlentory.zbmed.de/mlentory_graph/keyword1',
        'https://mlentory.zbmed.de/mlentory_graph/dataset_not_in_lookup'
    ]
    # Expect resolved names (literals) or plain URIs/literals
    expected_list = [
        '"Awesome Dataset One"^^<https://www.w3.org/2001/XMLSchema#string>',
        'https://mlentory.zbmed.de/mlentory_graph/keyword_no_name',
        'https://example.com/non_existent_uri',
        '"literal_keyword"^^<https://www.w3.org/2001/XMLSchema#string>',
        '"Cool Keyword"^^<https://www.w3.org/2001/XMLSchema#string>',
        'https://mlentory.zbmed.de/mlentory_graph/dataset_not_in_lookup'
    ]
    resolved_list = graph_handler_instance._resolve_identifier_list(identifiers_list, sample_entities_lookup)
    assert resolved_list == expected_list

def test_resolve_identifier_list_empty(graph_handler_instance, sample_entities_lookup):
    """Test resolving an empty list."""
    # Pass empty list
    identifiers_list = []
    # Expect empty list
    expected_list = []
    resolved_list = graph_handler_instance._resolve_identifier_list(identifiers_list, sample_entities_lookup)
    assert resolved_list == expected_list

def test_resolve_identifier_list_all_resolvable(graph_handler_instance, sample_entities_lookup):
    """Test resolving a list where all identifiers have names."""
    # Pass list of plain URIs
    identifiers_list = [
        'https://mlentory.zbmed.de/mlentory_graph/dataset1',
        'https://mlentory.zbmed.de/mlentory_graph/base_model1',
    ]
    # Expect list of resolved names (literals)
    expected_list = [
        '"Awesome Dataset One"^^<https://www.w3.org/2001/XMLSchema#string>',
        '"Solid Foundation Model"^^<https://www.w3.org/2001/XMLSchema#string>',
    ]
    resolved_list = graph_handler_instance._resolve_identifier_list(identifiers_list, sample_entities_lookup)
    assert resolved_list == expected_list

def test_resolve_identifier_list_none_resolvable(graph_handler_instance, sample_entities_lookup):
    """Test resolving a list where no identifiers can be resolved to names."""
    # Pass list of plain URIs and literals
    identifiers_list = [
        'https://mlentory.zbmed.de/mlentory_graph/keyword_no_name',
        'https://example.com/non_existent_uri',
        '"literal_keyword"^^<https://www.w3.org/2001/XMLSchema#string>',
    ]
    # Expect list of plain URIs and literals
    expected_list = [
        'https://mlentory.zbmed.de/mlentory_graph/keyword_no_name',
        'https://example.com/non_existent_uri',
        '"literal_keyword"^^<https://www.w3.org/2001/XMLSchema#string>',
    ]
    resolved_list = graph_handler_instance._resolve_identifier_list(identifiers_list, sample_entities_lookup)
    assert resolved_list == expected_list

def test_resolve_identifier_list_specific_predicates(graph_handler_instance, sample_entities_lookup):
    """Test resolving lists associated with the specific predicates added."""
    # Use N3 URI key for lookup in fixture
    model_data = sample_entities_lookup['<https://mlentory.zbmed.de/mlentory_graph/model1>']

    # Test keywords (list contains N3 URIs/literals from fixture)
    keywords_list_from_fixture = model_data[SCHEMA.keywords.n3()]
    # Pass plain URIs/literals to resolver
    keywords_input = [
        'https://mlentory.zbmed.de/mlentory_graph/keyword1',
        'https://mlentory.zbmed.de/mlentory_graph/keyword_no_name',
        '"literal_keyword"^^<https://www.w3.org/2001/XMLSchema#string>'
    ]
    # Expect resolved names (literals) or plain URIs/literals
    expected_keywords = [
        '"Cool Keyword"^^<https://www.w3.org/2001/XMLSchema#string>',
        'https://mlentory.zbmed.de/mlentory_graph/keyword_no_name',
        '"literal_keyword"^^<https://www.w3.org/2001/XMLSchema#string>'
    ]
    resolved_keywords = graph_handler_instance._resolve_identifier_list(keywords_input, sample_entities_lookup)
    assert resolved_keywords == expected_keywords

    # Test mlTask
    mltask_list_from_fixture = model_data[SCHEMA.mlTask.n3()]
    mltask_input = [
        'https://mlentory.zbmed.de/mlentory_graph/task1',
        'https://mlentory.zbmed.de/mlentory_graph/task_no_name'
    ]
    expected_mltask = [
        '"Text Generation"^^<https://www.w3.org/2001/XMLSchema#string>',
        'https://mlentory.zbmed.de/mlentory_graph/task_no_name'
    ]
    resolved_mltask = graph_handler_instance._resolve_identifier_list(mltask_input, sample_entities_lookup)
    assert resolved_mltask == expected_mltask

    # Test fineTunedFrom
    finetuned_list_from_fixture = model_data[SCHEMA.fineTunedFrom.n3()]
    finetuned_input = [
        'https://mlentory.zbmed.de/mlentory_graph/base_model1',
        '"Unknown Base Model"^^<https://www.w3.org/2001/XMLSchema#string>',
        'https://mlentory.zbmed.de/mlentory_graph/base_model_no_name'
    ]
    expected_finetuned = [
        '"Solid Foundation Model"^^<https://www.w3.org/2001/XMLSchema#string>',
        '"Unknown Base Model"^^<https://www.w3.org/2001/XMLSchema#string>',
        'https://mlentory.zbmed.de/mlentory_graph/base_model_no_name'
    ]
    resolved_finetuned = graph_handler_instance._resolve_identifier_list(finetuned_input, sample_entities_lookup)
    assert resolved_finetuned == expected_finetuned

    # Test trainedOn
    trainedon_list_from_fixture = model_data[SCHEMA.trainedOn.n3()]
    trainedon_input = [
        'https://mlentory.zbmed.de/mlentory_graph/dataset1',
        'https://mlentory.zbmed.de/mlentory_graph/dataset_no_name'
    ]
    expected_trainedon = [
        '"Awesome Dataset One"^^<https://www.w3.org/2001/XMLSchema#string>',
        'https://mlentory.zbmed.de/mlentory_graph/dataset_no_name'
    ]
    resolved_trainedon = graph_handler_instance._resolve_identifier_list(trainedon_input, sample_entities_lookup)
    assert resolved_trainedon == expected_trainedon

    # Test testedOn
    testedon_list_from_fixture = model_data[SCHEMA.testedOn.n3()]
    testedon_input = [
         'https://mlentory.zbmed.de/mlentory_graph/dataset1',
         'https://mlentory.zbmed.de/mlentory_graph/dataset2',
         'https://mlentory.zbmed.de/mlentory_graph/dataset_not_in_lookup'
    ]
    expected_testedon = [
        '"Awesome Dataset One"^^<https://www.w3.org/2001/XMLSchema#string>',
        '"Spectacular Dataset Two"^^<https://www.w3.org/2001/XMLSchema#string>',
        'https://mlentory.zbmed.de/mlentory_graph/dataset_not_in_lookup'
    ]
    resolved_testedon = graph_handler_instance._resolve_identifier_list(testedon_input, sample_entities_lookup)
    assert resolved_testedon == expected_testedon

# Potential future test: Add a test for update_indexes_with_kg itself,
# mocking create_hf_dataset_index_entity_with_dict and checking the
# dictionary passed to it after resolution.
