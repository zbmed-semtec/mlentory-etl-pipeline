"""
Enumeration classes for standardizing naming conventions across the MLentory transform module.

This module provides standardized enumerations for entity types, platforms, and extraction
methods used throughout the MLentory transform process.
"""

from enum import Enum, auto
from typing import List


class SchemasURL(str, Enum):
    """
    Enumeration of schema URLs for the MLentory knowledge graph.

    These URLs represent the different schema namespaces used in the knowledge graph.

    Args:
        None

    Example:
        >>> schema = SchemasURL.SCHEMA
        >>> print(schema)
        'http://schema.org/'
    """
    SCHEMA = "https://schema.org/"
    FAIR4ML = "http://w3id.org/fair4ml/"
    CODEMETA = "https://w3id.org/codemeta/"
    CROISSANT = "http://mlcommons.org/croissant/"
    OWL = "http://www.w3.org/2002/07/owl#"
    RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    RDFS = "http://www.w3.org/2000/01/rdf-schema#"
    XSD = "http://www.w3.org/2001/XMLSchema#"


class EntityType(str, Enum):
    """
    Enumeration of entity types in the MLentory knowledge graph.

    These entity types represent the main concepts and objects that can be
    represented in the knowledge graph.

    Args:
        None

    Example:
        >>> entity_type = EntityType.MODEL
        >>> print(entity_type)
        'model'
    """
    MODEL = "model"
    DATASET = "dataset"
    FRAMEWORK = "framework"
    LIBRARY = "library"
    TASK = "task"
    METRIC = "metric"
    LANGUAGE = "language"
    LICENSE = "license"
    AUTHOR = "author"
    ORGANIZATION = "organization"
    PAPER = "paper"
    FIELD = "field"
    FILE_SET = "file_set"
    FILE = "file"
    FILE_OBJECT = "file_object"
    FILE_OBJECT_SET = "file_object_set"
    RECORD_SET = "record_set"


class Platform(str, Enum):
    """
    Enumeration of supported ML model and dataset platforms.

    Represents the various platforms from which models and datasets can be
    extracted and transformed.

    Args:
        None

    Example:
        >>> platform = Platform.HUGGING_FACE
        >>> print(platform)
        'hugging_face'
    """
    HUGGING_FACE = "hugging_face"
    OPEN_ML = "open_ml"
    PYTORCH_HUB = "pytorch_hub"
    TENSORFLOW_HUB = "tensorflow_hub"
    PAPERS_WITH_CODE = "papers_with_code"
    KAGGLE = "kaggle"
    GITHUB = "github"


class ExtractionMethod(str, Enum):
    """
    Enumeration of supported extraction methods.

    Represents the different methods used to extract information from various
    sources during the transformation process.

    Args:
        None

    Example:
        >>> method = ExtractionMethod.API
        >>> print(method)
        'api'
    """
    API = "api"
    SCRAPING = "scraping"
    FILE_SYSTEM = "file_system"
    DATABASE = "database"
    MANUAL = "manual"
    PARSED_FROM_HF = "parsed_from_hf"
    BUILT_IN_TRANSFORM = "built_in_transform"
    ADDED_IN_TRANSFORM = "added_in_transform"
    NOT_EXTRACTED = "not_extracted"
    SYSTEM = "system"


class RelationType(str, Enum):
    """
    Enumeration of relationship types between entities in the knowledge graph.

    Defines the standard relationship types that can exist between different
    entities in the MLentory knowledge graph.

    Args:
        None

    Example:
        >>> relation = RelationType.USES
        >>> print(relation)
        'uses'
    """
    USES = "uses"
    CREATED_BY = "created_by"
    BELONGS_TO = "belongs_to"
    IMPLEMENTS = "implements"
    EVALUATED_ON = "evaluated_on"
    TRAINED_ON = "trained_on"
    CITES = "cites"
    DEPENDS_ON = "depends_on"
    COMPATIBLE_WITH = "compatible_with"
    DERIVED_FROM = "derived_from"


def get_all_values(enum_class: Enum) -> List[str]:
    """
    Get all values from an enumeration class.

    Args:
        enum_class (Enum): The enumeration class to get values from.

    Returns:
        List[str]: List of all enumeration values as strings.

    Example:
        >>> values = get_all_values(Platform)
        >>> print(values)
        ['hugging_face', 'pytorch_hub', 'tensorflow_hub', ...]
    """
    return [e.value for e in enum_class] 