from typing import List, Tuple, Dict, Any
import pandas as pd
import rdflib
from datetime import datetime
from tqdm import tqdm
import os

from ..hf_transform.TransformHF import TransformHF
from .KnowledgeGraphHandler import KnowledgeGraphHandler
from ..utils.enums import Platform


class MlentoryTransform:
    """
    A class for transforming data from different sources into a unified knowledge graph.

    This class provides functionality to:
    - Transform data from multiple sources (HF, OpenML, etc.)
    - Standardize data formats
    - Handle metadata extraction
    - Create unified knowledge representations

    Attributes:
        sources (List[Tuple[str, pd.DataFrame]]): List of data sources and their dataframes
        schema (pd.DataFrame): Target schema for the transformed data
        transformations (pd.DataFrame): Mapping rules for data transformation
    """

    def __init__(self, kg_handler: KnowledgeGraphHandler, transform_hf: TransformHF):
        """
        Initialize the transformer with schema and transformation rules.

        Args:
            kg_handler (KnowledgeGraphHandler): Knowledge graph handler
            transform_hf (TransformHF): Transform HF
        Example:
            >>> schema_df = pd.read_csv("schema.tsv", sep="\t")
            >>> transform_df = pd.read_csv("transformations.csv")
            >>> transformer = MlentoryTransform(schema_df, transform_df)
        """
        self.processed_data = []
        self.current_sources = {}
        self.kg_handler = kg_handler
        self.transform_hf = transform_hf

    def transform_HF_models_with_related_entities(
        self,
        extracted_entities: Dict[str, pd.DataFrame],
        save_output: bool = False,
        kg_output_dir: str = None,
        extraction_metadata_output_dir: str = None,
    ) -> Tuple[rdflib.Graph, rdflib.Graph]:
        """
        Transform the extracted data into a knowledge graph.
        """
        models_kg, models_extraction_metadata = self.transform_HF_models(
            extracted_df=extracted_entities["models"],
            save_output_in_json=False,
            output_dir=kg_output_dir+"/models",
        )

        datasets_kg, datasets_extraction_metadata = self.transform_HF_datasets(
            extracted_df=extracted_entities["datasets"],
            save_output_in_json=False,
            output_dir=kg_output_dir,
        )

        arxiv_kg, arxiv_extraction_metadata = self.transform_HF_arxiv(
            extracted_df=extracted_entities["articles"],
            save_output_in_json=False,
            output_dir=kg_output_dir,
        )
        
        keywords_kg, keywords_extraction_metadata = self.transform_HF_keywords(
            extracted_df=extracted_entities["keywords"],
            save_output_in_json=True,
            output_dir=kg_output_dir,
        )

        kg_integrated = self.unify_graphs(
            [models_kg, datasets_kg, arxiv_kg, keywords_kg],
            save_output_in_json=save_output,
            output_dir=kg_output_dir,
        )

        extraction_metadata_integrated = self.unify_graphs(
            [models_extraction_metadata,
             datasets_extraction_metadata,
             arxiv_extraction_metadata,
             keywords_extraction_metadata],
            save_output_in_json=save_output,
            output_dir=extraction_metadata_output_dir,
        )
        
        return kg_integrated, extraction_metadata_integrated

    def transform_data(
        self,
        extracted_df: pd.DataFrame,
        platform: str,
        identifier_column: str,
        save_output_in_json: bool = False,
        output_dir: str = None,
    ) -> Tuple[rdflib.Graph, rdflib.Graph]:
        """
        Transform the extracted data into a knowledge graph.

        Args:
            extracted_df (pd.DataFrame): DataFrame containing extracted data
            platform (str): Platform name (e.g., Platform.OPEN_ML.value or Platform.HUGGING_FACE.value)
            save_output_in_json (bool, optional): Whether to save the transformed data.
                Defaults to False.
            output_dir (str, optional): Directory to save the transformed data.
                Required if save_output_in_json is True.
            identifier_column (str, optional): Column name to use as identifier.

        Returns:
            Tuple[rdflib.Graph, rdflib.Graph]: Transformed knowledge graph and metadata graph

        Raises:
            ValueError: If save_output_in_json is True but output_dir is not provided
        """
        # Reset the knowledge graph handler before processing new data
        self.kg_handler.reset_graphs()

        # Ensure the directory exists if saving output
        if save_output_in_json:
            if not output_dir:
                raise ValueError("output_dir must be provided when save_output_in_json is True")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, mode=0o777)

        # Transform the dataframe to a knowledge graph
        knowledge_graph, metadata_graph = (
            self.kg_handler.dataframe_to_graph_FAIR4ML_schema(
                df=extracted_df,
                identifier_column=identifier_column,
                platform=platform
            )
        )

        self.current_sources[platform] = knowledge_graph
        self.current_sources[f"{platform}_metadata"] = metadata_graph

        if save_output_in_json:
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            kg_output_path = os.path.join(
                output_dir, f"{current_date}_Transformed_{platform}_kg.json"
            )
            metadata_output_path = os.path.join(
                output_dir, f"{current_date}_Transformed_{platform}_kg_metadata.json"
            )
            knowledge_graph.serialize(destination=kg_output_path, format="json-ld")
            metadata_graph.serialize(destination=metadata_output_path, format="json-ld")

        return knowledge_graph, metadata_graph
    
    def transform_OpenML_runs(self, extracted_df, save_output_in_json=False, output_dir=None):
        return self.transform_data(
            extracted_df=extracted_df,
            platform=Platform.OPEN_ML.value,
            save_output_in_json=save_output_in_json,
            output_dir=output_dir,
            identifier_column="schema.org:name"
        )
    
    def transform_OpenML_datasets(self, extracted_df, save_output_in_json=False, output_dir=None):
        return self.transform_data(
            extracted_df=extracted_df,
            platform=Platform.OPEN_ML.value,
            save_output_in_json=save_output_in_json,
            output_dir=output_dir,
            identifier_column="schema.org:identifier"
        )

    def transform_HF_models(self, extracted_df, save_output_in_json=False, output_dir=None):
        return self.transform_data(
            extracted_df=extracted_df,
            platform=Platform.HUGGING_FACE.value,
            save_output_in_json=save_output_in_json,
            output_dir=output_dir,
            identifier_column="schema.org:name"
        )

    def transform_HF_datasets(
        self,
        extracted_df: pd.DataFrame,
        save_output_in_json: bool = False,
        output_dir: str = None,
    ) -> Tuple[rdflib.Graph, rdflib.Graph]:
        """
        Transform the extracted data into a knowledge graph.

        Args:
            extracted_df (pd.DataFrame): DataFrame containing extracted dataset data
            It has three columns:
                - "datasetId": The HuggingFace dataset id
                - "croissant_metadata": Dataset data in croissant format in a json object
                - "extraction_metadata": Extraction metadata dictionary
            save_output_in_json (bool, optional): Whether to save the transformed data.
                Defaults to False.
            output_dir (str, optional): Directory to save the transformed data.
                Required if save_output_in_json is True.

        Returns:
            Tuple[rdflib.Graph, rdflib.Graph]: Transformed knowledge graph and metadata graph
        """
        # Reset the knowledge graph handler before processing new data
        self.kg_handler.reset_graphs()

        # Transform the dataframe to a knowledge graph
        knowledge_graph, metadata_graph = (
            self.kg_handler.dataframe_to_graph_Croissant_schema(
                df=extracted_df, 
                identifier_column="datasetId", 
                platform=Platform.HUGGING_FACE.value
            )
        )

        self.current_sources[f"{Platform.HUGGING_FACE.value}_dataset"] = knowledge_graph
        self.current_sources[f"{Platform.HUGGING_FACE.value}_dataset_metadata"] = metadata_graph

        if save_output_in_json:
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            kg_output_path = os.path.join(
                output_dir, f"{current_date}_Processed_HF_kg.json"
            )
            metadata_output_path = os.path.join(
                output_dir, f"{current_date}_Processed_HF_kg_metadata.json"
            )
            knowledge_graph.serialize(destination=kg_output_path, format="json-ld")
            metadata_graph.serialize(destination=metadata_output_path, format="json-ld")

        return knowledge_graph, metadata_graph

    def transform_HF_arxiv(
        self,
        extracted_df: pd.DataFrame,
        save_output_in_json: bool = False,
        output_dir: str = None,
    ) -> Tuple[rdflib.Graph, rdflib.Graph]:
        """
        Transform the extracted data into a knowledge graph.
        """
        # Reset the knowledge graph handler before processing new data
        self.kg_handler.reset_graphs()

        # Transform the dataframe to a knowledge graph
        knowledge_graph, metadata_graph = (
            self.kg_handler.dataframe_to_graph_arXiv_schema(
                df=extracted_df, 
                identifier_column="arxiv_id", 
                platform=Platform.HUGGING_FACE.value
            )
        )
        
        self.current_sources[f"{Platform.HUGGING_FACE.value}_arxiv"] = knowledge_graph
        self.current_sources[f"{Platform.HUGGING_FACE.value}_arxiv_metadata"] = metadata_graph
        
        if save_output_in_json:
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            kg_output_path = os.path.join(
                output_dir, f"{current_date}_Processed_HF_arxiv_kg.json"
            )
            metadata_output_path = os.path.join(
                output_dir, f"{current_date}_Processed_HF_arxiv_kg_metadata.json"
            )
            knowledge_graph.serialize(destination=kg_output_path, format="json-ld")
        
        return knowledge_graph, metadata_graph
    
    def transform_HF_keywords(
        self,
        extracted_df: pd.DataFrame,
        save_output_in_json: bool = False,
        output_dir: str = None,
    ) -> Tuple[rdflib.Graph, rdflib.Graph]:
        """
        Transform the extracted data into a knowledge graph.
        """
        # Reset the knowledge graph handler before processing new data
        self.kg_handler.reset_graphs()
        
        # Transform the dataframe to a knowledge graph
        knowledge_graph, metadata_graph = (
            self.kg_handler.dataframe_to_graph_keywords(
                df=extracted_df, 
                identifier_column="tag_name", 
                platform=Platform.HUGGING_FACE.value
            )
        )
        
        self.current_sources[f"{Platform.HUGGING_FACE.value}_keywords"] = knowledge_graph
        self.current_sources[f"{Platform.HUGGING_FACE.value}_keywords_metadata"] = metadata_graph
        
        if save_output_in_json:
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            kg_output_path = os.path.join(
                output_dir, f"{current_date}_Processed_HF_keywords_kg.json"
            )
            metadata_output_path = os.path.join(
                output_dir, f"{current_date}_Processed_HF_keywords_kg_metadata.json"
            )
            knowledge_graph.serialize(destination=kg_output_path, format="json-ld")
        
        return knowledge_graph, metadata_graph
    
    def unify_graphs(
        self,
        graphs: List[rdflib.Graph],
        save_output_in_json: bool = False,
        output_dir: str = None,
        disambiguate_extraction_metadata: bool = False,
    ) -> rdflib.Graph:
        """
        Unify the knowledge graph from the current sources.
        
        Args:
            graphs (List[rdflib.Graph]): List of graphs to unify
            save_output_in_json (bool, optional): Whether to save the transformed data.
                Defaults to False.
            output_dir (str, optional): Directory to save the transformed data.
                Required if save_output_in_json is True.
            disambiguate_extraction_metadata (bool, optional): Whether to disambiguate StatementMetadata
                entities after unification. Defaults to False.
                
        Returns:
            rdflib.Graph: Unified graph
            
        Example:
            >>> kg_handler = KnowledgeGraphHandler()
            >>> transform_hf = TransformHF()
            >>> transformer = MlentoryTransform(kg_handler, transform_hf)
            >>> unified_graph = transformer.unify_graphs([graph1, graph2])
        """
        unified_graph = rdflib.Graph()
        for graph in graphs:
            unified_graph += graph
            
        # Disambiguate metadata if requested
        if disambiguate_extraction_metadata:
            unified_graph = self.disambiguate_statement_metadata(unified_graph)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, mode=0o777)

        if save_output_in_json:
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            kg_output_path = os.path.join(output_dir, f"{current_date}_unified_kg.ttl")
            unified_graph.serialize(destination=kg_output_path, format="turtle")

        return unified_graph

    def disambiguate_statement_metadata(
        self, 
        graph: rdflib.Graph,
        save_output_in_json: bool = False,
        output_dir: str = None,
    ) -> rdflib.Graph:
        """
        Disambiguate StatementMetadata entities in the graph.
        
        When multiple StatementMetadata entities refer to the same statement 
        (same subject, predicate, object), this method selects the entity with
        the highest confidence and most recent extractionTime.
        
        Args:
            graph (rdflib.Graph): The graph containing potentially duplicate metadata
            save_output_in_json (bool, optional): Whether to save the transformed data.
                Defaults to False.
            output_dir (str, optional): Directory to save the transformed data.
                Required if save_output_in_json is True.
                
        Returns:
            rdflib.Graph: Graph with disambiguated metadata
            
        Example:
            >>> kg_handler = KnowledgeGraphHandler()
            >>> transform_hf = TransformHF()
            >>> transformer = MlentoryTransform(kg_handler, transform_hf)
            >>> unified_graph = transformer.unify_graphs([graph1, graph2])
            >>> disambiguated_graph = transformer.disambiguate_statement_metadata(unified_graph)
        """
        # Initialize a new graph for the disambiguated result
        disambiguated_graph = rdflib.Graph()
        
        # Define the RDF types and properties we need
        RDF = rdflib.Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        NS1 = rdflib.Namespace("http://mlentory.de/ns1#")
        TYPE = RDF.type
        STATEMENT_METADATA = NS1.StatementMetadata
        CONFIDENCE = NS1.confidence
        EXTRACTION_TIME = NS1.extractionTime
        SUBJECT = NS1.subject
        PREDICATE = NS1.predicate
        OBJECT = NS1.object
        
        # Find all StatementMetadata instances
        metadata_nodes = list(graph.subjects(TYPE, STATEMENT_METADATA))
        
        # Group metadata by subject-predicate-object triple
        statement_groups = {}
        
        for node in metadata_nodes:
            # Extract the statement this metadata is about
            subjects = list(graph.objects(node, SUBJECT))
            predicates = list(graph.objects(node, PREDICATE))
            objects = list(graph.objects(node, OBJECT))
            
            # Skip if any component is missing
            if not subjects or not predicates or not objects:
                continue
                
            # Use the first value if there are multiple (should not happen)
            statement_key = (str(subjects[0]), str(predicates[0]), str(objects[0]))
            
            # Extract confidence and extraction time
            confidences = list(graph.objects(node, CONFIDENCE))
            extraction_times = list(graph.objects(node, EXTRACTION_TIME))
            
            # Default values if not found
            confidence = float(confidences[0].toPython()) if confidences else 0.0
            
            # Convert extraction time to datetime for comparison
            if extraction_times:
                time_str = str(extraction_times[0])
                # Handle different datetime formats
                try:
                    extraction_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                except ValueError:
                    try:
                        # Try with different format
                        extraction_time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        # If all fails, use a minimum datetime
                        extraction_time = datetime.min
            else:
                extraction_time = datetime.min
            
            # Add to group or replace if better
            if statement_key not in statement_groups or (
                (confidence > statement_groups[statement_key]["confidence"]) or
                (confidence == statement_groups[statement_key]["confidence"] and
                 extraction_time > statement_groups[statement_key]["extraction_time"])
            ):
                statement_groups[statement_key] = {
                    "node": node,
                    "confidence": confidence,
                    "extraction_time": extraction_time
                }
        
        # Add all triples from the original graph except StatementMetadata instances
        for s, p, o in graph:
            if s not in metadata_nodes:
                disambiguated_graph.add((s, p, o))
        
        # Add only the best StatementMetadata for each statement
        for best_metadata in statement_groups.values():
            node = best_metadata["node"]
            for p, o in graph.predicate_objects(node):
                disambiguated_graph.add((node, p, o))
        
        # Save the disambiguated graph if requested
        if save_output_in_json:
            if not output_dir:
                raise ValueError("output_dir must be provided if save_output_in_json is True")
                
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            kg_output_path = os.path.join(output_dir, f"{current_date}_disambiguated_kg.ttl")
            disambiguated_graph.serialize(destination=kg_output_path, format="turtle")
        
        return disambiguated_graph

    def print_detailed_dataframe(self, df: pd.DataFrame):
        """
        Print the detailed dataframe
        """
        print("\n**DATAFRAME**")
        print("\nColumns:", df.columns.tolist())
        print("\nShape:", df.shape)
        print("\nSample Data:")
        for col in df.columns:
            print("--------------------------------------------")
            print(f"\n{col}:")
            for row in df[col]:
                # Limit the text to 100 characters
                if isinstance(row, list):
                    row_data = row[0]["data"]
                    if isinstance(row_data, str):
                        print(row_data[:100])
                    else:
                        print(row_data)
                else:
                    print(row)
            print("--------------------------------------------")
            print()
        print("\nDataFrame Info:")
        print(df.info())

    @staticmethod
    def disambiguate_unified_graph(
        input_file_path: str,
        output_file_path: str,
        format: str = "turtle"
    ) -> None:
        """
        Standalone utility method to disambiguate an existing unified RDF graph file.
        
        This method loads a graph from a file, disambiguates its StatementMetadata entities,
        and saves the result to a new file.
        
        Args:
            input_file_path (str): Path to the input graph file
            output_file_path (str): Path where the disambiguated graph will be saved
            format (str, optional): RDF serialization format. Defaults to "turtle".
            
        Returns:
            None
            
        Raises:
            FileNotFoundError: If the input file does not exist
            ValueError: If an invalid format is specified
            
        Example:
            >>> MlentoryTransform.disambiguate_unified_graph(
            ...     "path/to/unified_kg.ttl", 
            ...     "path/to/disambiguated_kg.ttl"
            ... )
        """
        # Check if input file exists
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")
            
        # Load the graph
        print(f"Loading graph from {input_file_path}...")
        graph = rdflib.Graph()
        graph.parse(input_file_path, format=format)
        print(f"Loaded graph with {len(graph)} triples")
        
        # Create an instance of MlentoryTransform with minimal dependencies
        # This is needed because disambiguate_statement_metadata is an instance method
        transform = MlentoryTransform(
            kg_handler=KnowledgeGraphHandler(),
            transform_hf=TransformHF()
        )
        
        # Disambiguate the graph
        print("Disambiguating graph metadata...")
        disambiguated_graph = transform.disambiguate_statement_metadata(graph)
        print(f"Disambiguated graph has {len(disambiguated_graph)} triples")
        
        # Save the result
        print(f"Saving disambiguated graph to {output_file_path}...")
        disambiguated_graph.serialize(destination=output_file_path, format=format)
        print("Done!")
