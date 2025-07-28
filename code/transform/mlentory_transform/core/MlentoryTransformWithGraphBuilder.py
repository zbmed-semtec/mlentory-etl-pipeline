from typing import List, Tuple, Dict, Any
import pandas as pd
import rdflib
from datetime import datetime
from tqdm import tqdm
import os
import hashlib

from ..utils.enums import Platform, SchemasURL
from .GraphBuilderBase import GraphBuilderBase
from .GraphBuilderFAIR4ML import GraphBuilderFAIR4ML
from .GraphBuilderCroissant import GraphBuilderCroissant
from .GraphBuilderArxiv import GraphBuilderArxiv
from .GraphBuilderKeyWords import GraphBuilderKeyWords
from .GraphBuilderLicense import GraphBuilderLicense

class MlentoryTransformWithGraphBuilder:
    def __init__(self, base_namespace: str = "https://example.org/", FAIR4ML_schema_data: pd.DataFrame = None):
        
        self.graph_builder_fair4ml = GraphBuilderFAIR4ML(base_namespace, FAIR4ML_schema_data)
        self.graph_builder_croissant = GraphBuilderCroissant(base_namespace)
        self.graph_builder_arxiv = GraphBuilderArxiv(base_namespace)
        self.graph_builder_keywords = GraphBuilderKeyWords(base_namespace)
        self.graph_builder_licenses = GraphBuilderLicense(base_namespace)
    

    def transform_HF_models_with_related_entities(
        self,
        extracted_entities: Dict[str, pd.DataFrame],
        save_intermediate_graphs: bool = False,
        save_output: bool = False,
        kg_output_dir: str = None,
        extraction_metadata_output_dir: str = None,
    ) -> Tuple[rdflib.Graph, rdflib.Graph]:
        """
        Transform the extracted data into a knowledge graph.
        """
        models_kg, models_extraction_metadata = self.transform_HF_models(
            extracted_df=extracted_entities["models"],
            save_output=save_intermediate_graphs,
            output_dir=kg_output_dir,
        )

        datasets_kg, datasets_extraction_metadata = self.transform_HF_datasets(
            extracted_df=extracted_entities["datasets"],
            save_output=save_intermediate_graphs,
            output_dir=kg_output_dir,
        )

        arxiv_kg, arxiv_extraction_metadata = self.transform_HF_arxiv(
            extracted_df=extracted_entities["articles"],
            save_output=save_intermediate_graphs,
            output_dir=kg_output_dir,
        )
        
        keywords_kg, keywords_extraction_metadata = self.transform_HF_keywords(
            extracted_df=extracted_entities["keywords"],
            save_output=save_intermediate_graphs,
            output_dir=kg_output_dir,
        )
        
        licenses_kg, licenses_extraction_metadata = self.transform_HF_licenses(
            extracted_df=extracted_entities["licenses"],
            save_output=save_intermediate_graphs,
            output_dir=kg_output_dir,
        )

        kg_integrated = self.unify_graphs(
            [models_kg, datasets_kg, arxiv_kg, keywords_kg, licenses_kg],
            save_output_in_json=save_output,
            output_dir=kg_output_dir,
            disambiguate_extraction_metadata=False
        )

        extraction_metadata_integrated = self.unify_graphs(
            [models_extraction_metadata,
             datasets_extraction_metadata,
             arxiv_extraction_metadata,
             keywords_extraction_metadata,
             licenses_extraction_metadata],
            save_output_in_json=save_output,
            output_dir=extraction_metadata_output_dir,
            disambiguate_extraction_metadata=True
        )
        
        return kg_integrated, extraction_metadata_integrated
    
    def transform_HF_models(
        self,
        extracted_df: pd.DataFrame,
        save_output: bool = False,
        output_dir: str = None,
    ) -> Tuple[rdflib.Graph, rdflib.Graph]:
        """
        Transform the extracted data into a knowledge graph and save it to a file.
        """
        
        knowledge_graph, extraction_metadata_graph = self.graph_builder_fair4ml.hf_dataframe_to_graph(extracted_df, identifier_column="schema.org:name", platform=Platform.HUGGING_FACE.value)
        
        if save_output:
            self.save_graph(knowledge_graph, "Transformed_HF_models_kg", output_dir)
            self.save_graph(extraction_metadata_graph, "Transformed_HF_models_kg_metadata", output_dir)

        return knowledge_graph, extraction_metadata_graph
    
    def transform_HF_datasets(
        self,
        extracted_df: pd.DataFrame,
        save_output: bool = False,
        output_dir: str = None,
    ) -> Tuple[rdflib.Graph, rdflib.Graph]:
        
        knowledge_graph, extraction_metadata_graph = self.graph_builder_croissant.hf_dataframe_to_graph(extracted_df, identifier_column="datasetId", platform=Platform.HUGGING_FACE.value)
        
        if save_output:
            self.save_graph(knowledge_graph, "Transformed_HF_datasets_kg", output_dir)
            self.save_graph(extraction_metadata_graph, "Transformed_HF_datasets_kg_metadata", output_dir)

        return knowledge_graph, extraction_metadata_graph
    
    def transform_HF_arxiv(
        self,
        extracted_df: pd.DataFrame,
        save_output: bool = False,
        output_dir: str = None,
    ) -> Tuple[rdflib.Graph, rdflib.Graph]:
        
        knowledge_graph, extraction_metadata_graph = self.graph_builder_arxiv.hf_dataframe_to_graph(extracted_df, identifier_column="arxiv_id", platform=Platform.HUGGING_FACE.value)
        
        if save_output:
            self.save_graph(knowledge_graph, "Transformed_HF_kg", output_dir)
            self.save_graph(extraction_metadata_graph, "Transformed_HF_kg_metadata", output_dir)

        return knowledge_graph, extraction_metadata_graph

    def transform_HF_licenses(
        self,
        extracted_df: pd.DataFrame,
        save_output: bool = False,
        output_dir: str = None,
    ) -> Tuple[rdflib.Graph, rdflib.Graph]:
        
        knowledge_graph, extraction_metadata_graph = self.graph_builder_licenses.hf_dataframe_to_graph(extracted_df, identifier_column="Name", platform=Platform.HUGGING_FACE.value)
        
        if save_output:
            self.save_graph(knowledge_graph, "Transformed_HF_licenses_kg", output_dir)
            self.save_graph(extraction_metadata_graph, "Transformed_HF_licenses_kg_metadata", output_dir)

        return knowledge_graph, extraction_metadata_graph
    
    def transform_HF_keywords(
        self,
        extracted_df: pd.DataFrame,
        save_output: bool = False,
        output_dir: str = None,
    ) -> Tuple[rdflib.Graph, rdflib.Graph]:
        
        knowledge_graph, extraction_metadata_graph = self.graph_builder_keywords.hf_dataframe_to_graph(extracted_df, identifier_column="tag_name", platform=Platform.HUGGING_FACE.value)
        
        if save_output:
            self.save_graph(knowledge_graph, "Transformed_HF_keywords_kg", output_dir)
            self.save_graph(extraction_metadata_graph, "Transformed_HF_keywords_kg_metadata", output_dir)

        return knowledge_graph, extraction_metadata_graph
    
    def unify_graphs(
        self,
        graphs: List[rdflib.Graph],
        save_output_in_json: bool = False,
        output_dir: str = None,
        disambiguate_extraction_metadata: bool = True,
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

        if save_output_in_json:
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            kg_output_path = os.path.join(output_dir, f"{current_date}_unified_kg.nt")
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
        
        
        print("DISAMBIGUATE STATEMENT METADATA ::::::::::: ")
        
        # Define the RDF types and properties we need
        RDF = rdflib.Namespace(SchemasURL.RDF.value)
        NS1 = rdflib.Namespace(SchemasURL.MLENTORY.value+"meta/")
        TYPE = RDF.type
        STATEMENT_METADATA = NS1.StatementMetadata
        CONFIDENCE = NS1.confidence
        EXTRACTION_TIME = NS1.extractionTime
        SUBJECT = NS1.subject
        PREDICATE = NS1.predicate
        OBJECT = NS1.object
        
        # Find all StatementMetadata instances
        metadata_nodes = set(graph.subjects(TYPE, STATEMENT_METADATA))
        
        print("NUMBER OF STATEMENT METADATA ::::::::::: ", len(metadata_nodes))
        
        # Group metadata by subject-predicate-object triple
        statement_groups = {}
        graph_nodes = list()
        
        for node in metadata_nodes:
            # Extract the statement this metadata is about
            subjects = list(graph.objects(node, SUBJECT))
            predicates = list(graph.objects(node, PREDICATE))
            objects = list(graph.objects(node, OBJECT))
            
            # Skip if any component is missing
            if not subjects or not predicates or not objects:
                continue
            
            # Use the first value if there are multiple (should not happen)
            statement_hash = hashlib.md5(
                    (
                        str(subjects[0].n3()) + str(predicates[0].n3()) + str(objects[0].n3())
                    ).encode()
                ).hexdigest()

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
            if statement_hash not in statement_groups or (
                (confidence > statement_groups[statement_hash][CONFIDENCE]) or
                (confidence == statement_groups[statement_hash][CONFIDENCE] and
                 extraction_time > statement_groups[statement_hash][EXTRACTION_TIME])
            ):
                # if (statement_hash in statement_groups) and (
                # (confidence > statement_groups[statement_hash][CONFIDENCE]) or
                # (confidence == statement_groups[statement_hash][CONFIDENCE] and
                #  extraction_time > statement_groups[statement_hash][EXTRACTION_TIME])
                # ):
                #     print("REPEATED TRIPLET ::::::::::: ", statement_hash)
                #     print("SUBJECTS", subjects)
                #     print("PREDICATES", predicates)
                #     print("OBJECTS", objects)
                #     print(statement_groups[statement_hash])
                    
                    
                statement_groups[statement_hash] = {
                    "node": node,
                    CONFIDENCE: confidence,
                    EXTRACTION_TIME: extraction_time
                }
                
        # Initialize a new graph for the disambiguated result
        disambiguated_graph = rdflib.Graph()
        
        # Add all triples from the original graph except StatementMetadata instances
        for node_info in statement_groups.values():
            for p, o in graph.predicate_objects(node_info["node"]):
                disambiguated_graph.add((node_info["node"], p, o))
            
        
        # Add only the best StatementMetadata for each statement
        # for best_metadata in statement_groups.values():
        #     node = best_metadata["node"]
        #     for p, o in graph.predicate_objects(node):
        #         disambiguated_graph.add((node, p, o))
        
        # Save the disambiguated graph if requested
        # save_output_in_json = True
        if save_output_in_json:
            if not output_dir:
                raise ValueError("output_dir must be provided if save_output_in_json is True")
                
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            kg_output_path = os.path.join(output_dir, f"{current_date}_disambiguated_kg.nt")
            before_disambiguation_kg_output_path = os.path.join(output_dir, f"{current_date}_before_disambiguation_kg.nt")
            graph.serialize(destination=before_disambiguation_kg_output_path, format="turtle")
            disambiguated_graph.serialize(destination=kg_output_path, format="turtle")
        
        return disambiguated_graph

    def save_graph(self, graph: rdflib.Graph, name: str, output_dir: str):
        """
        Save the graph to a file.
        """
        current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        kg_output_path = os.path.join(output_dir, f"{current_date}_{name}.nt")
        graph.serialize(destination=kg_output_path, format="nt")
        
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