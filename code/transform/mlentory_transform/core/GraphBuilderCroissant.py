from typing import List, Optional, Union, Dict, Tuple, Any
import pandas as pd
from rdflib import Graph, Literal, URIRef, BNode
from rdflib.namespace import RDF, XSD

from .GraphBuilderBase import GraphBuilderBase

class GraphBuilderCroissant(GraphBuilderBase):
    """
    A Knowledge Graph Handler specifically for the Croissant schema.

    Inherits common functionalities from BaseKnowledgeGraphHandler and implements
    methods to convert DataFrames based on the Croissant schema, handling JSON-LD
    and blank node replacements.

    Args:
        base_namespace (str): The base URI namespace for the knowledge graph entities.
            Default: "http://example.org/"
    """

    def __init__(self, base_namespace: str = "http://example.org/") -> None:
        """
        Initialize the CroissantKnowledgeGraphHandler.

        Args:
            base_namespace (str): The base URI namespace.
        """
        super().__init__(base_namespace)

    def hf_dataframe_to_graph(
        self,
        df: pd.DataFrame,
        identifier_column: Optional[str] = None,
        platform: str = None,
    ) -> Tuple[Graph, Graph]:
        """
        Convert a DataFrame to a Knowledge Graph using the Croissant schema.

        This method processes JSON-LD data within the DataFrame, parses it into
        temporary graphs, replaces blank nodes, and integrates the triples into
        the main graph along with metadata.

        Args:
            df (pd.DataFrame): The DataFrame to convert. Expects a column 'croissant_metadata'
                               containing JSON-LD strings and 'extraction_metadata'.
            identifier_column (Optional[str]): The column to use as the identifier for the entities.
                                           (Currently unused in this specific implementation but kept for consistency).
            platform (str): The platform name for the entities in the DataFrame.

        Returns:
            Tuple[Graph, Graph]: A tuple containing:
                - The Knowledge Graph created from the DataFrame
                - The Metadata Graph containing provenance information

        Raises:
            ValueError: If the DataFrame is empty or essential columns are missing.
        """
        if df.empty:
            print("Warning: Cannot convert empty DataFrame to graph")
            return self.graph, self.metadata_graph

        if identifier_column and identifier_column not in df.columns:
            raise ValueError(
                f"Identifier column '{identifier_column}' not found in DataFrame"
            )

        for idx, row in df.iterrows():
            item_json_ld = row["croissant_metadata"]
            temp_graph = Graph()
            temp_graph.parse(
                data=item_json_ld, format="json-ld", base=URIRef(self.base_namespace)
            )

            self.delete_unwanted_nodes(temp_graph)
            self.replace_blank_nodes_with_type(temp_graph, row, platform)
            self.delete_remaining_blank_nodes(temp_graph)
            # self.replace_blank_nodes_with_no_type(temp_graph,row,platform)
            # self.delete_remaining_blank_nodes(temp_graph)
            # self.replace_default_nodes(temp_graph, row, platform)
            

            # Go through the triples and add them
            for triple in temp_graph:
                # print("TRIPLE:", triple)
                # Transform the triple to the correct format

                self.add_triple_with_metadata(
                    triple[0],
                    triple[1],
                    triple[2],
                    {
                        "extraction_method": row["extraction_metadata"][
                            "extraction_method"
                        ],
                        "confidence": row["extraction_metadata"]["confidence"],
                    },
                    row["extraction_metadata"]["extraction_time"],
                )

        return self.graph, self.metadata_graph
    
    
    def replace_blank_nodes_with_type(
        self, graph: Graph, row: pd.Series, platform: str
    ) -> None:
        """
        Replace typed blank nodes in the graph with hashed URIs based on their type and identifier.

        Identifies blank nodes with types like schema:Dataset, schema:Organization, schema:Person,
        and replaces them with stable URIs generated using the entity hash function.

        Args:
            graph (Graph): The RDF graph to modify.
            row (pd.Series): The DataFrame row containing context (e.g., datasetId, name).
            platform (str): The platform identifier.
        """
        # Use a copy of keys to avoid modification during iteration issues
        blank_nodes_info = self.identify_blank_nodes_with_type(graph)
        type_map = {
             "https://schema.org/Dataset": ("datasetId", "Dataset"),
             "https://schema.org/Organization": ("name", "Organization"),
             "https://schema.org/Person": ("name", "Person"),
        }

        for node_type_uri, nodes in blank_nodes_info.items():
             if node_type_uri in type_map:
                 identifier_key, entity_type_name = type_map[node_type_uri]
                 for node_data in nodes:
                     # Determine the identifier: use row data if possible, else node properties
                     identifier = None
                     if node_type_uri == "https://schema.org/Dataset" and "datasetId" in row:
                         identifier = row["datasetId"]
                     elif node_type_uri in ["https://schema.org/Organization", "https://schema.org/Person"]:
                         # Prefer name from properties if available
                         identifier = node_data.get("properties", {}).get("https://schema.org/name")
                         # Fallback: Use name from row if needed (adjust if row structure differs)
                         # if not identifier and 'name' in row: # Example fallback
                         #     identifier = row['name']

                     if identifier:
                         # Replace using the new style with explicit type
                         self.replace_node(
                             old_id=BNode(node_data["node_id"].split("_:")[1]), # Convert BNode string back to BNode
                             new_id=str(identifier), # Pass the identifier string
                             graph=graph,
                             platform=platform,
                             type=node_type_uri # Use the actual schema type URI
                         )
                     else:
                         print(f"Warning: Could not determine identifier for blank node {node_data['node_id']} of type {node_type_uri}. Skipping replacement.")


    def identify_blank_nodes_with_type(
        self, graph: Graph
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify all blank nodes in the graph that have an explicit rdf:type.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary where keys are type URIs (str)
            and values are lists of dictionaries, each representing a blank node with:
                - node_id: The blank node identifier string (e.g., '_:b0').
                - properties: A dictionary of predicate URIs (str) to object values (str).
        """
        blank_nodes = {}
        query = """
        SELECT DISTINCT ?node ?type
        WHERE {
            ?node a ?type .
            FILTER(isBlank(?node))
        }
        """
        for res_row in graph.query(query):
            node, node_type = res_row
            type_str = str(node_type)
            node_id_str = node.n3() # Get the string representation like '_:b0'

            if type_str not in blank_nodes:
                blank_nodes[type_str] = []

            properties = {}
            for pred, obj in graph.predicate_objects(node):
                pred_str = str(pred)
                # Store object value as string for simplicity here
                if isinstance(obj, Literal):
                    properties[pred_str] = obj.toPython() # Get Python native value
                elif isinstance(obj, URIRef):
                     properties[pred_str] = str(obj)
                elif isinstance(obj, BNode):
                    properties[pred_str] = obj.n3() # Keep BNode reference string
                else:
                     properties[pred_str] = str(obj)

            # Avoid adding duplicates if the same node is processed multiple times (shouldn't happen with DISTINCT)
            if not any(d['node_id'] == node_id_str for d in blank_nodes[type_str]):
                blank_nodes[type_str].append(
                    {"node_id": node_id_str, "properties": properties}
                )

        return blank_nodes

    # def replace_blank_nodes_with_no_type(
    #     self, graph: Graph, row: pd.Series, platform: str
    # ) -> None:
    #     """
    #     Identify blank nodes that don't have a type and create a new node with an unique ID.
    #     NOTE: This method's logic might need review/adjustment based on current needs.
    #           It assumes a parent/relation structure that might not always exist.
    #           Currently commented out as its necessity is unclear.

    #     Args:
    #         graph (Graph): The RDF graph to replace blank nodes in
    #         row (pd.Series): The row containing the datasetId and other metadata
    #         platform (str): The platform prefix to use in the URIs
    #     """
    #     # This function needs review - identifying untyped BNodes and determining
    #     # their appropriate replacement ID and type can be complex.
    #     # The original logic assumed a specific parent/relation structure.
    #     print("Warning: replace_blank_nodes_with_no_type is currently inactive and needs review.")
    #     # blank_nodes = self.identify_blank_nodes_with_no_type(graph)
    #     # for blank_node in blank_nodes:
    #     #     # Logic to determine new_id and type needs refinement
    #     #     new_id = (
    #     #         blank_node["parent_id"] # Requires parent_id to be identified correctly
    #     #         + "/"
    #     #         + blank_node["relation_type"].split("/")[-1] # Requires relation_type
    #     #     )
    #     #     self.replace_node(
    #     #         old_id=BNode(blank_node["node_id"].split("_:")[1]),
    #     #         new_id=new_id,
    #     #         graph=graph,
    #     #         platform=platform,
    #     #         type="https://schema.org/Thing" # Default type, might need adjustment
    #     #     )
    #     pass

    # def identify_blank_nodes_with_no_type(self, graph: Graph) -> List[Dict[str, Any]]:
    #      """
    #      Identify blank nodes that do not have an rdf:type assertion.
    #      NOTE: This method's logic relies on assumptions about identifying parent/relation
    #            and might need significant revision.

    #      Returns:
    #          List[Dict[str, Any]]: List of dictionaries for untyped blank nodes.
    #                                Structure needs defining based on requirements.
    #      """
    #      # This query/logic needs careful consideration. How do we reliably
    #      # identify the context (parent, relation) for an untyped BNode?
    #      untyped_bnodes = []
    #      all_bnodes = {node for node in graph.all_nodes() if isinstance(node, BNode)}
    #      typed_bnodes = {s for s, p, o in graph.triples((None, RDF.type, None)) if isinstance(s, BNode)}
    #      nodes_to_check = all_bnodes - typed_bnodes

    #      for bnode in nodes_to_check:
    #          # How to determine parent_id and relation_type?
    #          # This is non-trivial and depends on the expected graph structure.
    #          # Placeholder logic:
    #          parent_id = "unknown_parent"
    #          relation_type = "unknown_relation"
    #          # Example: Find incoming triple?
    #          # for s, p, o in graph.triples((None, None, bnode)):
    #          #     if isinstance(s, URIRef):
    #          #         parent_id = str(s)
    #          #         relation_type = str(p)
    #          #         break

    #          untyped_bnodes.append({
    #              "node_id": bnode.n3(),
    #              # "parent_id": parent_id, # Requires reliable identification
    #              # "relation_type": relation_type # Requires reliable identification
    #          })
    #      return untyped_bnodes

    def replace_default_nodes(self, temp_graph: Graph, row: pd.Series, platform: str) -> None:
        """
        Identify and update nodes in the Croissant schema with default URIs.

        Replaces default URIs (often based on `mlcommons.org/croissant/.../default/...`)
        for types like Field, RecordSet, etc., with hashed URIs incorporating
        the dataset ID to ensure uniqueness.

        Args:
            temp_graph (Graph): The temporary graph parsed from JSON-LD.
            row (pd.Series): DataFrame row containing dataset metadata (requires 'datasetId').
            platform (str): Platform identifier (e.g., 'HF').
        """
        if "datasetId" not in row or not row["datasetId"]:
            print("Warning: 'datasetId' missing or empty in row. Skipping default node replacement.")
            return

        dataset_id = row["datasetId"]

        field_types = [
            URIRef("http://mlcommons.org/croissant/Field"),
            URIRef("http://mlcommons.org/croissant/RecordSet"),
            # Add other types if needed, e.g.:
            # URIRef("http://mlcommons.org/croissant/FileSet"),
            # URIRef("http://mlcommons.org/croissant/FileObject"),
        ]

        nodes_to_replace = []
        for field_type in field_types:
            for field_node in temp_graph.subjects(RDF.type, field_type):
                if isinstance(field_node, URIRef):
                    # Check if it looks like a default/relative URI needing replacement
                    # This check might need refinement based on actual URI patterns
                    if "/default/" in str(field_node) or not field_node.startswith("http"):
                         nodes_to_replace.append((field_node, field_type))
                    elif str(field_node).startswith(self.base_namespace):
                         # Check if it's potentially a base-namespaced default node
                         # Extract the part after the base namespace
                         relative_part = str(field_node)[len(self.base_namespace):]
                         if not '/' in relative_part: # Simple check: if it has no slashes, might be default
                              nodes_to_replace.append((field_node, field_type))


        for field_node, field_type in nodes_to_replace:
             original_type_name = str(field_type).split("/")[-1]
             original_node_path = str(field_node).split(self.base_namespace)[-1] # Get path relative to base or full path

             # Create a unique identifier string for hashing
             # Combine original path and dataset ID for uniqueness
             unique_id_str = f"{original_node_path}/{dataset_id}"

             # Generate hash using the base class method
             entity_hash = self.generate_entity_hash(platform, original_type_name, unique_id_str)
             new_uri = self.base_namespace[entity_hash]

             # Replace the node using the base class method, providing the specific node_type
             self.replace_node(
                 old_id=field_node,
                 new_id=new_uri, # Pass the generated URIRef
                 graph=temp_graph,
                 node_type=field_type # Provide the explicit RDF type
             )

             # Optionally add a name/label based on the original path if helpful
             original_name = original_node_path.split('/')[-1]
             if original_name:
                 temp_graph.add((new_uri, self.namespaces["schema"]["name"], Literal(original_name, datatype=XSD.string)))


    def delete_remaining_blank_nodes(self, graph: Graph) -> None:
        """
        Delete all triples involving any remaining blank nodes (BNode).

        Removes triples where the subject, predicate, or object is a BNode.
        This is typically used as a cleanup step after attempting to replace
        typed BNodes.

        Args:
            graph (Graph): The RDF graph to clean.
        """
        triples_to_remove = [
            triple for triple in graph
            if isinstance(triple[0], BNode) or isinstance(triple[2], BNode)
             # Don't remove based on predicate being BNode (less common, might be valid)
        ]

        if triples_to_remove:
            # print(f"Deleting {len(triples_to_remove)} triples involving remaining blank nodes.")
            for triple in triples_to_remove:
                graph.remove(triple)

    def delete_unwanted_nodes(self, graph: Graph) -> None:
        """
        Delete all triples related to specific unwanted entity types from the Croissant schema.

        Removes triples where the subject or object is of type FileSet, File,
        FileObject, FileObjectSet, or RecordSet. This simplifies the graph by
        removing detailed file/record structure information if it's not needed.

        Args:
            graph (Graph): The RDF graph to modify.
        """
        unwanted_types = [
            URIRef("http://mlcommons.org/croissant/FileSet"),
            URIRef("http://mlcommons.org/croissant/File"),
            URIRef("http://mlcommons.org/croissant/FileObject"),
            URIRef("http://mlcommons.org/croissant/FileObjectSet"),
            # Keep RecordSet and Field for now, delete if strictly not needed
            # URIRef("http://mlcommons.org/croissant/RecordSet"),
            # URIRef("http://mlcommons.org/croissant/Field"),
        ]

        unwanted_nodes = set()
        for type_uri in unwanted_types:
            for subject in graph.subjects(RDF.type, type_uri):
                 unwanted_nodes.add(subject)

        if not unwanted_nodes:
            return # Nothing to delete

        triples_to_remove = set()
        for node in unwanted_nodes:
             # Find triples where the node is the subject OR the object
             for triple in graph.triples((node, None, None)):
                 triples_to_remove.add(triple)
             for triple in graph.triples((None, None, node)):
                 triples_to_remove.add(triple)

        if triples_to_remove:
            # print(f"Deleting {len(triples_to_remove)} triples related to unwanted node types: {unwanted_types}")
            for triple in triples_to_remove:
                graph.remove(triple) 