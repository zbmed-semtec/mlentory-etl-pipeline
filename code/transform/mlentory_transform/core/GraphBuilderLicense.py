from typing import List, Optional, Union, Dict, Tuple, Any
import pandas as pd
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from .GraphBuilderBase import GraphBuilderBase

class GraphBuilderLicense(GraphBuilderBase):
    """
    A Knowledge Graph Handler specifically for License data using SPDX information.

    Inherits common functionalities from GraphBuilderBase and implements
    methods to convert DataFrames containing license metadata into an RDF graph,
    primarily creating schema:CreativeWork entities for licenses.

    Args:
        base_namespace (str): The base URI namespace for the knowledge graph entities.
            Default: "http://example.org/"
    """
    def __init__(self, base_namespace: str = "http://example.org/") -> None:
        """
        Initialize the GraphBuilderLicense.

        Args:
            base_namespace (str): The base URI namespace.
        """
        super().__init__(base_namespace)

    def hf_dataframe_to_graph(
        self,
        df: pd.DataFrame,
        identifier_column: Optional[str] = None,
        platform: str = None, # Default platform to be set by caller
    ) -> Tuple[Graph, Graph]:
        """
        Convert a DataFrame containing SPDX license data to a Knowledge Graph.

        Maps columns like 'Name', 'Identifier', 'URL', 'Text' to
        schema.org properties for CreativeWork entities.

        Args:
            df (pd.DataFrame): The DataFrame with license data.
            identifier_column (Optional[str]): The column containing the license identifier (e.g., SPDX ID).
                                           If None, it might default or raise error depending on usage.
            platform (str): The platform name (e.g., "SPDX", "HuggingFace").

        Returns:
            Tuple[Graph, Graph]: A tuple containing:
                - The Knowledge Graph created from the DataFrame
                - The Metadata Graph containing provenance information

        Raises:
            ValueError: If the DataFrame is empty or the identifier column is missing/invalid when provided.
        """
        if df.empty:
            return self.graph, self.metadata_graph

        if identifier_column and identifier_column not in df.columns:
            raise ValueError(
                f"Identifier column '{identifier_column}' not found in DataFrame"
            )
        
        # Determine a default identifier column if not provided and possible, or raise error
        # For licenses, 'Identifier' or 'Name' could be common.
        # For now, we'll assume identifier_column is correctly passed or an error is acceptable.


        for idx, row in df.iterrows():
            # Use the provided identifier_column or a fallback if sensible (e.g. 'Name')
            # For this implementation, we stick to the provided identifier_column
            if not identifier_column or pd.isna(row[identifier_column]):
                # Fallback to 'Name' if identifier is missing, or skip row
                if "Name" in row and not pd.isna(row["Name"]):
                    entity_id_source = row["Name"]
                else:
                    print(f"Skipping row {idx} due to missing identifier and Name.")
                    continue # Skip this row if no usable identifier
            else:
                entity_id_source = row[identifier_column]

            entity_id = str(entity_id_source[0]).strip().lower()
            print(f"Creating in License CreativeWork for {entity_id}")
            # Using "License" as a more specific type for hash generation within the platform context
            id_hash = self.generate_entity_hash(platform, "CreativeWork", entity_id) 
            creative_work_uri = self.base_namespace[id_hash]
            
            # Default extraction metadata if not present in the row
            extraction_meta = row.get("extraction_metadata", {"extraction_method": "Unknown", "confidence": 1.0})
            if not isinstance(extraction_meta, dict): # Ensure it's a dict
                 extraction_meta = {"extraction_method": "Unknown (invalid format)", "confidence": 1.0}


            self.add_triple_with_metadata(
                creative_work_uri,
                RDF.type,
                self.namespaces["schema"]["CreativeWork"], # Licenses are a form of creative work
                extraction_meta 
            )
            
            # Adding schema:license as a more specific type as well, if appropriate
            # For now, focusing on CreativeWork as per SPDX mapping guidance often points there or to DigitalDocument
            # self.add_triple_with_metadata(
            # creative_work_uri,
            # RDF.type,
            # self.namespaces["schema"]["License"], # If a schema:License type is defined and preferred
            # extraction_meta

            if "Name" in row and row["Name"] is not None and not pd.isna(row["Name"]):
                self.add_triple_with_metadata(
                    creative_work_uri,
                    self.namespaces["schema"]["name"],
                    Literal(row["Name"], datatype=XSD.string),
                    extraction_meta
                )
            
            if "Identifier" in row and row["Identifier"] is not None and not pd.isna(row["Identifier"]):
                self.add_triple_with_metadata(
                    creative_work_uri,
                    self.namespaces["schema"]["identifier"],
                    Literal(row["Identifier"], datatype=XSD.string),
                    extraction_meta
                )
            
            if "URL" in row and row["URL"] is not None and not pd.isna(row["URL"]):
                self.add_triple_with_metadata(
                    creative_work_uri,
                    self.namespaces["schema"]["url"],
                    URIRef(row["URL"]), # Assuming URL is a valid URI
                    extraction_meta
                )
            
            if "Text" in row and row["Text"] is not None and not pd.isna(row["Text"]):
                # Use specific metadata for 'Text' as per user's original function
                self.add_triple_with_metadata(
                    creative_work_uri,
                    self.namespaces["schema"]["description"], # Or schema:text if more appropriate
                    Literal(row["Text"], datatype=XSD.string),
                    extraction_meta 
                )
            
            # Example for OSI Approved (custom property or map to existing schema.org property)
            # if "OSI Approved" in row and row["OSI Approved"] is not None and not pd.isna(row["OSI Approved"]):
            #     # This might need a custom predicate, e.g., self.namespaces["spdx"]["isOsiApproved"]
            #     # For now, let's use schema:additionalProperty or a simple schema:value
            #     osi_approved_value = Literal(row["OSI Approved"], datatype=XSD.boolean) # Assuming it's boolean
                
            #     # Create a blank node for the PropertyValue
            #     pv_osi_hash = self.generate_entity_hash(platform, "LicenseProperty", f"{entity_id}_osiApproved")
            #     property_value_uri_osi = self.base_namespace[pv_osi_hash]

            #     self.add_triple_with_metadata(property_value_uri_osi, RDF.type, self.namespaces["schema"]["PropertyValue"], extraction_meta)
            #     self.add_triple_with_metadata(property_value_uri_osi, self.namespaces["schema"]["propertyID"], Literal("isOsiApproved", datatype=XSD.string), extraction_meta)
            #     self.add_triple_with_metadata(property_value_uri_osi, self.namespaces["schema"]["value"], osi_approved_value, extraction_meta)
            #     self.add_triple_with_metadata(creative_work_uri, self.namespaces["schema"]["additionalProperty"], property_value_uri_osi, extraction_meta)

            # if "Deprecated" in row and row["Deprecated"] is not None and not pd.isna(row["Deprecated"]):
            #      # Similar to OSI Approved, map to schema:additionalProperty or a more specific term if available
            #     deprecated_value = Literal(row["Deprecated"], datatype=XSD.boolean) # Assuming boolean, adjust if not

            #     pv_deprecated_hash = self.generate_entity_hash(platform, "LicenseProperty", f"{entity_id}_deprecated")
            #     property_value_uri_deprecated = self.base_namespace[pv_deprecated_hash]
                
            #     self.add_triple_with_metadata(property_value_uri_deprecated, RDF.type, self.namespaces["schema"]["PropertyValue"], extraction_meta)
            #     self.add_triple_with_metadata(property_value_uri_deprecated, self.namespaces["schema"]["propertyID"], Literal("isDeprecated", datatype=XSD.string), extraction_meta)
            #     self.add_triple_with_metadata(property_value_uri_deprecated, self.namespaces["schema"]["value"], deprecated_value, extraction_meta)
            #     self.add_triple_with_metadata(creative_work_uri, self.namespaces["schema"]["additionalProperty"], property_value_uri_deprecated, extraction_meta)

        return self.graph, self.metadata_graph
