import rdflib
import json
import os
import hashlib
import pprint
import logging
import pandas as pd
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, XSD, FOAF
from rdflib.util import from_n3
from pandas import Timestamp
from tqdm import tqdm
from datetime import datetime
from typing import Callable, List, Dict, Set, Tuple, Optional
import pprint

from mlentory_load.dbHandler import SQLHandler, RDFHandler, IndexHandler
from mlentory_load.core.Entities import HFModel


class GraphHandler:
    """
    Handler for graph operations and version control across databases.

    This class manages:
    - Graph construction and updates
    - Version control of triples
    - Synchronization between databases
    - Metadata tracking

    Attributes:
        SQLHandler (SQLHandler): Handler for SQL operations
        RDFHandler (RDFHandler): Handler for RDF operations
        IndexHandler (IndexHandler): Handler for search index
        kg_files_directory (str): Directory for knowledge graph files
        platform (str): Platform identifier (e.g., "hugging_face")
        graph_identifier (str): URI for main graph
        deprecated_graph_identifier (str): URI for deprecated triples
    """

    def __init__(
        self,
        SQLHandler: SQLHandler,
        RDFHandler: RDFHandler,
        IndexHandler: IndexHandler,
        kg_files_directory: str = "./../kg_files",
        platform: str = "hugging_face",
        graph_identifier: str = "https://example.com/data_1",
        deprecated_graph_identifier: str = "https://example.com/data_2",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize GraphHandler with handlers and configuration.

        Args:
            SQLHandler (SQLHandler): SQL database handler
            RDFHandler (RDFHandler): RDF store handler
            IndexHandler (IndexHandler): Search index handler
            kg_files_directory (str): Path to graph files
            platform (str): Platform identifier
            graph_identifier (str): Main graph URI
            deprecated_graph_identifier (str): Deprecated graph
            logger (Optional[logging.Logger]): Logger instance
        """
        self.SQLHandler = SQLHandler
        self.RDFHandler = RDFHandler
        self.IndexHandler = IndexHandler
        self.kg_files_directory = kg_files_directory
        self.platform = platform
        self.graph_identifier = graph_identifier
        self.deprecated_graph_identifier = deprecated_graph_identifier
        self.logger = logger if logger else logging.getLogger(__name__)

        self.new_triplets = []
        self.old_triplets = []

    def update_current_graph(self):
        """
        Update the current graph with new triplets.

        This method:
        1. Creates new triplets graph
        2. Creates deprecated triplets graph
        3. Updates the RDF store
        """
        self.logger.info("Updating current graph...")
        new_triplets_graph = rdflib.Graph(
            identifier=self.graph_identifier
        )
        new_triplets_graph.bind("fair4ml", URIRef("https://fair4ml.com/"))
        new_triplets_graph.bind("codemeta", URIRef("https://codemeta.com/"))
        new_triplets_graph.bind("schema", URIRef("https://schema.org/"))
        new_triplets_graph.bind("mlentory", URIRef("https://mlentory.com/"))
        new_triplets_graph.bind("prov", URIRef("https://www.w3.org/ns/prov#"))

        old_triplets_graph = rdflib.Graph(
            identifier=self.deprecated_graph_identifier
        )
        old_triplets_graph.bind("fair4ml", URIRef("https://fair4ml.com/"))
        old_triplets_graph.bind("codemeta", URIRef("https://codemeta.com/"))
        old_triplets_graph.bind("schema", URIRef("https://schema.org/"))
        old_triplets_graph.bind("mlentory", URIRef("https://mlentory.com/"))
        old_triplets_graph.bind("prov", URIRef("https://www.w3.org/ns/prov#"))

        self.logger.info("Adding new triplets...")
        for new_triplet in tqdm(self.new_triplets, desc="Processing new triplets"):
            new_triplets_graph.add(new_triplet)

        self.logger.info("Processing deprecated triplets...")
        for old_triplet in tqdm(
            self.old_triplets, desc="Processing deprecated triplets"
        ):
            old_triplets_graph.add(old_triplet)

        current_date = datetime.now().strftime("%Y-%m-%d")
        path_new_triplets_graph = os.path.join(
            self.kg_files_directory, f"new_triplets_graph_{current_date}.nt"
        )
        self.logger.info(f"Serializing new triplets to {path_new_triplets_graph}")
        new_triplets_graph.serialize(
            destination=path_new_triplets_graph, format="nt"
        )

        self.logger.info("Loading new triplets into RDF store...")
        self.RDFHandler.load_graph(
            ttl_file_path=path_new_triplets_graph,
            graph_identifier=self.graph_identifier,
        )

        if len(old_triplets_graph) > 0:
            path_old_triplets_graph = os.path.join(
                self.kg_files_directory, f"old_triplets_graph_{current_date}.nt"
            )
            self.logger.info(
                f"Serializing deprecated triplets to {path_old_triplets_graph}"
            )
            old_triplets_graph.serialize(
                destination=path_old_triplets_graph, format="nt"
            )

            self.logger.info("Updating RDF store with deprecated triplets...")
            self.RDFHandler.delete_graph(
                ttl_file_path=path_old_triplets_graph,
                graph_identifier=self.graph_identifier,
                deprecated_graph_identifier=self.deprecated_graph_identifier,
            )


    def process_triplet(self, subject, predicate, object, extraction_info):
        """
        Process a triplet by managing its storage and version control in the database.

        Args:
            subject: The subject of the triplet
            predicate: The predicate of the triplet
            object: The object of the triplet
            extraction_info (dict): Dictionary containing extraction metadata

        Raises:
            SQLError: If there's an error executing any SQL query
            ValueError: If extraction_info lacks required fields
        """
        triplet_id, is_new_triplet = self._get_or_create_triplet_id(
            subject, predicate, object
        )

        extraction_info_id = self._get_or_create_extraction_info_id(extraction_info)

        extraction_time = datetime.strptime(
            extraction_info["extraction_time"], "%Y-%m-%d_%H-%M-%S"
        )

        self._manage_version_range(triplet_id, extraction_info_id, extraction_time)

        if is_new_triplet:
            self.new_triplets.append((subject, predicate, object))

    def _get_or_create_triplet_id(
        self, subject: URIRef, predicate: URIRef, object: URIRef
    ) -> Tuple[int, bool]:
        """
        Get or create a triplet ID in the database.

        Args:
            subject (URIRef): The subject of the triplet
            predicate (URIRef): The predicate of the triplet
            object (URIRef): The object of the triplet

        Returns:
            tuple[int, bool]: A tuple containing the triplet ID and a boolean indicating if it's new

        Raises:
            SQLError: If there's an error executing the SQL query
        """
        subject_json = str(subject.n3())
        predicate_json = str(predicate.n3())
        object_json = str(object.n3())

        # Create a combined hash of all three components of the triplet
        triplet_hash = hashlib.md5(
            (subject_json + predicate_json + object_json).encode()
        ).hexdigest()

        # First try to find the triplet by its hash
        triplet_id_df = self.SQLHandler.query(
            """SELECT id FROM "Triplet" WHERE triplet_hash = %s""", (triplet_hash,)
        )

        if triplet_id_df.empty:
            # If not found by hash, insert new triplet
            triplet_id = self.SQLHandler.insert(
                "Triplet",
                {
                    "subject": subject_json,
                    "predicate": predicate_json,
                    "object": object_json,
                    "triplet_hash": triplet_hash,
                },
            )
            return triplet_id, True
        return triplet_id_df.iloc[0]["id"], False

    def _get_or_create_extraction_info_id(self, extraction_info: dict) -> int:
        """
        Get or create an extraction info ID in the database.

        Args:
            extraction_info (dict): Dictionary containing extraction method and confidence

        Returns:
            int: The extraction info ID

        Raises:
            SQLError: If there's an error executing the SQL query
        """
        extraction_info_id_df = self.SQLHandler.query(
            """SELECT id FROM "Triplet_Extraction_Info" WHERE 
               method_description = %s 
               AND extraction_confidence = %s""",
            (extraction_info["extraction_method"], extraction_info["confidence"]),
        )

        if extraction_info_id_df.empty:
            return self.SQLHandler.insert(
                "Triplet_Extraction_Info",
                {
                    "method_description": extraction_info["extraction_method"],
                    "extraction_confidence": extraction_info["confidence"],
                },
            )
        return extraction_info_id_df.iloc[0]["id"]

    def _manage_version_range(
        self, triplet_id: int, extraction_info_id: int, extraction_time: datetime
    ) -> None:
        """
        Manage version range for a triplet, either creating a new one or updating existing.

        Args:
            triplet_id (int): The ID of the triplet
            extraction_info_id (int): The ID of the extraction info
            extraction_time (datetime): The extraction timestamp

        Raises:
            SQLError: If there's an error executing the SQL query
        """
        version_range_df = self.SQLHandler.query(
            """SELECT vr.id, vr.use_start, vr.use_end FROM "Version_Range" vr WHERE
               triplet_id = %s
               AND extraction_info_id = %s
               AND deprecated = %s""",
            (str(triplet_id), str(extraction_info_id), False),
        )

        if version_range_df.empty:
            self.SQLHandler.insert(
                "Version_Range",
                {
                    "triplet_id": str(triplet_id),
                    "extraction_info_id": str(extraction_info_id),
                    "use_start": extraction_time,
                    "use_end": extraction_time,
                    "deprecated": False,
                },
            )
        else:
            version_range_id = version_range_df.iloc[0]["id"]
            # Use execute_sql instead of query for UPDATE statements
            self.SQLHandler.execute_sql(
                """UPDATE "Version_Range" SET use_end = %s WHERE id = %s""",
                (extraction_time, version_range_id),
            )

    def deprecate_old_triplets(self, model_uri):

        model_uri_json = str(model_uri.n3())

        old_triplets_df = self.SQLHandler.query(
            """SELECT t.id,t.subject,t.predicate,t.object, vr.deprecated, vr.use_end
               FROM "Triplet" t 
               JOIN "Version_Range" vr 
               ON t.id = vr.triplet_id
               WHERE t.subject = %s
               AND vr.deprecated = %s
               AND vr.use_end < %s""",
            (model_uri_json, False, self.curr_update_date),
        )

        if not old_triplets_df.empty:
            for index, old_triplet in old_triplets_df.iterrows():
                self.old_triplets.append(
                    (
                        self.n3_to_term(old_triplet["subject"]),
                        self.n3_to_term(old_triplet["predicate"]),
                        self.n3_to_term(old_triplet["object"]),
                    )
                )

        # Use parameterized query for update
        update_query = """
            UPDATE "Version_Range" vr
            SET deprecated = %s, use_end = %s
            FROM "Triplet" t
            JOIN "Version_Range" on t.id = triplet_id
            WHERE t.subject = %s
            AND vr.use_end < %s
            AND vr.deprecated = %s
            AND vr.triplet_id = t.id
        """
        self.SQLHandler.execute_sql(
            update_query,
            (True, self.curr_update_date, model_uri_json, self.curr_update_date, False),
        )

    def update_triplet_ranges_for_unchanged_models(self, curr_date: str) -> None:
        """
        Update all triplet ranges that were not modified in the last update
        to have the same end date as the current date.

        Args:
            curr_date (str): Current date in format 'YYYY-MM-DD_HH-MM-SS'
        """
        # Convert the date string from 'YYYY-MM-DD_HH-MM-SS' to PostgreSQL timestamp format
        formatted_date = datetime.strptime(curr_date, "%Y-%m-%d_%H-%M-%S").strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        update_query = """
            UPDATE "Version_Range"
            SET use_end = %s
            WHERE use_end != %s
            AND deprecated = %s
        """

        self.SQLHandler.execute_sql(
            update_query, (formatted_date, formatted_date, False)
        )

    def get_current_graph(self) -> Graph:
        """
        Retrieve current version of the RDF graph.

        Returns:
            Graph: Current active graph with all valid triplets
        """
        current_graph = rdflib.Graph(identifier=self.graph_identifier)
        current_graph.bind("fair4ml", URIRef("https://fair4ml.com/"))
        current_graph.bind("codemeta", URIRef("https://codemeta.com/"))
        current_graph.bind("schema", URIRef("https://schema.org/"))
        current_graph.bind("mlentory", URIRef("https://mlentory.com/"))
        current_graph.bind("prov", URIRef("https://www.w3.org/ns/prov#"))

        query = f"""
        CONSTRUCT {{ ?s ?p ?o }}
        WHERE {{
            GRAPH <{self.graph_identifier}> {{
                ?s ?p ?o
            }}
        }}
        """

        result_graph = self.RDFHandler.query(self.RDFHandler.sparql_endpoint, query)
        current_graph += result_graph

        return current_graph

    def n3_to_term(self, n3):
        return from_n3(n3.encode("unicode_escape").decode("unicode_escape"))

    def text_to_uri_term(self, text):
        return URIRef(f"mlentory:/{self.platform}/{text}")

    def _batch_get_or_create_triplet_ids(
        self, triplets: List[Tuple[URIRef, URIRef, URIRef]]
    ) -> Tuple[List[int], List[bool]]:
        """
        Get or create multiple triplet IDs in the database using batch processing.

        Args:
            triplets (List[Tuple[URIRef, URIRef, URIRef]]): List of triplets to process

        Returns:
            Tuple[List[int], List[bool]]: Lists of triplet IDs and their new status

        Raises:
            SQLError: If there's an error executing the SQL queries
        """
        if not triplets:
            return [], []

        # Convert triplets to JSON format and calculate hashes
        triplet_data = [
            {
                "subject": str(subject.n3()),
                "predicate": str(predicate.n3()),
                "object": str(object_.n3()),
                "triplet_hash": hashlib.md5(
                    (
                        str(subject.n3()) + str(predicate.n3()) + str(object_.n3())
                    ).encode()
                ).hexdigest(),
            }
            for subject, predicate, object_ in triplets
        ]

        if not triplet_data:
            return [], []

        hashes = [d["triplet_hash"] for d in triplet_data]

        # Use WHERE ... = ANY(...) for a more efficient query
        query = """
            SELECT id, triplet_hash FROM "Triplet"
            WHERE triplet_hash = ANY(%s)
        """

        existing_triplets_df = self.SQLHandler.query(query, (hashes,))

        hash_to_id_map = {
            row["triplet_hash"]: row["id"] for _, row in existing_triplets_df.iterrows()
        }

        # Initialize results
        triplet_ids = [-1] * len(triplets)
        is_new = [True] * len(triplets)

        new_triplets_data_map = {}

        # Process existing triplets
        for i, data in enumerate(triplet_data):
            triplet_hash = data["triplet_hash"]
            if triplet_hash in hash_to_id_map:
                triplet_ids[i] = hash_to_id_map[triplet_hash]
                is_new[i] = False
            else:
                new_triplets_data_map[triplet_hash] = (
                    data["subject"],
                    data["predicate"],
                    data["object"],
                    triplet_hash,
                )

        if new_triplets_data_map:
            new_triplets_data = list(new_triplets_data_map.values())
            new_ids = self.SQLHandler.batch_insert(
                "Triplet",
                ["subject", "predicate", "object", "triplet_hash"],
                new_triplets_data,
                batch_size=len(new_triplets_data),
            )

            # Create a map of hash to new_id for the inserted triplets
            new_hash_to_id_map = {
                data[3]: new_id for data, new_id in zip(new_triplets_data, new_ids)
            }

            # Update triplet_ids with new IDs
            for i, data in enumerate(triplet_data):
                triplet_hash = data["triplet_hash"]
                if is_new[i]:
                    triplet_ids[i] = new_hash_to_id_map[triplet_hash]

        return triplet_ids, is_new

    def _batch_get_or_create_extraction_info_ids(
        self, extraction_infos: List[Dict]
    ) -> List[int]:
        """
        Get or create multiple extraction info IDs in the database using batch processing.

        Args:
            extraction_infos (List[Dict]): List of extraction info dictionaries

        Returns:
            List[int]: List of extraction info IDs

        Raises:
            SQLError: If there's an error executing the SQL queries
        """
        if not extraction_infos:
            return []

        # Separate infos into parallel lists for unnesting
        info_tuples = []
        extraction_methods = []
        extraction_info_hashes = set()

        extraction_confidences = []

        for info in extraction_infos:
            method = info["extraction_method"]
            confidence = round(info["confidence"], 5)
            extraction_info_hash = hashlib.md5(
                (str(method) + str(confidence)).encode()
            ).hexdigest()
            extraction_methods.append(method)
            extraction_confidences.append(confidence)
            extraction_info_hashes.add(extraction_info_hash)
            info_tuples.append((method, confidence, extraction_info_hash))

        # Use unnest on parallel arrays for a more robust query
        query = """
            SELECT t.id, t.extraction_info_hash
            FROM "Triplet_Extraction_Info" t
            WHERE t.extraction_info_hash = ANY(%s)
        """
        existing_infos_df = self.SQLHandler.query(
            query, (list(extraction_info_hashes),)
        )

        # Create a map for existing infos
        search_results_map_hash_to_id = {
            row["extraction_info_hash"]: row["id"]
            for _, row in existing_infos_df.iterrows()
        }

        # Initialize results
        info_ids = [-1] * len(extraction_infos)
        new_entries_map_hash_to_id = {}

        # Process results
        for i, info_tuple in enumerate(info_tuples):
            # Check if the hash (info_tuple[2]) is in the map of existing infos
            if info_tuple[2] in search_results_map_hash_to_id:
                info_ids[i] = search_results_map_hash_to_id[info_tuple[2]]
            else:
                # Use a map to collect unique new infos
                if info_tuple[2] not in new_entries_map_hash_to_id:
                    new_entries_map_hash_to_id[info_tuple[2]] = info_tuple

        if new_entries_map_hash_to_id:
            new_entries_to_insert = list(new_entries_map_hash_to_id.values())
            new_ids = self.SQLHandler.batch_insert(
                "Triplet_Extraction_Info",
                ["method_description", "extraction_confidence", "extraction_info_hash"],
                new_entries_to_insert,
                batch_size=len(new_entries_to_insert),
            )

            # Create a map for newly inserted infos
            new_info_to_id_map = {
                data[2]: new_id for data, new_id in zip(new_entries_to_insert, new_ids)
            }

            # Update info_ids with new IDs
            for i, info_tuple in enumerate(info_tuples):
                if info_ids[i] == -1:
                    info_ids[i] = new_info_to_id_map[info_tuple[2]]

        return info_ids

    def _batch_manage_version_ranges(
        self,
        triplet_ids: List[int],
        extraction_info_ids: List[int],
        extraction_times: List[datetime],
    ) -> None:
        """
        Manage version ranges for multiple triplets using batch processing.

        Args:
            triplet_ids (List[int]): List of triplet IDs
            extraction_info_ids (List[int]): List of extraction info IDs
            extraction_times (List[datetime]): List of extraction timestamps

        Raises:
            SQLError: If there's an error executing the SQL queries
        """
        if not triplet_ids:
            return

        # Use unnest on parallel arrays for a more robust query
        query = """
            SELECT t.id, u.t_id, u.e_id
            FROM "Version_Range" t
            JOIN (
                SELECT tid.t_id, eid.e_id
                FROM unnest(%s) WITH ORDINALITY AS tid(t_id, rn)
                JOIN unnest(%s) WITH ORDINALITY AS eid(e_id, rn)
                ON tid.rn = eid.rn
            ) AS u ON t.triplet_id = u.t_id AND t.extraction_info_id = u.e_id
            WHERE t.deprecated = %s
        """
        existing_ranges_df = self.SQLHandler.query(
            query, (triplet_ids, extraction_info_ids, False)
        )

        # Create a map for existing ranges
        range_to_id_map = {
            (row["t_id"], row["e_id"]): row["id"]
            for _, row in existing_ranges_df.iterrows()
        }

        # Prepare updates for existing ranges
        updates = []
        new_ranges_data = []

        for i, (t_id, e_id, time) in enumerate(
            zip(triplet_ids, extraction_info_ids, extraction_times)
        ):
            key = (t_id, e_id)
            if key in range_to_id_map:
                updates.append((time, range_to_id_map[key]))
            else:
                new_ranges_data.append((str(t_id), str(e_id), time, time, False))

        if updates:
            # Update each row individually with parameterized queries
            for time, range_id in updates:
                self.SQLHandler.execute_sql(
                    """UPDATE "Version_Range" SET use_end = %s WHERE id = %s""",
                    (time, range_id),
                )

        if new_ranges_data:
            # Deduplicate before inserting
            unique_new_ranges = sorted(list(set(new_ranges_data)))
            self.SQLHandler.batch_insert(
                "Version_Range",
                [
                    "triplet_id",
                    "extraction_info_id",
                    "use_start",
                    "use_end",
                    "deprecated",
                ],
                unique_new_ranges,
                batch_size=len(unique_new_ranges),
            )

    def print_query_stats(self):
        """Prints the query statistics."""
        print("--- SQL Query Statistics ---")
        stats = self.SQLHandler.query_stats["queries"]
        if not stats:
            print("No queries were tracked.")
            return

        for query, data in stats.items():
            count = data["count"]
            total_time = data["total_time"]
            avg_time = total_time / count if count > 0 else 0
            print(f"Query: {query}")
            print(f"  Count: {count}")
            print(f"  Total Time: {total_time:.4f} seconds")
            print(f"  Average Time: {avg_time:.4f} seconds")
            print("-" * 20)

    def process_triplet_batch(
        self, triplets_data: List[Tuple[URIRef, URIRef, URIRef, Dict]]
    ) -> None:
        """
        Process multiple triplets in batch for efficient database operations.

        Args:
            triplets_data (List[Tuple[URIRef, URIRef, URIRef, Dict]]): List of tuples containing
                (subject, predicate, object, extraction_info)

        Raises:
            SQLError: If there's an error executing any SQL query
            ValueError: If extraction_info lacks required fields
        """
        if not triplets_data:
            return

        # Unzip the triplets data
        subjects, predicates, objects, extraction_infos = zip(*triplets_data)
        triplets = list(zip(subjects, predicates, objects))

        # Process triplets in batch
        triplet_ids, is_new = self._batch_get_or_create_triplet_ids(triplets)

        # Process extraction infos in batch
        extraction_info_ids = self._batch_get_or_create_extraction_info_ids(
            extraction_infos
        )

        # Process extraction times
        extraction_times = [
            datetime.strptime(info["extraction_time"], "%Y-%m-%d_%H-%M-%S")
            for info in extraction_infos
        ]

        # Manage version ranges in batch
        self._batch_manage_version_ranges(
            triplet_ids, extraction_info_ids, extraction_times
        )

        # Add new triplets to the list
        for i, (is_new_triplet, triplet) in enumerate(zip(is_new, triplets)):
            if is_new_triplet:
                self.new_triplets.append(triplet)
        
        self.print_query_stats()
