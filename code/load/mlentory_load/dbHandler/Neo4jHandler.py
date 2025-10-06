import os
from typing import Optional, Any, Dict, List, Union
from pathlib import Path
import requests

from rdflib import Graph
from rdflib_neo4j import (
    Neo4jStoreConfig,
    Neo4jStore,
    HANDLE_VOCAB_URI_STRATEGY,
)

try:
    # Optional dependency for administrative Cypher (constraints, reset)
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None


class Neo4jHandler:
    """
    Handler for RDF operations backed by Neo4j using rdflib-neo4j.

    This provides a minimal interface compatible with the existing RDFHandler usage
    in the loading flow (load_graph, delete_graph, reset_db, query-like behavior).

    Args:
        uri (str): Neo4j Bolt URI (e.g., bolt://neo4j_db:7687)
        user (str): Neo4j username
        password (str): Neo4j password
        database (str): Neo4j database name (default: "neo4j")
        kg_files_directory (str): Directory for temporary KG files
        batching (bool): Enable rdflib-neo4j batching
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        kg_files_directory: Optional[str] = None,
        batching: bool = True,
        http_scheme: str = "http",
        http_port: int = 7474,
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.kg_files_directory = kg_files_directory or os.getcwd()
        self.is_neo4j = True
        self.sparql_endpoint = None  # for compatibility only
        self.http_scheme = http_scheme
        self.http_port = http_port

        auth_data = {
            "uri": self.uri,
            "database": self.database,
            "user": self.user,
            "pwd": self.password,
        }

        self._config = Neo4jStoreConfig(
            auth_data=auth_data,
            handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.IGNORE,
            batching=batching,
        )
        self._store = Neo4jStore(config=self._config)

        # Ensure uniqueness constraint exists (idempotent)
        self._ensure_uniqueness_constraint()
        # Also ensure n10s-named constraint if plugin expects it
        self._ensure_rdf_constraint()

    def _ensure_uniqueness_constraint(self) -> None:
        if GraphDatabase is None:
            return
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with driver.session(database=self.database) as session:
                session.run(
                    """
                    CREATE CONSTRAINT resource_uri_unique
                    IF NOT EXISTS
                    FOR (r:Resource) REQUIRE r.uri IS UNIQUE
                    """
                )
        except Exception:
            # Do not fail pipeline if constraint creation fails; log upstream if needed
            pass

    def get_graph(self, identifier: Optional[str] = None) -> Graph:
        """
        Return an rdflib Graph backed by the Neo4j store.
        """
        return Graph(store=self._store, identifier=identifier)

    def reset_db(self) -> None:
        """
        Remove all data from the Neo4j database (dangerous).
        """
        if GraphDatabase is None:
            # Fallback: best-effort via Graph.remove for all triples
            g = self.get_graph()
            to_remove = list(g.triples((None, None, None)))
            for t in to_remove:
                g.remove(t)
            return

        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
        except Exception:
            # Swallow to avoid stopping the pipeline; caller may clean state differently
            pass

    def load_graph(self, ttl_file_path: str, graph_identifier: str) -> None:
        """
        Load triples from an NT/Turtle file into Neo4j.
        """
        g = self.get_graph()
        fmt = "nt" if ttl_file_path.endswith(".nt") else "turtle"
        g.parse(ttl_file_path, format=fmt)
        try:
            g.close(True)
        except Exception:
            pass

    def delete_graph(
        self,
        ttl_file_path: str,
        graph_identifier: str,
        deprecated_graph_identifier: str,
    ) -> None:
        """
        Delete triples present in the given file from Neo4j.
        """
        remove_graph = Graph()
        fmt = "nt" if ttl_file_path.endswith(".nt") else "turtle"
        remove_graph.parse(ttl_file_path, format=fmt)

        target_graph = self.get_graph()
        # Batch-style removal
        batch_size = 50000
        batch = []
        for s, p, o in remove_graph.triples((None, None, None)):
            batch.append((s, p, o))
            if len(batch) >= batch_size:
                for t in batch:
                    target_graph.remove(t)
                batch = []
        if batch:
            for t in batch:
                target_graph.remove(t)

    def query(self, sparql_endpoint: str, query: str) -> Graph:
        """
        Compatibility shim: return the current Neo4j-backed Graph.
        """
        return self.get_graph()

    # ----------------------
    # Optional n10s helpers
    # ----------------------

    def _driver(self):
        if GraphDatabase is None:
            raise RuntimeError("neo4j Python driver not available")
        return GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def _execute_cypher(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query via Bolt and return list of dicts."""
        if params is None:
            params = {}
        driver = self._driver()
        with driver.session(database=self.database) as session:
            result = session.run(query, **params)
            try:
                return [r.data() for r in result]
            except Exception:
                # Some procedures return summary only
                return []

    def _build_http_url(self) -> str:
        """Build base HTTP URL from Bolt URI and configured http scheme/port."""
        # Expecting bolt://host:port
        host = self.uri.replace("bolt://", "").split(":")[0]
        return f"{self.http_scheme}://{host}:{self.http_port}"

    def describe_resource_neosemantics(
        self,
        resource_id: str,
        file_path: Optional[str] = None,
        format: str = "Turtle",
    ) -> Union[str, Dict[str, Any]]:
        """
        Describe a specific resource using the Neosemantics HTTP endpoint.

        Args:
            resource_id (str): ID or URI of the resource to describe
            file_path (Optional[str]): If provided, path to write the RDF response
            format (str): Response format, one of 'Turtle', 'RDF/XML', 'JSON-LD'

        Returns:
            Union[str, Dict[str, Any]]: RDF string when file_path is None, else export metadata
        """
        base_url = self._build_http_url()
        endpoint = f"/rdf/{self.database}/describe/{resource_id}"
        params: Dict[str, str] = {}
        if format and format != "Turtle":
            params["format"] = format

        try:
            response = requests.get(
                f"{base_url}{endpoint}",
                auth=(self.user, self.password),
                params=params,
                timeout=30,
            )
            if response.status_code == 404:
                raise RuntimeError(f"Resource with ID '{resource_id}' not found")
            if response.status_code != 200:
                raise RuntimeError(
                    f"Neosemantics endpoint returned {response.status_code}: {response.text}"
                )

            rdf_content = response.text
            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(rdf_content)
                return {
                    "file_path": file_path,
                    "resource_id": resource_id,
                    "format": format,
                    "endpoint": endpoint,
                    "success": True,
                }
            return rdf_content
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to connect to Neosemantics endpoint: {e}")

    def clear_graph(self) -> None:
        """Delete all nodes and relationships in the graph."""
        self.reset_db()

    def check_neosemantics(self) -> Dict[str, Any]:
        """
        Verify Neosemantics availability via CALL n10s.version().

        Returns:
            Dict[str, Any]: availability and optional version/error
        """
        try:
            data = self._execute_cypher("CALL n10s.version()")
            return {"available": True, "version": data[0] if data else None}
        except Exception as e:
            return {"available": False, "error": str(e)}

    def init_neosemantics(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Initialize or override n10s graph configuration.

        Args:
            config (Optional[Dict[str, Any]]): Optional configuration map for n10s

        Returns:
            Dict[str, Any]: Operation result
        """
        # Ensure a canonical uniqueness constraint for n10s
        self._ensure_rdf_constraint()
        if config:
            data = self._execute_cypher("CALL n10s.graphconfig.init($cfg)", {"cfg": config})
        else:
            data = self._execute_cypher("CALL n10s.graphconfig.init()")
        return {"ok": True, "result": data}

    def get_rdf_graph(
        self,
        identifier: Optional[str] = None,
        shorten_prefixes: bool = True,
        custom_prefixes: Optional[Dict[str, Union[str, Any]]] = None,
    ) -> Graph:
        """
        Create an rdflib Graph bound to Neo4j with optional prefix shortening.

        Args:
            identifier (Optional[str]): Graph identifier
            shorten_prefixes (bool): Whether to use SHORTEN strategy
            custom_prefixes (Optional[Dict[str, Union[str, Any]]]): Prefix map to configure store

        Returns:
            Graph: rdflib Graph backed by Neo4j store
        """
        auth_data = {
            "uri": self.uri,
            "database": self.database,
            "user": self.user,
            "pwd": self.password,
        }
        strategy = (
            HANDLE_VOCAB_URI_STRATEGY.SHORTEN if shorten_prefixes else HANDLE_VOCAB_URI_STRATEGY.IGNORE
        )
        store_config = Neo4jStoreConfig(
            auth_data=auth_data,
            handle_vocab_uri_strategy=strategy,
            batching=True,
            custom_prefixes=custom_prefixes or {},
        )
        return Graph(store=Neo4jStore(config=store_config), identifier=identifier)

    def load_rdf_into_graph(self, file_path: Union[str, Path], format: Optional[str] = None) -> Dict[str, int]:
        """
        Load RDF data via rdflib-neo4j and return simple graph stats.

        Args:
            file_path (Union[str, Path]): Path to RDF file
            format (Optional[str]): rdflib format; auto-guessed if None

        Returns:
            Dict[str, int]: node_count and relationship_count after import
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"RDF file not found: {file_path}")

        g = self.get_rdf_graph()
        try:
            if format is None:
                fmt = p.suffix.lstrip(".")
                fmt = "turtle" if fmt == "ttl" else ("json-ld" if fmt == "jsonld" else fmt)
            else:
                fmt = format
            g.parse(str(p), format=fmt)
            try:
                g.close(True)
            except Exception:
                pass

            # Simple counts
            nodes = self._execute_cypher("MATCH (n) RETURN count(n) AS node_count")
            rels = self._execute_cypher("MATCH ()-[r]->() RETURN count(r) AS relationship_count")
            return {
                "nodes_created": int(nodes[0]["node_count"]) if nodes else 0,
                "relationships_created": int(rels[0]["relationship_count"]) if rels else 0,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load RDF file: {e}")

    def load_rdf(self, file_path: Union[str, Path], format: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load RDF via n10s.rdf.import.fetch if Neosemantics is available.

        Args:
            file_path (Union[str, Path]): Path or URL to the RDF content
            format (Optional[str]): RDF format string (e.g., 'Turtle','RDF/XML','JSON-LD')

        Returns:
            List[Dict[str, Any]]: Raw procedure output rows
        """
        fmt = (format or "turtle").strip()
        results = self._execute_cypher(
            "CALL n10s.rdf.import.fetch($file_path, $format)",
            {"file_path": str(file_path).strip(), "format": fmt},
        )
        return results

    def list_prefixes(self) -> List[Dict[str, str]]:
        """List namespace prefixes configured in Neosemantics."""
        rows = self._execute_cypher("CALL n10s.nsprefixes.list()")
        prefixes: List[Dict[str, str]] = []
        for r in rows:
            prefixes.append({"prefix": r.get("prefix"), "namespace": r.get("namespace")})
        return prefixes

    def add_prefix(self, prefix: str, namespace: str) -> Dict[str, str]:
        """Add a namespace prefix mapping in Neosemantics."""
        if not prefix or not prefix.strip():
            raise ValueError("Prefix cannot be empty")
        if not namespace or not namespace.strip():
            raise ValueError("Namespace cannot be empty")
        rows = self._execute_cypher(
            "CALL n10s.nsprefixes.add($prefix, $namespace)",
            {"prefix": prefix.strip(), "namespace": namespace.strip()},
        )
        if rows:
            return {"prefix": rows[0].get("prefix"), "namespace": rows[0].get("namespace")}
        raise RuntimeError("Failed to add prefix - no result returned")

    def add_prefixes_from_text(self, text: str) -> List[Dict[str, str]]:
        """Add multiple prefixes parsed from RDF header text using Neosemantics."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        rows = self._execute_cypher(
            "CALL n10s.nsprefixes.addFromText($text) YIELD prefix, namespace RETURN prefix, namespace",
            {"text": text.strip()},
        )
        prefixes: List[Dict[str, str]] = []
        for r in rows:
            prefixes.append({"prefix": r.get("prefix"), "namespace": r.get("namespace")})
        if not prefixes:
            raise ValueError("No valid namespace declarations found in the provided text")
        return prefixes

    def _ensure_rdf_constraint(self) -> None:
        """Ensure a uniqueness constraint exists for Resource.uri via n10s/Neo4j."""
        try:
            self._execute_cypher(
                """
                CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS
                FOR (r:Resource) REQUIRE r.uri IS UNIQUE
                """
            )
        except Exception:
            # Fallback already handled by _ensure_uniqueness_constraint
            pass
    
    def export_graph_neosemantics(
        self,
        file_path: Optional[str] = None,
        format: str = "Turtle",
        cypher_query: Optional[str] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Export the entire graph using Neosemantics HTTP endpoints.

        This method uses the Neosemantics /rdf/<dbname>/cypher endpoint to export
        the entire graph as RDF. It provides robust RDF serialization.

        Args:
            file_path (Optional[str]): Path to save RDF file. If None, returns RDF string.
            format (str): RDF serialization format ('Turtle', 'RDF/XML', 'JSON-LD', 'N-Triples')
            cypher_query (Optional[str]): Custom Cypher query. If None, uses default whole-graph query.

        Returns:
            Union[str, Dict[str, Any]]: RDF string if file_path is None, else export stats

        Raises:
            RuntimeError: If Neosemantics endpoint is not available or Neo4j handler not configured

        Example:
            >>> transformer = MlentoryTransformWithGraphBuilder(neo4j_handler=handler)
            >>> rdf_data = transformer.export_graph_neosemantics(format="Turtle")
            >>> # Or save to file:
            >>> stats = transformer.export_graph_neosemantics("graph.ttl", format="Turtle")
        """
        if not self.use_neo4j_store or not self.neo4j_handler:
            raise RuntimeError("Neo4j handler not configured. Cannot use Neosemantics export.")
        
        if cypher_query is None:
            # Default query to get all nodes and relationships
            cypher_query = "MATCH (n)-[r]->(m) RETURN n, r, m"

        # Build the Neosemantics endpoint URL
        base_url = self.neo4j_handler._build_http_url()
        endpoint = f"/rdf/{self.neo4j_handler.database}/cypher"

        # Prepare the request payload
        payload = {"cypher": cypher_query, "format": format}

        # Add authentication
        auth = (self.neo4j_handler.user, self.neo4j_handler.password)

        try:
            response = requests.post(
                f"{base_url}{endpoint}",
                json=payload,
                auth=auth,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Neosemantics endpoint returned {response.status_code}: {response.text}"
                )

            rdf_content = response.text

            if file_path:
                # Save to file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(rdf_content)

                return {
                    "file_path": file_path,
                    "format": format,
                    "endpoint": endpoint,
                    "success": True,
                }
            else:
                # Return RDF string
                return rdf_content

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to connect to Neosemantics endpoint: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to export graph using Neosemantics: {e}")

