from rdflib import Graph
from .GraphHandler import GraphHandler


class GraphHandlerForKG(GraphHandler):
    """Compatibility shim. Prefer using `GraphHandler` directly.

    This class simply preserves imports for existing code that referenced
    `GraphHandlerForKG`. It inherits from `GraphHandler` without adding
    functionality. Default identifiers remain the mlentory graph URIs.
    """

    def __init__(
        self,
        SQLHandler,
        RDFHandler,
        IndexHandler,
        kg_files_directory: str = "./../kg_files",
        platform: str = "hugging_face",
        graph_identifier: str = "https://w3id.org/mlentory/mlentory_graph",
        deprecated_graph_identifier: str = "https://w3id.org/mlentory/deprecated_mlentory_graph",
        logger=None,
    ):
        super().__init__(
            SQLHandler,
            RDFHandler,
            IndexHandler,
            kg_files_directory,
            platform,
            graph_identifier,
            deprecated_graph_identifier,
            logger,
        )