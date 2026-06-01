"""
kg_retrieval_skill/core/graph_loader.py
---------------------------------------
Loads and indexes a power dispatch knowledge graph from a NetworkX JSON
file or a live Neo4j instance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class GraphLoader:
    """
    Loads a knowledge graph and exposes it as a NetworkX DiGraph.

    Parameters
    ----------
    source : str
        Either a file path to a NetworkX JSON file or a Neo4j connection URI.
    source_type : str
        ``"networkx"`` (default) or ``"neo4j"``.
    """

    def __init__(self, source: str, source_type: str = "networkx") -> None:
        self.source = source
        self.source_type = source_type
        self.graph: nx.DiGraph = self._load()

    def _load(self) -> nx.DiGraph:
        if self.source_type == "networkx":
            return self._load_networkx()
        elif self.source_type == "neo4j":
            return self._load_neo4j()
        else:
            raise ValueError(f"Unsupported source_type: {self.source_type!r}")

    def _load_networkx(self) -> nx.DiGraph:
        path = Path(self.source)
        logger.info("Loading NetworkX graph from %s", path)
        with path.open() as f:
            data = json.load(f)
        graph = nx.node_link_graph(data, directed=True, multigraph=False)
        logger.info("Graph loaded: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
        return graph

    def _load_neo4j(self) -> nx.DiGraph:
        """Convert a Neo4j database to an in-memory NetworkX DiGraph."""
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as exc:
            raise ImportError("Install neo4j>=5.0 to use source_type='neo4j'.") from exc

        driver = GraphDatabase.driver(self.source)
        graph = nx.DiGraph()
        with driver.session() as session:
            # Load nodes
            for record in session.run("MATCH (n) RETURN n"):
                node = record["n"]
                graph.add_node(str(node.id), **dict(node.items()))
            # Load edges
            for record in session.run("MATCH ()-[r]->() RETURN r"):
                rel = record["r"]
                graph.add_edge(
                    str(rel.start_node.id),
                    str(rel.end_node.id),
                    relation=rel.type,
                    **dict(rel.items()),
                )
        driver.close()
        logger.info("Neo4j graph loaded: %d nodes, %d edges",
                    graph.number_of_nodes(), graph.number_of_edges())
        return graph
