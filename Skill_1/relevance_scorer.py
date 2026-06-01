"""
kg_retrieval_skill/core/relevance_scorer.py
-------------------------------------------
Scores each BFS-retrieved KG node by combining textual TF-IDF relevance
and graph topological distance from the seed nodes.
"""

from __future__ import annotations

import math
import logging
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class RelevanceScorer:
    """
    Scores knowledge graph nodes retrieved by BFS.

    Score = tfidf_weight * text_score + distance_weight * topology_score

    Both sub-scores are in [0, 1]; higher is more relevant.

    Parameters
    ----------
    tfidf_weight : float
        Weight for the textual TF-IDF component (default 0.5).
    distance_weight : float
        Weight for the topological-distance component (default 0.5).
    """

    def __init__(
        self,
        tfidf_weight: float = 0.5,
        distance_weight: float = 0.5,
    ) -> None:
        if abs(tfidf_weight + distance_weight - 1.0) > 1e-6:
            raise ValueError("tfidf_weight + distance_weight must equal 1.0")
        self.tfidf_weight = tfidf_weight
        self.distance_weight = distance_weight

    def score_nodes(
        self,
        nodes: list[str],
        seed_nodes: list[str],
        grid_snapshot: dict[str, Any],
        graph: nx.DiGraph,
    ) -> list[tuple[str, float]]:
        """
        Return a list of (node_id, score) pairs, sorted descending by score.

        Parameters
        ----------
        nodes : list[str]
            All node IDs returned by BFS.
        seed_nodes : list[str]
            The original seed node IDs (contingency-affected elements).
        grid_snapshot : dict
            Real-time operational state, used to build a query term set.
        graph : nx.DiGraph
            The full knowledge graph (for shortest-path computation).
        """
        # Build query terms from grid snapshot keys + values
        query_terms = set()
        for key in grid_snapshot:
            query_terms.add(key.lower())
        for val in grid_snapshot.get("overloaded_lines", []):
            query_terms.add(str(val).lower())

        # Precompute shortest distances from each seed to every node
        distance_map: dict[str, int] = {}
        for seed in seed_nodes:
            try:
                lengths = nx.single_source_shortest_path_length(graph, seed)
                for nid, dist in lengths.items():
                    existing = distance_map.get(nid, 9999)
                    if dist < existing:
                        distance_map[nid] = dist
            except nx.NetworkXError:
                pass

        max_dist = max(distance_map.values()) if distance_map else 1

        scored: list[tuple[str, float]] = []
        for nid in nodes:
            text_score = self._text_score(nid, graph, query_terms)
            dist = distance_map.get(nid, max_dist)
            topo_score = 1.0 - (dist / max(max_dist, 1))
            combined = self.tfidf_weight * text_score + self.distance_weight * topo_score
            scored.append((nid, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _text_score(
        node_id: str,
        graph: nx.DiGraph,
        query_terms: set[str],
    ) -> float:
        """
        Simple TF-like score: fraction of query terms appearing in the
        node's attributes (id + all string attribute values).
        """
        if not query_terms:
            return 0.5  # neutral if no query terms

        node_text = node_id.lower()
        for val in graph.nodes[node_id].values():
            node_text += " " + str(val).lower()

        hits = sum(1 for term in query_terms if term in node_text)
        return hits / len(query_terms)
