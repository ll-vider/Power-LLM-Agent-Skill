"""
kg_retrieval_skill/core/multi_hop_bfs.py
----------------------------------------
Multi-hop breadth-first search over a power dispatch knowledge graph.

The KGRetriever is the public entry point for this Skill. It accepts a
contingency identifier and a real-time grid snapshot, performs multi-hop BFS
from the affected nodes, scores every retrieved node, prunes to a token budget,
and returns a validated local knowledge subgraph.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import yaml

from .graph_loader import GraphLoader
from .relevance_scorer import RelevanceScorer
from .subgraph_pruner import SubgraphPruner

logger = logging.getLogger(__name__)


class KGRetriever:
    """
    Multi-hop BFS retrieval over a power dispatch knowledge graph.

    Parameters
    ----------
    graph_loader : GraphLoader
        Loaded and indexed knowledge graph.
    scorer : RelevanceScorer
        Node / edge relevance scorer.
    pruner : SubgraphPruner
        Token-budget-aware subgraph pruner.
    hop_count : int
        Maximum number of BFS hops from seed nodes (default: 3).
    edge_type_filter : list[str] | None
        If provided, only traverse edges whose type is in this list.
    """

    def __init__(
        self,
        graph_loader: GraphLoader,
        scorer: RelevanceScorer,
        pruner: SubgraphPruner,
        hop_count: int = 3,
        edge_type_filter: list[str] | None = None,
    ) -> None:
        self.graph = graph_loader.graph
        self.scorer = scorer
        self.pruner = pruner
        self.hop_count = hop_count
        self.edge_type_filter = set(edge_type_filter) if edge_type_filter else None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path) -> "KGRetriever":
        """Instantiate from a YAML configuration file."""
        config_path = Path(config_path)
        with config_path.open() as f:
            cfg = yaml.safe_load(f)

        graph_loader = GraphLoader(
            source=cfg["graph"]["source"],
            source_type=cfg["graph"].get("source_type", "networkx"),
        )
        scorer = RelevanceScorer(
            tfidf_weight=cfg["scoring"].get("tfidf_weight", 0.5),
            distance_weight=cfg["scoring"].get("distance_weight", 0.5),
        )
        pruner = SubgraphPruner(
            max_nodes=cfg["pruning"].get("max_nodes", 80),
            token_budget=cfg["pruning"].get("token_budget", 1500),
        )
        return cls(
            graph_loader=graph_loader,
            scorer=scorer,
            pruner=pruner,
            hop_count=cfg.get("hop_count", 3),
            edge_type_filter=cfg.get("edge_type_filter"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        contingency_id: str,
        grid_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Retrieve a local knowledge subgraph for the given contingency.

        Parameters
        ----------
        contingency_id : str
            Identifier of the faulted element (e.g. "L12").
        grid_snapshot : dict
            Real-time operational state: line loadings, bus voltages,
            overloaded lines, etc.

        Returns
        -------
        dict
            Local knowledge subgraph conforming to
            ``schemas/output_schema.json``.
        """
        logger.info("Retrieving subgraph for contingency '%s'", contingency_id)

        # 1. Resolve seed nodes from the contingency identifier
        seed_nodes = self._resolve_seeds(contingency_id, grid_snapshot)
        if not seed_nodes:
            logger.warning(
                "No seed nodes found for contingency '%s'; returning empty subgraph.",
                contingency_id,
            )
            return self._empty_subgraph(contingency_id)

        # 2. BFS expansion
        raw_nodes, raw_triples = self._bfs_expand(seed_nodes)
        logger.debug(
            "BFS yielded %d nodes and %d triples before pruning.",
            len(raw_nodes),
            len(raw_triples),
        )

        # 3. Score every node
        scored_nodes = self.scorer.score_nodes(
            nodes=raw_nodes,
            seed_nodes=seed_nodes,
            grid_snapshot=grid_snapshot,
            graph=self.graph,
        )

        # 4. Prune to token budget
        kept_node_ids, procedure_texts = self.pruner.prune(
            scored_nodes=scored_nodes,
            triples=raw_triples,
            graph=self.graph,
        )

        # 5. Filter triples to kept nodes
        kept_triples = [
            t for t in raw_triples
            if t[0] in kept_node_ids and t[2] in kept_node_ids
        ]

        subgraph = self._build_output(
            contingency_id=contingency_id,
            kept_node_ids=kept_node_ids,
            triples=kept_triples,
            procedure_texts=procedure_texts,
        )
        logger.info(
            "Subgraph built: %d nodes, %d triples, ~%d tokens.",
            len(subgraph["nodes"]),
            len(subgraph["triples"]),
            subgraph["token_count"],
        )
        return subgraph

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_seeds(
        self,
        contingency_id: str,
        grid_snapshot: dict[str, Any],
    ) -> list[str]:
        """Return the list of KG node IDs that are direct seeds for the fault."""
        seeds = []
        if contingency_id in self.graph.nodes:
            seeds.append(contingency_id)
        # Also seed from overloaded lines reported in the snapshot
        for line_id in grid_snapshot.get("overloaded_lines", []):
            if line_id in self.graph.nodes and line_id not in seeds:
                seeds.append(line_id)
        return seeds

    def _bfs_expand(
        self,
        seed_nodes: list[str],
    ) -> tuple[list[str], list[tuple[str, str, str]]]:
        """
        Execute BFS from seed nodes up to ``self.hop_count`` hops.

        Returns
        -------
        nodes : list[str]
            All node IDs visited during BFS.
        triples : list[tuple[str, str, str]]
            All (subject, relation, object) triples encountered.
        """
        visited: set[str] = set()
        triples: list[tuple[str, str, str]] = []

        queue: deque[tuple[str, int]] = deque(
            (node, 0) for node in seed_nodes
        )
        for node in seed_nodes:
            visited.add(node)

        while queue:
            node_id, depth = queue.popleft()
            if depth >= self.hop_count:
                continue
            for neighbor in self.graph.neighbors(node_id):
                edge_data = self.graph[node_id][neighbor]
                rel_type = edge_data.get("relation", "related_to")

                # Apply optional edge-type filter
                if self.edge_type_filter and rel_type not in self.edge_type_filter:
                    continue

                triples.append((node_id, rel_type, neighbor))

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return list(visited), triples

    def _build_output(
        self,
        contingency_id: str,
        kept_node_ids: set[str],
        triples: list[tuple[str, str, str]],
        procedure_texts: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Assemble the final output dict from kept nodes and triples."""
        nodes = []
        for nid in kept_node_ids:
            node_data = self.graph.nodes[nid]
            nodes.append({
                "id": nid,
                "type": node_data.get("type", "unknown"),
                "attrs": {
                    k: v for k, v in node_data.items() if k != "type"
                },
            })

        serialised_triples = [list(t) for t in triples]

        # Rough token estimate: 4 chars ≈ 1 token
        raw_text = json.dumps({"nodes": nodes, "triples": serialised_triples,
                                "procedure_texts": procedure_texts})
        token_estimate = len(raw_text) // 4

        return {
            "contingency_id": contingency_id,
            "nodes": nodes,
            "triples": serialised_triples,
            "procedure_texts": procedure_texts,
            "token_count": token_estimate,
        }

    @staticmethod
    def _empty_subgraph(contingency_id: str) -> dict[str, Any]:
        return {
            "contingency_id": contingency_id,
            "nodes": [],
            "triples": [],
            "procedure_texts": [],
            "token_count": 0,
        }
