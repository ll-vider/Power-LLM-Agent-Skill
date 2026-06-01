"""
kg_retrieval_skill/core/subgraph_pruner.py
------------------------------------------
Prunes the scored BFS result to fit within a node-count or token budget,
and extracts procedure-text entries for kept nodes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class SubgraphPruner:
    """
    Trims a scored node list to a maximum node count and token budget.

    Procedure nodes (type == "procedure") are always kept if their parent
    topology node is kept, up to the token budget.

    Parameters
    ----------
    max_nodes : int
        Hard cap on the number of non-procedure nodes retained (default 80).
    token_budget : int
        Approximate token budget for the entire subgraph JSON (default 1500).
        Token estimate: len(json_string) // 4.
    """

    def __init__(self, max_nodes: int = 80, token_budget: int = 1500) -> None:
        self.max_nodes = max_nodes
        self.token_budget = token_budget

    def prune(
        self,
        scored_nodes: list[tuple[str, float]],
        triples: list[tuple[str, str, str]],
        graph: nx.DiGraph,
    ) -> tuple[set[str], list[dict[str, str]]]:
        """
        Select nodes to keep and collect procedure texts.

        Returns
        -------
        kept_node_ids : set[str]
        procedure_texts : list[dict]  — [{"id": ..., "text": ...}, ...]
        """
        kept: list[str] = []
        procedure_texts: list[dict[str, str]] = []
        running_tokens = 0

        for node_id, _score in scored_nodes:
            if len(kept) >= self.max_nodes:
                break
            node_data = graph.nodes[node_id]
            node_json = json.dumps({
                "id": node_id,
                "type": node_data.get("type", "unknown"),
                "attrs": {k: v for k, v in node_data.items() if k != "type"},
            })
            node_tokens = len(node_json) // 4

            if running_tokens + node_tokens > self.token_budget:
                logger.debug(
                    "Token budget reached at node '%s' (%d tokens used).",
                    node_id, running_tokens,
                )
                break

            kept.append(node_id)
            running_tokens += node_tokens

            # Collect procedure texts linked to this node
            for _src, rel, tgt in triples:
                if _src == node_id and rel == "has_procedure":
                    if tgt in graph.nodes:
                        proc_data = graph.nodes[tgt]
                        proc_text = proc_data.get("text", "")
                        if proc_text:
                            procedure_texts.append({"id": tgt, "text": proc_text})

        logger.info(
            "Pruner kept %d nodes, %d procedure entries (~%d tokens).",
            len(kept), len(procedure_texts), running_tokens,
        )
        return set(kept), procedure_texts
