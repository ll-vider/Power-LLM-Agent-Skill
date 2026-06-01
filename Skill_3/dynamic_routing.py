"""
power_llm_agent_skill/core/dynamic_routing.py
----------------------------------------------
Layer 1 of the Power LLM Agent Skill: Dynamic Routing Layer.

Analyses the structured context C to determine the knowledge scope and
dispatch-rule categories relevant to the current contingency, and focuses
downstream reasoning on the appropriate knowledge branches.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Mapping from overload severity to candidate rule categories
_SEVERITY_RULE_MAP: dict[str, list[str]] = {
    "critical":  ["N-1_thermal", "emergency_load_shedding", "spinning_reserve"],
    "severe":    ["N-1_thermal", "spinning_reserve", "reactive_compensation"],
    "moderate":  ["N-1_thermal", "reactive_compensation"],
    "normal":    ["reactive_compensation"],
}

# Overload ratio thresholds for severity levels
_SEVERITY_THRESHOLDS = {
    "critical": 1.30,
    "severe":   1.15,
    "moderate": 1.05,
}


class DynamicRoutingLayer:
    """
    Identifies the knowledge scope and dispatch-rule categories for the
    current contingency scenario.

    Parameters
    ----------
    additional_rule_categories : list[str] | None
        Extra rule categories to append to the standard set, if any.
    """

    def __init__(
        self,
        additional_rule_categories: list[str] | None = None,
    ) -> None:
        self.additional_rule_categories = additional_rule_categories or []

    def route(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Analyse context C and return routing metadata.

        Parameters
        ----------
        context : dict
            Structured context C from the Context Assembler Skill.

        Returns
        -------
        dict with keys:
            severity        : str   — "critical" | "severe" | "moderate" | "normal"
            max_loading     : float — maximum per-unit line loading ratio
            rule_categories : list[str]
            focused_triples : list  — triples filtered to relevant rule categories
            focused_procedures : list — procedure texts filtered to relevant categories
        """
        line_loading: dict[str, float] = context.get("line_loading", {})
        overloaded_lines: list[str] = context.get("overloaded_lines", [])
        subgraph: dict[str, Any] = context.get("subgraph", {})

        # Determine overload severity
        max_loading = max(line_loading.values()) if line_loading else 0.0
        severity = self._classify_severity(max_loading)

        # Select rule categories
        rule_categories: list[str] = list(
            _SEVERITY_RULE_MAP.get(severity, ["reactive_compensation"])
        )
        rule_categories.extend(self.additional_rule_categories)

        # Filter subgraph triples to those relevant to rule categories
        focused_triples, focused_procedures = self._filter_subgraph(
            subgraph=subgraph,
            rule_categories=rule_categories,
        )

        routing = {
            "severity": severity,
            "max_loading": round(max_loading, 4),
            "overloaded_lines": overloaded_lines,
            "rule_categories": rule_categories,
            "focused_triples": focused_triples,
            "focused_procedures": focused_procedures,
        }

        logger.info(
            "Routing: severity=%s (max_loading=%.2f), categories=%s, "
            "focused_procedures=%d",
            severity, max_loading, rule_categories, len(focused_procedures),
        )
        return routing

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_severity(max_loading: float) -> str:
        if max_loading >= _SEVERITY_THRESHOLDS["critical"]:
            return "critical"
        elif max_loading >= _SEVERITY_THRESHOLDS["severe"]:
            return "severe"
        elif max_loading >= _SEVERITY_THRESHOLDS["moderate"]:
            return "moderate"
        else:
            return "normal"

    @staticmethod
    def _filter_subgraph(
        subgraph: dict[str, Any],
        rule_categories: list[str],
    ) -> tuple[list, list]:
        """
        Retain only triples and procedure texts whose category tag matches
        any of the identified rule categories.
        """
        triples = subgraph.get("triples", [])
        procedures = subgraph.get("procedure_texts", [])

        # Keep triples involving "has_procedure" or topology relations
        relevant_triples = [
            t for t in triples
            if t[1] in ("has_procedure", "connects_to", "controls", "monitors")
        ]

        # Filter procedures by keyword presence
        keywords = set()
        for cat in rule_categories:
            keywords.update(cat.lower().replace("_", " ").split())

        relevant_procedures = []
        for proc in procedures:
            text_lower = proc.get("text", "").lower()
            if any(kw in text_lower for kw in keywords):
                relevant_procedures.append(proc)

        # Fallback: include all procedures if none matched
        if not relevant_procedures:
            relevant_procedures = procedures

        return relevant_triples, relevant_procedures
