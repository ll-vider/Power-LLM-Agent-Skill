"""
context_assembler_skill/core/context_builder.py
------------------------------------------------
Main assembly logic for the Structured Dispatch Context Assembler Skill.

Takes a local knowledge subgraph (Skill 1 output), real-time grid state,
and the RL training-stage indicator φ, and produces the unified structured
context object C that the Power LLM Agent Skill (Skill 3) expects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from .schema_validator import SchemaValidator
from .state_parser import StateParser
from .resource_enumerator import ResourceEnumerator

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Assembles the structured context object C.

    Parameters
    ----------
    state_parser : StateParser
        Parses raw simulation output into the standardised grid-state dict.
    resource_enumerator : ResourceEnumerator
        Returns the current controllable resource list with margins.
    validator : SchemaValidator
        Validates the assembled C against the output JSON Schema.
    phi_thresholds : dict
        Thresholds that map φ float values to named stages for logging.
    """

    def __init__(
        self,
        state_parser: StateParser,
        resource_enumerator: ResourceEnumerator,
        validator: SchemaValidator,
        phi_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.state_parser = state_parser
        self.resource_enumerator = resource_enumerator
        self.validator = validator
        self.phi_thresholds = phi_thresholds or {
            "exploratory": 0.4,
            "converging": 0.8,
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path) -> "ContextBuilder":
        config_path = Path(config_path)
        with config_path.open() as f:
            cfg = yaml.safe_load(f)

        schema_dir = Path(config_path).parent.parent / "schemas"
        validator = SchemaValidator(
            schema_path=str(schema_dir / "output_context_schema.json")
        )
        state_parser = StateParser(
            simulator=cfg.get("simulator", "pandapower"),
            field_map=cfg.get("field_map", {}),
        )
        resource_enumerator = ResourceEnumerator(
            source=cfg.get("resource_source", "simulation"),
        )
        return cls(
            state_parser=state_parser,
            resource_enumerator=resource_enumerator,
            validator=validator,
            phi_thresholds=cfg.get("phi_thresholds"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        subgraph: dict[str, Any],
        grid_state: Any,
        phi: float,
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Assemble and return the structured context C.

        Parameters
        ----------
        subgraph : dict
            Output of the KG-Retrieval Skill (Skill 1).
        grid_state : Any
            Raw simulation output (PandaPower net, PSS/E dict, or pre-parsed dict).
        phi : float
            Training-stage indicator in [0, 1].  0 = early exploration,
            1 = fully converged policy.
        extra_fields : dict | None
            Optional additional fields to merge into C.

        Returns
        -------
        dict
            Structured context C, validated against output_context_schema.json.
        """
        if not 0.0 <= phi <= 1.0:
            raise ValueError(f"phi must be in [0, 1], got {phi}")

        # 1. Parse grid state into standard fields
        parsed_state = self.state_parser.parse(grid_state)

        # 2. Enumerate controllable resources
        resource_list = self.resource_enumerator.enumerate(parsed_state)

        # 3. Assemble context
        context: dict[str, Any] = {
            "contingency_id": subgraph.get("contingency_id", "unknown"),
            "overloaded_lines": parsed_state.get("overloaded_lines", []),
            "bus_voltage": parsed_state.get("bus_voltage", {}),
            "line_loading": parsed_state.get("line_loading", {}),
            "resource_list": resource_list,
            "subgraph": subgraph,
            "phi": phi,
            "phi_stage": self._phi_to_stage(phi),
        }

        if extra_fields:
            context.update(extra_fields)

        # 4. Validate
        self.validator.validate(context)

        logger.info(
            "Context C assembled: contingency='%s', phi=%.2f (%s), "
            "%d resources, subgraph nodes=%d",
            context["contingency_id"],
            phi,
            context["phi_stage"],
            len(resource_list),
            len(subgraph.get("nodes", [])),
        )
        return context

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _phi_to_stage(self, phi: float) -> str:
        if phi <= self.phi_thresholds.get("exploratory", 0.4):
            return "exploratory"
        elif phi <= self.phi_thresholds.get("converging", 0.8):
            return "converging"
        else:
            return "converged"
