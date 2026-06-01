"""
context_assembler_skill/core/resource_enumerator.py
----------------------------------------------------
Enumerates controllable resources available in the current grid state
and computes their regulation margins.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ResourceEnumerator:
    """
    Returns a list of controllable resource dicts for the current step.

    Each resource dict contains:
        id                  : str
        type                : str  (generator | load | shunt | tap_changer)
        current_output_mw   : float
        regulation_margin_mw: float   (positive = room to reduce; negative means at limit)
        status              : str  (online | offline | standby)

    Parameters
    ----------
    source : str
        ``"simulation"`` — read from PandaPower net (passed at enumerate time);
        ``"static"``     — read from a pre-configured resource registry.
    resource_registry : dict | None
        Used when source == "static". Maps resource_id → metadata.
    """

    def __init__(
        self,
        source: str = "simulation",
        resource_registry: dict[str, dict] | None = None,
    ) -> None:
        self.source = source
        self.resource_registry = resource_registry or {}

    def enumerate(self, parsed_state: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Return the controllable resource list for the current operating state.

        Parameters
        ----------
        parsed_state : dict
            Standard state dict from StateParser.

        Returns
        -------
        list[dict]  — one entry per controllable resource.
        """
        if self.source == "static":
            return self._from_registry(parsed_state)
        elif self.source == "simulation":
            return self._from_simulation(parsed_state)
        else:
            raise ValueError(f"Unknown resource source: {self.source!r}")

    # ------------------------------------------------------------------

    def _from_registry(self, parsed_state: dict) -> list[dict[str, Any]]:
        resources = []
        for rid, meta in self.resource_registry.items():
            resources.append({
                "id": rid,
                "type": meta.get("type", "generator"),
                "current_output_mw": meta.get("current_output_mw", 0.0),
                "regulation_margin_mw": meta.get("pmax_mw", 0.0) - meta.get("current_output_mw", 0.0),
                "status": meta.get("status", "online"),
            })
        return resources

    def _from_simulation(self, parsed_state: dict) -> list[dict[str, Any]]:
        """
        Infer resources from the parsed state dict.
        In real deployments, replace this with SCADA / Environment API calls.
        """
        # Placeholder: returns a minimal resource list derived from parsed_state.
        # Override or subclass for production use.
        resources = []
        for key in parsed_state:
            if key.startswith("gen_") or key.startswith("G"):
                val = parsed_state[key]
                if isinstance(val, dict):
                    resources.append({
                        "id": key,
                        "type": "generator",
                        "current_output_mw": val.get("p_mw", 0.0),
                        "regulation_margin_mw": val.get("pmax_mw", 0.0) - val.get("p_mw", 0.0),
                        "status": val.get("status", "online"),
                    })
        return resources


# ---------------------------------------------------------------------------


"""
context_assembler_skill/core/schema_validator.py
-------------------------------------------------
Validates assembled context C against the output JSON Schema.
"""

import json
from pathlib import Path

import jsonschema


class SchemaValidator:
    """
    Validates a dict against a JSON Schema file.

    Raises ``jsonschema.ValidationError`` with a field-located message on failure.
    """

    def __init__(self, schema_path: str | Path) -> None:
        schema_path = Path(schema_path)
        if schema_path.exists():
            with schema_path.open() as f:
                self.schema = json.load(f)
        else:
            # Permissive fallback if schema file not yet present
            self.schema = {}

    def validate(self, instance: dict) -> None:
        if not self.schema:
            return  # no schema loaded — skip
        jsonschema.validate(instance=instance, schema=self.schema)
