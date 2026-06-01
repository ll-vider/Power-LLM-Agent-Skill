"""
context_assembler_skill/core/state_parser.py
---------------------------------------------
Parses raw simulation output from PandaPower, PSS/E, or a pre-parsed dict
into the standardised grid-state dictionary expected by ContextBuilder.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default overload threshold (per-unit line loading ratio)
DEFAULT_OVERLOAD_THRESHOLD = 1.0


class StateParser:
    """
    Converts raw simulator output to a standard grid-state dict.

    Supported simulator values: ``"pandapower"``, ``"psse"``, ``"dict"``.
    Pass ``"dict"`` when the input is already a pre-parsed dict (e.g. from a
    cached training episode).

    Parameters
    ----------
    simulator : str
        Name of the source simulator.
    field_map : dict
        Optional key remapping for non-standard field names.
    overload_threshold : float
        Line loading ratio above which a line is labelled overloaded.
    """

    def __init__(
        self,
        simulator: str = "pandapower",
        field_map: dict[str, str] | None = None,
        overload_threshold: float = DEFAULT_OVERLOAD_THRESHOLD,
    ) -> None:
        self.simulator = simulator
        self.field_map = field_map or {}
        self.overload_threshold = overload_threshold

    def parse(self, raw: Any) -> dict[str, Any]:
        """
        Parse raw simulation output and return a standardised state dict.

        Returns
        -------
        dict with keys:
            line_loading : dict[str, float]   — line_id → per-unit loading ratio
            bus_voltage  : dict[str, float]   — bus_id  → per-unit voltage
            overloaded_lines : list[str]
        """
        if self.simulator == "pandapower":
            return self._parse_pandapower(raw)
        elif self.simulator == "psse":
            return self._parse_psse(raw)
        elif self.simulator == "dict":
            return self._parse_dict(raw)
        else:
            raise ValueError(f"Unsupported simulator: {self.simulator!r}")

    # ------------------------------------------------------------------

    def _parse_pandapower(self, net: Any) -> dict[str, Any]:
        line_loading: dict[str, float] = {}
        bus_voltage: dict[str, float] = {}

        try:
            for idx, row in net.res_line.iterrows():
                line_id = f"L{idx}"
                loading = row.get("loading_percent", 0.0) / 100.0
                line_loading[line_id] = round(loading, 4)

            for idx, row in net.res_bus.iterrows():
                bus_id = f"BUS_{idx}"
                bus_voltage[bus_id] = round(row.get("vm_pu", 1.0), 4)
        except AttributeError:
            logger.warning("PandaPower net does not have result tables; "
                           "ensure net has been solved (pp.runpp).")

        overloaded = [lid for lid, val in line_loading.items()
                      if val > self.overload_threshold]
        return {
            "line_loading": line_loading,
            "bus_voltage": bus_voltage,
            "overloaded_lines": overloaded,
        }

    def _parse_psse(self, psse_dict: dict) -> dict[str, Any]:
        """Parse a pre-serialised PSS/E state dict."""
        line_loading = {
            self.field_map.get(k, k): v
            for k, v in psse_dict.get("line_loading", {}).items()
        }
        bus_voltage = {
            self.field_map.get(k, k): v
            for k, v in psse_dict.get("bus_voltage", {}).items()
        }
        overloaded = [lid for lid, val in line_loading.items()
                      if val > self.overload_threshold]
        return {
            "line_loading": line_loading,
            "bus_voltage": bus_voltage,
            "overloaded_lines": overloaded,
        }

    def _parse_dict(self, state_dict: dict) -> dict[str, Any]:
        line_loading = state_dict.get("line_loading", {})
        bus_voltage = state_dict.get("bus_voltage", {})
        overloaded = state_dict.get(
            "overloaded_lines",
            [lid for lid, val in line_loading.items()
             if val > self.overload_threshold],
        )
        return {
            "line_loading": line_loading,
            "bus_voltage": bus_voltage,
            "overloaded_lines": overloaded,
        }
