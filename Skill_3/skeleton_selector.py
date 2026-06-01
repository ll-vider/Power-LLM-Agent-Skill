"""
power_llm_agent_skill/core/skeleton_selector.py
------------------------------------------------
Layer 2 of the Power LLM Agent Skill: Instruction Skeleton Layer.

Selects the appropriate reasoning skeleton (S_exp or S_con) based on the
training-stage indicator φ and the contingency severity returned by the
Dynamic Routing Layer, then renders the skeleton prompt text for injection
into the per-resource eligibility prompt.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)


class SkeletonMode(str, Enum):
    EXPLORATORY  = "exploratory"
    CONSERVATIVE = "conservative"


class SkeletonSelector:
    """
    Selects and renders the instruction skeleton for the current step.

    Decision logic
    --------------
    - Use ``S_con`` (conservative) if:
        * φ > phi_exp_threshold  (policy nearing convergence), OR
        * severity is "critical" or "severe" (high-risk operating state), OR
        * fallback_triggered is True (previous mask was invalid)
    - Use ``S_exp`` (exploratory) otherwise.

    Parameters
    ----------
    prompt_dir : str | Path
        Directory containing the Jinja2 skeleton templates.
    phi_exp_threshold : float
        φ value above which the conservative skeleton is preferred (default 0.4).
    """

    TEMPLATE_MAP = {
        SkeletonMode.EXPLORATORY:  "skeleton_exploratory.jinja2",
        SkeletonMode.CONSERVATIVE: "skeleton_conservative.jinja2",
    }

    def __init__(
        self,
        prompt_dir: str | Path,
        phi_exp_threshold: float = 0.4,
    ) -> None:
        self.phi_exp_threshold = phi_exp_threshold
        prompt_dir = Path(prompt_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(prompt_dir)),
            autoescape=select_autoescape([]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_and_render(
        self,
        phi: float,
        severity: str,
        rule_categories: list[str],
        fallback_triggered: bool = False,
    ) -> tuple[SkeletonMode, str]:
        """
        Select the skeleton mode and render the instruction text.

        Parameters
        ----------
        phi : float
            Training-stage indicator ∈ [0, 1].
        severity : str
            Overload severity from DynamicRoutingLayer ("critical", "severe", etc.).
        rule_categories : list[str]
            Dispatch-rule categories identified by DynamicRoutingLayer.
        fallback_triggered : bool
            Whether the previous mask generation used the fallback mask.

        Returns
        -------
        mode : SkeletonMode
        rendered_text : str
        """
        mode = self._decide_mode(phi, severity, fallback_triggered)
        template_name = self.TEMPLATE_MAP[mode]
        template = self.env.get_template(template_name)
        rendered = template.render(
            phi=phi,
            severity=severity,
            rule_categories=rule_categories,
        )
        logger.info(
            "Skeleton selected: %s (phi=%.2f, severity=%s, fallback=%s)",
            mode.value, phi, severity, fallback_triggered,
        )
        return mode, rendered

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _decide_mode(
        self,
        phi: float,
        severity: str,
        fallback_triggered: bool,
    ) -> SkeletonMode:
        if fallback_triggered:
            logger.warning("Previous fallback triggered — switching to conservative skeleton.")
            return SkeletonMode.CONSERVATIVE
        if severity in ("critical", "severe"):
            return SkeletonMode.CONSERVATIVE
        if phi > self.phi_exp_threshold:
            return SkeletonMode.CONSERVATIVE
        return SkeletonMode.EXPLORATORY
