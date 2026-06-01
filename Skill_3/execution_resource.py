"""
power_llm_agent_skill/core/execution_resource.py
-------------------------------------------------
Layer 3 of the Power LLM Agent Skill: Execution Resource Layer.

Public entry point: PowerLLMAgentSkill — the top-level orchestrator that
wires together all three layers and the alignment checker, and exposes the
``run()`` method consumed by the KM-PPO Integrator (Skill 4).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .dynamic_routing import DynamicRoutingLayer
from .skeleton_selector import SkeletonSelector
from .alignment_checker import AlignmentChecker
from .evidence_recorder import EvidenceRecorder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM client abstraction
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper around an LLM API.

    Supports "openai", "anthropic", and "local" backends.
    Override ``chat()`` to integrate a custom power-domain LLM.
    """

    def __init__(self, backend: str, model: str, temperature: float, max_tokens: int) -> None:
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt and return the raw text response."""
        if self.backend == "openai":
            return self._openai_chat(system_prompt, user_prompt)
        elif self.backend == "anthropic":
            return self._anthropic_chat(system_prompt, user_prompt)
        else:
            raise NotImplementedError(
                f"Backend '{self.backend}' not implemented. "
                "Override LLMClient.chat() for custom backends."
            )

    def _openai_chat(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI  # type: ignore
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def _anthropic_chat(self, system_prompt: str, user_prompt: str) -> str:
        import anthropic  # type: ignore
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text if message.content else ""


# ---------------------------------------------------------------------------
# Per-resource eligibility reasoning
# ---------------------------------------------------------------------------

class ExecutionResourceLayer:
    """
    Iterates over the RL action resources and calls the LLM to perform
    the three-condition eligibility check for each resource.

    Parameters
    ----------
    llm : LLMClient
    prompt_dir : Path
    """

    def __init__(self, llm: LLMClient, prompt_dir: Path) -> None:
        self.llm = llm
        env = Environment(
            loader=FileSystemLoader(str(prompt_dir)),
            autoescape=select_autoescape([]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.eligibility_template = env.get_template("resource_eligibility.jinja2")
        self.routing_template = env.get_template("routing_prompt.jinja2")

    def run(
        self,
        action_resources: list[str],
        context: dict[str, Any],
        routing: dict[str, Any],
        skeleton_text: str,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        """
        Run per-resource eligibility reasoning.

        Returns
        -------
        raw_mask : list[int]
            Binary mask, one entry per resource (may be ill-formed — checked
            by AlignmentChecker afterwards).
        evidence_list : list[dict]
            One evidence entry per resource.
        """
        resource_set = set(
            node["id"] for node in context.get("subgraph", {}).get("nodes", [])
        )

        raw_mask: list[int] = []
        evidence_list: list[dict[str, Any]] = []

        system_prompt = (
            "You are a power system dispatch knowledge reasoning engine. "
            "Your task is to evaluate whether a specific controllable resource "
            "is eligible to participate in post-contingency power flow regulation. "
            "Respond ONLY with a valid JSON object. No preamble, no markdown."
        )

        for resource_id in action_resources:
            # Build the resource metadata from context
            resource_meta = self._get_resource_meta(resource_id, context)

            user_prompt = self.eligibility_template.render(
                resource_id=resource_id,
                resource_meta=resource_meta,
                contingency_id=context.get("contingency_id", "unknown"),
                overloaded_lines=routing.get("overloaded_lines", []),
                focused_procedures=routing.get("focused_procedures", []),
                rule_categories=routing.get("rule_categories", []),
                skeleton_instruction=skeleton_text,
                subgraph_resource_ids=list(resource_set),
            )

            try:
                response_text = self.llm.chat(system_prompt, user_prompt)
                result = self._parse_response(response_text)
            except Exception as exc:
                logger.warning("LLM call failed for resource '%s': %s", resource_id, exc)
                result = {
                    "eligible": True,  # safe fallback: include the resource
                    "C1_in_subgraph": None,
                    "C2_margin_ok": None,
                    "C3_rule_ok": None,
                    "source_text": f"LLM error: {exc}",
                }

            mask_val = 1 if result.get("eligible", True) else 0
            raw_mask.append(mask_val)

            evidence_list.append({
                "resource_id": resource_id,
                "C1_in_subgraph": result.get("C1_in_subgraph"),
                "C2_margin_ok":   result.get("C2_margin_ok"),
                "C3_rule_ok":     result.get("C3_rule_ok"),
                "eligible":       bool(mask_val),
                "source_text":    result.get("source_text", ""),
            })

        return raw_mask, evidence_list

    # ------------------------------------------------------------------

    @staticmethod
    def _get_resource_meta(
        resource_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        for res in context.get("resource_list", []):
            if res["id"] == resource_id:
                return res
        return {"id": resource_id, "status": "unknown"}

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        """Parse the LLM JSON response, stripping markdown fences if present."""
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1])
        return json.loads(clean)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

class PowerLLMAgentSkill:
    """
    Top-level entry point for the Power LLM Agent Skill.

    Wires together:
        Layer 1 — DynamicRoutingLayer
        Layer 2 — SkeletonSelector
        Layer 3 — ExecutionResourceLayer
        Post   — AlignmentChecker + EvidenceRecorder
    """

    def __init__(
        self,
        routing_layer: DynamicRoutingLayer,
        skeleton_selector: SkeletonSelector,
        execution_layer: ExecutionResourceLayer,
        alignment_checker: AlignmentChecker,
        evidence_recorder: EvidenceRecorder,
    ) -> None:
        self.routing_layer = routing_layer
        self.skeleton_selector = skeleton_selector
        self.execution_layer = execution_layer
        self.alignment_checker = alignment_checker
        self.evidence_recorder = evidence_recorder
        self._last_fallback_triggered = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path) -> "PowerLLMAgentSkill":
        config_path = Path(config_path)
        with config_path.open() as f:
            cfg = yaml.safe_load(f)

        skill_dir = config_path.parent.parent
        prompt_dir = skill_dir / "prompts"
        llm_cfg_path = skill_dir / "configs" / "llm_config.yaml"

        with llm_cfg_path.open() as f:
            llm_cfg = yaml.safe_load(f)

        llm = LLMClient(
            backend=llm_cfg.get("backend", "openai"),
            model=llm_cfg.get("model", "gpt-4o"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 512),
        )

        return cls(
            routing_layer=DynamicRoutingLayer(
                additional_rule_categories=cfg.get("additional_rule_categories"),
            ),
            skeleton_selector=SkeletonSelector(
                prompt_dir=prompt_dir,
                phi_exp_threshold=cfg.get("phi_exp_threshold", 0.4),
            ),
            execution_layer=ExecutionResourceLayer(llm=llm, prompt_dir=prompt_dir),
            alignment_checker=AlignmentChecker(),
            evidence_recorder=EvidenceRecorder(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        context: dict[str, Any],
        action_resources: list[str],
    ) -> tuple[list[int], list[dict[str, Any]]]:
        """
        Generate the knowledge mask and reasoning evidence for one RL step.

        Parameters
        ----------
        context : dict
            Structured context C from the Context Assembler Skill.
        action_resources : list[str]
            Ordered list of resource IDs matching the RL action dimensions.

        Returns
        -------
        mask : list[int]
            Binary mask, length == len(action_resources), values ∈ {0, 1}.
        evidence : list[dict]
            One evidence dict per resource.
        """
        phi = context.get("phi", 0.0)

        # Layer 1: Dynamic Routing
        routing = self.routing_layer.route(context)

        # Layer 2: Skeleton Selection
        _mode, skeleton_text = self.skeleton_selector.select_and_render(
            phi=phi,
            severity=routing["severity"],
            rule_categories=routing["rule_categories"],
            fallback_triggered=self._last_fallback_triggered,
        )

        # Layer 3: Execution Resource
        raw_mask, evidence_list = self.execution_layer.run(
            action_resources=action_resources,
            context=context,
            routing=routing,
            skeleton_text=skeleton_text,
        )

        # Alignment Check
        mask, fallback_used = self.alignment_checker.check(
            raw_mask=raw_mask,
            action_resources=action_resources,
        )
        self._last_fallback_triggered = fallback_used

        # Record evidence
        output = self.evidence_recorder.record(
            contingency_id=context.get("contingency_id", "unknown"),
            mask=mask,
            fallback_used=fallback_used,
            evidence_list=evidence_list,
        )

        logger.info(
            "Mask generated: %d/%d resources retained, fallback=%s",
            sum(mask), len(mask), fallback_used,
        )
        return mask, output["evidence"]
