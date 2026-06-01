---
name: power-llm-agent-skill
description: >
  Use this skill whenever a structured context C (from the Context Assembler Skill)
  must be converted into a binary knowledge mask m aligned with the RL action space.
  Triggers: before every PPO action-sampling step during training or inference,
  when unstructured dispatch-rule text must be converted into actionable boolean
  constraints, or when LLM-generated dispatch reasoning needs to be grounded to
  specific controllable resource dimensions. Do NOT use for KG retrieval, context
  assembly, or PPO training — those are handled by Skills 1, 2, and 4 respectively.
license: Apache-2.0
version: 1.0.0
---

# Context-Aware Power LLM Agent Skill for Knowledge-Mask Generation

## Overview

The central Skill of the LLM-KM-RL framework. Receives the structured context `C`
and, through a **three-layer architecture**, performs dispatch-rule-level eligibility
reasoning for every controllable resource in the RL action space. Outputs a binary
knowledge mask `m` whose dimensions are aligned one-to-one with the action space,
together with a traceable reasoning evidence record `E`.

This Skill converts *unstructured dispatch-rule reasoning into action-space-aligned
constraints* for policy learning. The LLM output is not an open-ended dispatch
recommendation — it is a validated binary vector that the PPO trainer can immediately
apply.

## Three-Layer Architecture

```
Structured Context C
        │
        ▼
┌───────────────────────────┐
│  Layer 1: Dynamic Routing  │  Identifies knowledge scope and rule categories
└──────────────┬────────────┘
               │  scope labels
               ▼
┌───────────────────────────┐
│  Layer 2: Instruction      │  Selects S_exp or S_con based on φ
│           Skeleton         │
└──────────────┬────────────┘
               │  skeleton + focused context
               ▼
┌───────────────────────────┐
│  Layer 3: Execution        │  Per-resource eligibility: C1 ∧ C2 ∧ C3
│           Resource         │
└──────────────┬────────────┘
               │  raw mask m̃ + evidence E
               ▼
┌───────────────────────────┐
│  Alignment Check           │  Dimension validation + fallback to m_default
└──────────────┬────────────┘
               │
               ▼
        (m, E)  →  Skill 4
```

### Layer 1 — Dynamic Routing Layer

Parses `C` to identify the **knowledge scope** and **dispatch-rule category** relevant
to the current contingency. Focuses subsequent reasoning on the relevant knowledge
branch and suppresses irrelevant subgraph sections.

Output: a set of scope labels, e.g. `["N-1_thermal", "spinning_reserve"]`.

### Layer 2 — Instruction Skeleton Layer

Selects the reasoning skeleton based on the stage indicator `φ ∈ [0, 1]`:

| Condition | Skeleton | Behaviour |
|-----------|----------|-----------|
| `φ ≤ phi_exp_threshold` (default 0.4) | `S_exp` — Exploratory | Retains more candidate resources; tolerates uncertainty |
| `φ > phi_exp_threshold` | `S_con` — Conservative | Enforces security rules strictly; requires evidence chains |

The skeleton is injected into the resource-eligibility prompt as an explicit
*checking logic* specification, not merely a tone preference.

### Layer 3 — Execution Resource Layer

For each resource `r_i` in the RL action space, performs a **three-condition
eligibility check**:

```
C1(r_i): r_i ∈ candidate set selected by knowledge subgraph?
C2(r_i): r_i has sufficient regulation margin?
C3(r_i): r_i satisfies currently loaded dispatch rules?

m̃_i = 1  iff  C1 ∧ C2 ∧ C3
m̃_i = 0  otherwise
```

Simultaneously records `{r_id, C1, C2, C3, source_text}` as reasoning evidence `E`.

### Alignment Check

Verifies that `m̃` can be parsed as a valid integer vector of length `|A_RL|` with
one-to-one correspondence to the action resource set. On failure, returns the
fallback all-ones mask `m_default` and logs a warning.

## Formal Expression

```
(m̃, E) = PowerLLMAgentSkill(C, A_RL, φ)

m = AlignCheck(m̃, A_RL) ? m̃ : m_default
```

## Source Hierarchy

```
power_llm_agent_skill/
├── SKILL.md                              ← you are here
├── core/
│   ├── dynamic_routing.py               ← Layer 1: scope + rule-category identification
│   ├── skeleton_selector.py             ← Layer 2: S_exp / S_con selection
│   ├── execution_resource.py            ← Layer 3: per-resource eligibility loop
│   ├── alignment_checker.py             ← output alignment + fallback mask
│   └── evidence_recorder.py             ← structured evidence E serialisation
├── prompts/
│   ├── routing_prompt.jinja2            ← Layer 1 prompt template
│   ├── skeleton_exploratory.jinja2      ← S_exp instruction template
│   ├── skeleton_conservative.jinja2     ← S_con instruction template
│   └── resource_eligibility.jinja2      ← per-resource eligibility template
├── schemas/
│   ├── input_context_schema.json        ← interface contract with Skill 2
│   └── output_mask_schema.json          ← knowledge mask m + evidence E
├── configs/
│   ├── llm_config.yaml                  ← LLM backend, temperature, max_tokens
│   └── skill_config.yaml                ← φ thresholds, fallback policy
├── tests/
│   ├── test_routing.py
│   ├── test_skeleton_selection.py
│   ├── test_resource_eligibility.py
│   └── test_alignment_checker.py
└── examples/
    ├── mask_generation_example.json
    └── evidence_example.json
```

## When to Use

- **Before every RL training step** (or inference step), prior to PPO action sampling.
- When RL exploration drifts into **equipment-state-disallowed action regions**.
- When dispatch procedures contain **unstructured natural-language rules** that cannot
  be encoded as explicit mathematical constraints.
- Use **`S_exp`** during early training for broad exploration; switch to **`S_con`**
  during late training or under severe overload for strict safety enforcement.
- Supports **offline batch pre-generation** of masks to reduce online LLM latency.

## Usage

```python
from power_llm_agent_skill.core.execution_resource import PowerLLMAgentSkill

skill = PowerLLMAgentSkill.from_config("power_llm_agent_skill/configs/skill_config.yaml")
mask, evidence = skill.run(
    context=context,                   # output of Context Assembler Skill
    action_resources=env.action_resources,  # list of resource IDs matching action dims
)
# mask : list[int]  — length == len(action_resources), values in {0, 1}
# evidence : list[dict]  — one entry per resource with C1/C2/C3 judgments
```

## Output format (`schemas/output_mask_schema.json`)

```json
{
  "contingency_id": "L12",
  "mask": [1, 0, 1, 1, 0],
  "fallback_used": false,
  "evidence": [
    {
      "resource_id": "G5",
      "C1_in_subgraph": true,
      "C2_margin_ok": true,
      "C3_rule_ok": true,
      "eligible": true,
      "source_text": "PROC_TH_01: G5 is in the candidate set and has 70 MW margin."
    },
    {
      "resource_id": "G7",
      "C1_in_subgraph": false,
      "C2_margin_ok": true,
      "C3_rule_ok": true,
      "eligible": false,
      "source_text": "G7 not found in local knowledge subgraph for L12."
    }
  ]
}
```
