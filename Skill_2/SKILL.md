---
name: context-assembler-skill
description: >
  Use this skill whenever a local knowledge subgraph (from the KG-Retrieval Skill)
  and a real-time grid operational state need to be merged into the unified structured
  context object C that the Power LLM Agent Skill expects. Triggers: before every call
  to Power LLM Agent Skill, when bridging PSS/E or PandaPower simulation output to
  LLM prompt format, or when the training-stage indicator φ must be injected into the
  context. Do NOT use for subgraph retrieval, mask generation, or PPO training.
license: Apache-2.0
version: 1.0.0
---

# Structured Dispatch Context Assembler Skill

## Overview

Fuses a local knowledge subgraph (output of the KG-Retrieval Skill) with real-time
grid operational data and RL training metadata into a single **structured context
object `C`**. This object is the unified input interface for the Power LLM Agent
Skill and eliminates the format mismatch between raw simulation data and LLM prompts.

## What It Does

1. **Operational state injection** — Parses real-time power-flow data (line loading
   ratios, bus voltages, overloaded lines) into standardised fields.
2. **Controllable resource enumeration** — Queries the simulation environment or
   SCADA interface for the current controllable resource list with per-device
   regulation margins.
3. **Subgraph knowledge embedding** — Embeds the triple-based knowledge and procedure
   entry texts from Skill 1 as structured fields, preserving node-type labels.
4. **Training-stage indicator injection** — Writes `φ` (float in [0, 1]) into the
   context to allow the downstream skeleton-selection layer to choose the appropriate
   reasoning mode.
5. **Schema validation** — Validates the assembled `C` against
   `schemas/output_context_schema.json` before returning.

## Source Hierarchy

```
context_assembler_skill/
├── SKILL.md                               ← you are here
├── core/
│   ├── state_parser.py                    ← PSS/E / PandaPower state parsing
│   ├── resource_enumerator.py             ← controllable resource enumeration
│   ├── context_builder.py                 ← main assembly logic
│   └── schema_validator.py                ← output schema validation
├── schemas/
│   ├── input_subgraph_schema.json         ← Skill 1 output (input)
│   ├── input_grid_state_schema.json       ← real-time grid state (input)
│   └── output_context_schema.json         ← structured context C (output)
├── configs/
│   └── assembler_config.yaml
└── tests/
    └── test_context_builder.py
```

## When to Use

- **Before every invocation** of the Power LLM Agent Skill.
- When adapting different simulator outputs (PSS/E `.raw`, PandaPower `net`,
  Matpower `.mat`) to the common context format.
- During **offline batch assembly**: pre-generate and cache `C` objects for all
  contingency scenarios in the training set.

## Usage

```python
from context_assembler_skill.core.context_builder import ContextBuilder

builder = ContextBuilder.from_config("context_assembler_skill/configs/assembler_config.yaml")
context = builder.build(
    subgraph=subgraph,        # output of KG-Retrieval Skill
    grid_state=grid_state,    # dict from state_parser or simulation env
    phi=0.3,                  # training-stage indicator ∈ [0, 1]
)
# context conforms to schemas/output_context_schema.json
```

## Output format (`schemas/output_context_schema.json`)

```json
{
  "contingency_id": "L12",
  "overloaded_lines": ["L12"],
  "bus_voltage": {"BUS_12": 0.97},
  "line_loading": {"L12": 1.18},
  "resource_list": [
    {
      "id": "G5",
      "type": "generator",
      "current_output_mw": 180,
      "regulation_margin_mw": 70,
      "status": "online"
    }
  ],
  "subgraph": { "...": "local knowledge subgraph from Skill 1" },
  "phi": 0.3
}
```
