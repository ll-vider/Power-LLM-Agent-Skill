# LLM-KM-RL: Knowledge-Mask Reinforcement Learning for Power Flow Dispatch

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> Official open-source Skill framework accompanying the paper:
> **"LLM-KM-RL: A Large-Language-Model-Enabled Knowledge-Mask Reinforcement Learning Framework for Post-Contingency Power Flow Dispatch"**

---

## Overview

Post-contingency power flow dispatch in renewable-rich power systems demands rapid response and strict operational security. This repository implements the **LLM-KM-RL framework**, which integrates a knowledge graph with LLM-based knowledge reasoning to convert unstructured dispatch-rule knowledge into action-space-aligned constraints for reinforcement learning policy optimization.

The framework is organized as **four modular, composable Skills**:

| Skill | Role | Key Output |
|-------|------|------------|
| [Skill 1 — KG-Retrieval](./kg_retrieval_skill/) | Multi-hop knowledge graph retrieval | Local knowledge subgraph |
| [Skill 2 — Context Assembler](./context_assembler_skill/) | Structured context construction | Unified context object `C` |
| [Skill 3 — Power LLM Agent](./power_llm_agent_skill/) | Dispatch-rule reasoning & mask generation | Knowledge mask `m` + evidence `E` |
| [Skill 4 — KM-PPO Integrator](./km_ppo_integrator_skill/) | Mask-embedded PPO policy learning | Trained dispatch policy `π` |

### Pipeline

```
Contingency Signal
      │
      ▼
 KG-Retrieval ──► Context Assembler ──► Power LLM Agent ──► KM-PPO Integrator
      │                  │                     │                    │
  Subgraph C         Context C           Mask m + E C          Policy π
```

---

## Quick Start

```bash
git clone https://github.com/your-org/llm-km-rl.git
cd llm-km-rl
pip install -r requirements.txt
```

### Minimal end-to-end example (IEEE 39-bus)

```python
from kg_retrieval_skill.core.multi_hop_bfs import KGRetriever
from context_assembler_skill.core.context_builder import ContextBuilder
from power_llm_agent_skill.core.execution_resource import PowerLLMAgentSkill
from km_ppo_integrator_skill.core.km_ppo_trainer import KMPPOTrainer

# Step 1 — Retrieve local knowledge subgraph
retriever = KGRetriever.from_config("kg_retrieval_skill/configs/retrieval_config.yaml")
subgraph = retriever.retrieve(contingency_id="L12", grid_snapshot=snapshot)

# Step 2 — Assemble structured context
builder = ContextBuilder.from_config("context_assembler_skill/configs/assembler_config.yaml")
context = builder.build(subgraph=subgraph, grid_state=grid_state, phi=0.3)

# Step 3 — Generate knowledge mask
agent_skill = PowerLLMAgentSkill.from_config("power_llm_agent_skill/configs/skill_config.yaml")
mask, evidence = agent_skill.run(context=context, action_resources=env.action_resources)

# Step 4 — Train KM-PPO policy
trainer = KMPPOTrainer.from_config("km_ppo_integrator_skill/configs/ppo_config.yaml")
policy = trainer.train(env=env, mask_provider=agent_skill)
```

---

## Requirements

```
python>=3.9
torch>=2.0
networkx>=3.0
openai>=1.0          # or anthropic / transformers for local LLM
pandapower>=2.13
jsonschema>=4.0
jinja2>=3.1
pyyaml>=6.0
stable-baselines3>=2.0
```

---

## Repository Structure

```
llm_km_rl/
├── README.md
├── requirements.txt
├── kg_retrieval_skill/
│   ├── SKILL.md
│   ├── core/
│   ├── schemas/
│   ├── configs/
│   └── examples/
├── context_assembler_skill/
│   ├── SKILL.md
│   ├── core/
│   ├── schemas/
│   └── configs/
├── power_llm_agent_skill/
│   ├── SKILL.md
│   ├── core/
│   ├── prompts/
│   ├── schemas/
│   └── configs/
└── km_ppo_integrator_skill/
    ├── SKILL.md
    ├── core/
    ├── networks/
    ├── schemas/
    └── configs/
```

---

## Citation

```bibtex
@article{llm_km_rl_2025,
  title   = {LLM-KM-RL: A Large-Language-Model-Enabled Knowledge-Mask Reinforcement
             Learning Framework for Post-Contingency Power Flow Dispatch},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2025}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
