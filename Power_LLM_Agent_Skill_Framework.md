# Power LLM Agent Skill — Multi-Skill Framework Design

> **Open-Source Skill Framework for the LLM-KM-RL Paper**
> Associated paper: *LLM-KM-RL: A Large-Language-Model-Enabled Knowledge-Mask Reinforcement Learning Framework for Post-Contingency Power Flow Dispatch*

---

## Framework Overview

This framework consists of **four collaborative Skills** that cover the complete pipeline from raw knowledge graph to reinforcement learning policy generation. Each Skill can be used independently or chained together in the following sequence:

```
[Contingency Signal]
        │
        ▼
┌─────────────────────────────────┐
│  Skill 1: KG-Retrieval Skill    │  Multi-hop retrieval → Local knowledge subgraph
└──────────────┬──────────────────┘
               │  Local Knowledge Subgraph
               ▼
┌─────────────────────────────────┐
│  Skill 2: Context Assembler     │  Structured context C construction
└──────────────┬──────────────────┘
               │  Structured Context C
               ▼
┌─────────────────────────────────┐
│  Skill 3: Power LLM Agent Skill │  Three-layer reasoning → Knowledge mask m
└──────────────┬──────────────────┘
               │  Knowledge Mask m + Evidence E
               ▼
┌─────────────────────────────────┐
│  Skill 4: KM-PPO Integrator     │  Mask-embedded PPO policy learning
└─────────────────────────────────┘
```

All inter-Skill interfaces are defined as standard JSON schemas, supporting both standalone invocation and fully orchestrated pipelines.

---

---

## Skill 1 — KG-Retrieval Skill

### Title
**Contingency Knowledge Graph Multi-Hop Retrieval Skill**

### Description
Performs multi-hop retrieval over a pre-built power dispatch knowledge graph in response to a detected contingency event. Extracts topology nodes, operating-procedure entries, and equipment-state nodes that are directly relevant to the current contingency scenario, and outputs a lightweight local knowledge subgraph for use by the downstream Context Assembler Skill.

### What It Does
1. **Contingency trigger parsing** — Receives a contingency identifier (faulted line or bus) together with a real-time operational snapshot; identifies the set of primary equipment affected by the fault.
2. **Multi-hop graph traversal** — Starting from the affected equipment nodes, performs breadth-first search over the knowledge graph for a configurable number of hops (default: 2–3), collecting neighboring topology nodes (buses, transformers, generators, loads).
3. **Procedure entry association** — Maps topology nodes to operating-procedure edges in the knowledge graph and extracts the dispatch-rule entries applicable to the current contingency category.
4. **Subgraph pruning** — Filters redundant nodes by relevance score to ensure the subgraph size remains within the downstream LLM context-window budget.
5. **Output serialization** — Serializes the local subgraph as structured JSON, containing node types, relation types, and procedure-text triples.

### Source Hierarchy
```
kg_retrieval_skill/
├── SKILL.md                       # Entry point (this file)
├── core/
│   ├── graph_loader.py            # KG loading and indexing (Neo4j / NetworkX)
│   ├── multi_hop_bfs.py           # Multi-hop BFS retrieval logic
│   ├── relevance_scorer.py        # Node/edge relevance scoring (TF-IDF + graph distance)
│   └── subgraph_pruner.py         # Pruning and token-budget control
├── schemas/
│   ├── input_schema.json          # Input: contingency signal + operational snapshot
│   └── output_schema.json         # Output: local knowledge subgraph JSON
├── configs/
│   └── retrieval_config.yaml      # Hop count, pruning threshold, token budget
├── tests/
│   └── test_bfs_retrieval.py
└── examples/
    └── ieee118_contingency_example.json
```

### File Structure
| File | Responsibility |
|------|----------------|
| `graph_loader.py` | Loads the KG and builds an in-memory index; supports offline NetworkX graphs and online Neo4j instances |
| `multi_hop_bfs.py` | Core retrieval logic with configurable hop count and edge-type filters |
| `relevance_scorer.py` | Scores each retrieved node by combining textual similarity and graph topological distance |
| `subgraph_pruner.py` | Sorts nodes by score and trims to a target node count or token budget |
| `input_schema.json` | Defines the input format: contingency ID, affected node list, and operational state vector |
| `output_schema.json` | Defines the output format: subgraph node list, triples, and procedure entry texts |

### When to Use
- When the system detects an **N-1 or N-k contingency** and must rapidly locate the relevant knowledge scope.
- As the **mandatory prerequisite step** before invoking the Power LLM Agent Skill (Skill 3), to prune irrelevant knowledge before LLM reasoning.
- Whenever the full knowledge graph **exceeds the LLM context-window limit** — subgraph extraction is required before passing knowledge to Skill 3.
- Supports both **offline batch retrieval** (pre-caching subgraphs for training) and **online real-time retrieval** (on-the-fly invocation during inference).

### Design Intent
The core design principle of this Skill is to **decouple knowledge localization from knowledge reasoning**. A power dispatch knowledge graph typically contains thousands of nodes; feeding it entirely to an LLM would exceed context limits and introduce excessive noise. Multi-hop BFS retrieval narrows the reasoning scope to the local neighborhood most relevant to the current contingency, thereby maximizing the signal-to-noise ratio of the LLM input while preserving completeness of critical knowledge coverage. The relevance scorer fuses textual semantic distance with graph topological distance, ensuring that critical procedure entries are not missed while irrelevant equipment information is filtered out.

---

---

## Skill 2 — Context Assembler Skill

### Title
**Structured Dispatch Context Assembler Skill**

### Description
Receives the local knowledge subgraph from the KG-Retrieval Skill and combines it with real-time grid operational data (power flow readings, overload information, and the list of controllable resources) to produce a single structured context object `C`. This object serves as the unified input interface for the Power LLM Agent Skill, bridging the gap between raw simulation data and LLM prompt formatting.

### What It Does
1. **Operational state injection** — Parses a real-time power-flow snapshot (line loading ratios, bus voltages, overloaded line set) into standardised fields and attaches them to the context object.
2. **Controllable resource enumeration** — Queries the dispatch database or SCADA interface to retrieve the current list of controllable resources (generator output adjustment, load shedding, reactive compensation, etc.) and annotates each resource with its regulation margin and current status.
3. **Subgraph knowledge embedding** — Embeds the triple-based knowledge and procedure entry texts from Skill 1 as structured fields within the context, preserving node-type labels for use by the downstream routing layer.
4. **Training-stage indicator injection** — Writes the stage indicator `φ` (exploratory / converging / anomaly-recovery) into the context based on the current RL training progress or policy state, so that Skill 3's instruction skeleton layer can select the appropriate reasoning mode.
5. **Schema validation** — Runs a JSON Schema check on the assembled `C` to guarantee that downstream Skills can parse it without errors.

### Source Hierarchy
```
context_assembler_skill/
├── SKILL.md
├── core/
│   ├── state_parser.py              # Real-time state parsing (PSS/E, PandaPower)
│   ├── resource_enumerator.py       # Controllable resource enumeration and margin computation
│   ├── context_builder.py           # Main assembly logic: subgraph + state + resources + φ
│   └── schema_validator.py          # Output schema validation
├── schemas/
│   ├── input_subgraph_schema.json   # Skill 1 output subgraph format (input)
│   ├── input_grid_state_schema.json # Real-time grid state format (input)
│   └── output_context_schema.json   # Structured context C format (output)
├── configs/
│   └── assembler_config.yaml        # Field mappings, φ threshold configuration
├── tests/
│   └── test_context_builder.py
└── examples/
    └── assembled_context_example.json
```

### File Structure
| File | Responsibility |
|------|----------------|
| `state_parser.py` | Parses PSS/E `.raw` files or PandaPower `net` objects; extracts overloaded lines and bus-state fields |
| `resource_enumerator.py` | Queries SCADA or the simulation environment; returns the controllable resource list with per-device regulation margin `Δ_r` |
| `context_builder.py` | Assembles all inputs into `C` according to `output_context_schema.json` |
| `schema_validator.py` | Validates `C` using `jsonschema`; raises field-located exceptions on failure |
| `output_context_schema.json` | Defines the complete field structure of `C`; acts as the interface contract between Skill 2 and Skill 3 |

### When to Use
- **Before every invocation of the Power LLM Agent Skill**, to complete context assembly.
- When there is a **data-format mismatch between the simulation environment and the LLM interface** (e.g., PandaPower objects ↔ JSON), acting as the format adaptation layer.
- When the training process requires **dynamic switching of the reasoning mode** (exploratory / converging), relying on the `φ` field injected by this Skill.
- Supports **offline batch assembly** (pre-generating context caches for historical contingency scenarios) to accelerate training throughput.

### Design Intent
The motivation for this Skill is to **separate data-fusion complexity from LLM prompt engineering**. In the LLM-KM-RL framework, context `C` is the information foundation for all subsequent reasoning, and its completeness and structural consistency directly affect the quality of the generated knowledge mask. Encapsulating assembly logic as a standalone Skill makes it straightforward to develop dedicated adapters for different power-system simulators (PSS/E, PandaPower, Matpower) while keeping the Skill 3 prompt templates stable and independent of data-source changes. The `φ` field design allows the same Skill 3 to adopt different reasoning skeletons during early and late training phases, dynamically balancing exploration breadth against the strictness of safety constraints.

---

---

## Skill 3 — Power LLM Agent Skill ⭐

### Title
**Context-Aware Power LLM Agent Skill for Knowledge-Mask Generation**

### Description
The central Skill of the framework. Receives the structured context `C` and, through a three-layer architecture — **Dynamic Routing Layer → Instruction Skeleton Layer → Execution Resource Layer** — performs dispatch-rule-level eligibility reasoning for each controllable resource. Outputs a binary knowledge mask `m` whose dimensions are aligned with the RL action space, together with a traceable reasoning evidence record `E`. This Skill converts unstructured dispatch-rule reasoning into action-space-aligned constraints that can be directly consumed by the policy learning algorithm.

### What It Does

**Overall flow: `C` → [Dynamic Routing → Instruction Skeleton → Execution Resource] → `(m̃, E)`**

**Layer 1 · Dynamic Routing Layer**
- Parses the contingency state, overload information, and candidate controllable resources from `C` to determine the knowledge scope and dispatch-rule category required for the current reasoning process.
- Focuses subsequent reasoning on information relevant to the current grid operating condition, filtering out irrelevant knowledge branches.
- Output: a set of knowledge-scope labels (e.g., "N-1 thermal constraint", "spinning-reserve procedure").

**Layer 2 · Instruction Skeleton Layer**
- Selects a reasoning skeleton based on the stage indicator `φ`:
  - **Exploratory skeleton `S_exp`**: Used during early training or low-risk operating states — retains more candidate controllable resources to reduce the risk of over-constraining.
  - **Conservative skeleton `S_con`**: Used under severe overload, late-stage policy refinement, or anomalous mask-generation cases — strengthens security rules, dispatch-rule evidence, and resource-activation priorities.
- The skeleton is not a simple LLM preference setting; it specifies the **resource-eligibility checking logic** used in mask generation.

**Layer 3 · Execution Resource Layer**
- Performs a three-condition eligibility check for each controllable resource `r_i` in the RL action space:
  - `C1(r_i)`: Does the resource belong to the candidate controllable resource set selected by the knowledge subgraph?
  - `C2(r_i)`: Does the resource still have sufficient regulation margin?
  - `C3(r_i)`: Does the resource satisfy the currently loaded dispatch rules?
- If all three conditions are met: `m̃_i = 1` (retained). Otherwise: `m̃_i = 0` (masked).
- Simultaneously records the judgment rationale for each resource, forming traceable reasoning evidence `E`.

**Output Alignment Verification**
- Verifies that `m̃` can be parsed as a valid vector of dimension `|A_RL|` with one-to-one correspondence to the action resource set.
- If verification fails, activates the fallback mask (all-ones vector `m_default`) to prevent an anomalous LLM output from completely blocking the action space.

**Formal Expression**

```
(m̃, E) = PowerLLMAgentSkill(C, A_RL, φ)

∀ r_i ∈ A_RL:
  m̃_i = 1  iff  C1(r_i) ∧ C2(r_i) ∧ C3(r_i)
  m̃_i = 0  otherwise

m = AlignCheck(m̃, A_RL) ? m̃ : m_default
```

### Source Hierarchy
```
power_llm_agent_skill/
├── SKILL.md                           # Entry point (this file)
├── core/
│   ├── dynamic_routing.py             # Routing layer: knowledge scope and rule-category identification
│   ├── skeleton_selector.py           # Skeleton layer: S_exp / S_con selection based on φ
│   ├── execution_resource.py          # Resource layer: per-resource three-condition eligibility logic
│   ├── alignment_checker.py           # Output alignment check + fallback mask logic
│   └── evidence_recorder.py           # Structured recording and serialization of reasoning evidence E
├── prompts/
│   ├── routing_prompt.jinja2          # Dynamic routing layer prompt template
│   ├── skeleton_exploratory.jinja2    # Exploratory skeleton S_exp instruction template
│   ├── skeleton_conservative.jinja2   # Conservative skeleton S_con instruction template
│   └── resource_eligibility.jinja2    # Per-resource eligibility reasoning prompt template
├── schemas/
│   ├── input_context_schema.json      # Interface contract with Skill 2 (input C)
│   └── output_mask_schema.json        # Knowledge mask m + evidence E output format
├── configs/
│   ├── llm_config.yaml                # LLM backend selection, temperature, max_tokens, etc.
│   └── skill_config.yaml              # φ thresholds, fallback policy, skeleton switching logic
├── tests/
│   ├── test_routing.py
│   ├── test_skeleton_selection.py
│   ├── test_resource_eligibility.py
│   └── test_alignment_checker.py
├── examples/
│   ├── mask_generation_example.json   # Full input-to-output example
│   └── evidence_example.json          # Reasoning evidence example
└── notebooks/
    └── skill_walkthrough.ipynb        # Interactive demonstration notebook
```

### File Structure

| File | Responsibility |
|------|----------------|
| `dynamic_routing.py` | Parses the contingency type and overload info from `C`; returns knowledge-scope labels used to focus the routing prompt |
| `skeleton_selector.py` | Selects `S_exp` or `S_con` based on the `φ` value (float or enum); injects the corresponding eligibility-checking logic description into the LLM |
| `execution_resource.py` | Main reasoning loop: iterates over `A_RL`, invokes the eligibility prompt for each `r_i`, and parses the binary result |
| `alignment_checker.py` | Verifies that `m̃` dimensions match `A_RL`; returns `m_default` and logs a warning if parsing fails |
| `evidence_recorder.py` | Serializes the three-condition judgment results (`C1/C2/C3`) and LLM source text for each resource into structured `E` |
| `routing_prompt.jinja2` | Routing-layer prompt template; variables: `{contingency_info}`, `{overload_info}`, `{resource_list}` |
| `skeleton_exploratory.jinja2` | Exploratory skeleton template; emphasizes retaining candidate resources with reasonable uncertainty tolerance |
| `skeleton_conservative.jinja2` | Conservative skeleton template; enforces safety-rule priority and full evidence chains |
| `resource_eligibility.jinja2` | Per-resource prompt; variables: `{resource_id}`, `{resource_state}`, `{dispatch_rules}`, `{skeleton}` |
| `llm_config.yaml` | Supports GPT-4o / Claude / domain-tuned power-LLM backends |
| `output_mask_schema.json` | Defines `m` (int array) and `E` (list of dicts) in the complete output structure |

### When to Use
- **Before every RL training step (or inference step)**, prior to action sampling by the PPO policy network, to obtain the knowledge mask for that step.
- When **RL exploration drifts into action regions disallowed by equipment states** (e.g., adjusting a generator already at its limit), to preemptively block those actions via the knowledge mask.
- When dispatch procedures contain **unstructured natural-language rules** (e.g., "prohibit voltage reduction on transformers above 220 kV under heavy overload") that cannot be directly encoded as mathematical constraints.
- Use `S_exp` during **early training** (high uncertainty) to preserve sufficient exploration space; switch to `S_con` during **late training or anomalous states** to enforce stricter safety constraints.
- Supports **offline batch pre-generation of masks** (for an entire training dataset) to reduce online LLM call latency.

### Design Intent
The core design philosophy of the Power LLM Agent Skill is: **the LLM output is not an open-ended dispatch recommendation — it is a binary constraint vector strictly aligned with the RL action space dimensions**.

This design addresses two fundamental challenges in applying LLMs to RL policy learning:

**1. Operationalizing unstructured knowledge.** Power dispatch procedures are written in natural language and contain extensive conditional logic and equipment-state dependencies that conventional methods struggle to encode directly as optimization constraints. The three-layer architecture progressively decomposes the problem: dynamic routing identifies *which knowledge is relevant*, skeleton selection determines *what reasoning logic to apply*, and resource eligibility judgment decides *which actions are permissible*. This cascade transforms ambiguous procedure text into a precise boolean mask.

**2. Engineering control over output uncertainty.** LLM outputs carry inherent formatting instability. Using them directly as RL action-space constraints is hazardous. The alignment check combined with the fallback mechanism ensures that even under LLM hallucination or formatting errors, RL training does not crash due to a completely blocked action space. The `m_default` all-ones vector degrades an anomaly to "unconstrained exploration" rather than "system failure".

The dual-skeleton design (`S_exp` / `S_con`) explicitly models the **exploration–exploitation trade-off in RL**, coupling the LLM reasoning mode to the training stage dynamically rather than relying on static prompt engineering.

---

---

## Skill 4 — KM-PPO Integrator Skill

### Title
**Knowledge-Mask Embedded Proximal Policy Optimization Integrator Skill**

### Description
Embeds the knowledge mask `m` generated by the Power LLM Agent Skill into both the policy generation and value estimation modules of the standard PPO algorithm, implementing the knowledge-mask-embedded Proximal Policy Optimization (KM-PPO) algorithm. Enables the RL agent to make decisions within a knowledge-constrained continuous action subspace while reducing exploration in invalid action regions.

### What It Does
1. **Mask–policy coupling** — During the action-sampling phase of the policy network, applies the knowledge mask `m` to the action distribution, suppressing the action dimensions corresponding to resources with `m_i = 0`, and confining sampling to the knowledge-approved continuous action subspace.
2. **Mask–value coupling** — Injects mask information into the value network's state-value estimation so that the value function is aware of the feasible action set under the current knowledge constraints, improving the accuracy of advantage estimation.
3. **PPO constraint adaptation** — In the PPO clip-update step, zeros out or scales the gradients of action dimensions masked by `m_i = 0`, preventing policy updates from being polluted by gradients originating in knowledge-infeasible regions.
4. **Mask cache management** — Maintains a cache mapping training steps to masks, supporting a hybrid mode that combines offline pre-generated masks with online real-time masks.
5. **Training monitoring metrics** — Records per-step statistics including the fraction of masked action dimensions, fallback trigger count, and effective action-space dimensionality, for training analysis and interpretability.

### Source Hierarchy
```
km_ppo_integrator_skill/
├── SKILL.md
├── core/
│   ├── masked_policy_network.py      # Mask–policy coupling: action-distribution clipping
│   ├── masked_value_network.py       # Mask–value coupling: knowledge-aware value estimation
│   ├── km_ppo_trainer.py             # KM-PPO main training loop (clip + masked gradient handling)
│   ├── mask_cache_manager.py         # Mask cache: offline pre-generation + online real-time hybrid
│   └── training_monitor.py           # Mask-related training metrics logging and visualization
├── networks/
│   ├── policy_net.py                 # Base policy network definition (MLP / GNN selectable)
│   └── value_net.py                  # Base value network definition
├── configs/
│   ├── ppo_config.yaml               # clip_ratio, lr, gamma, GAE-λ, and other standard PPO params
│   └── mask_integration_config.yaml  # Mask embedding mode, gradient handling, cache configuration
├── schemas/
│   ├── input_mask_schema.json        # Skill 3 output mask format (input)
│   └── training_log_schema.json      # Training log format
├── tests/
│   ├── test_masked_sampling.py
│   ├── test_value_estimation.py
│   └── test_gradient_zeroing.py
└── examples/
    └── km_ppo_training_example.py    # Full training example on IEEE 118-bus system
```

### File Structure

| File | Responsibility |
|------|----------------|
| `masked_policy_network.py` | Applies the mask to the mean and standard deviation of the continuous Gaussian action distribution; injects a strong penalty or direct zero-out for `m_i = 0` dimensions |
| `masked_value_network.py` | Concatenates mask `m` to the state feature vector so the value function is informed of the feasible action set size |
| `km_ppo_trainer.py` | Extended PPO training loop; calls Skill 3 before each rollout to obtain the mask for that step |
| `mask_cache_manager.py` | Keys the cache on `(episode_id, step_id)`; supports HDF5 offline storage and in-memory LRU caching |
| `training_monitor.py` | Logs real-time metrics: effective action-dimension ratio, knowledge-constraint violation rate, fallback trigger rate |
| `ppo_config.yaml` | Standard PPO hyperparameter configuration; compatible with Stable-Baselines3 / RLlib format |
| `mask_integration_config.yaml` | Defines mask embedding mode (`hard-mask` / `soft-penalty`), gradient handling (`zero` / `scale`), and cache strategy |

### When to Use
- As the **terminal training module** of the LLM-KM-RL framework, invoked after Skills 1–3 have been configured.
- When the RL agent exhibits **excessive invalid exploration** in a large continuous action space (many controllable resources), and a knowledge mask is needed to narrow the feasible action subspace.
- In scenarios where **physical constraints and operating procedures cannot be expressed analytically**, using the knowledge mask as a soft-constraint alternative.
- Supports **integration with existing PPO implementations** (Stable-Baselines3, RLlib, CleanRL) via a wrapper injection pattern, without requiring a full rewrite of the training framework.
- In **interpretability-required settings**, combines Skill 3's reasoning evidence `E` with this Skill's training logs to provide a full knowledge-traceable chain for each action decision.

### Design Intent
The KM-PPO Integrator Skill is designed to **embed knowledge constraints into standard PPO with minimal algorithmic intrusion** while preserving the algorithm's theoretical convergence properties.

Applying the mask **simultaneously** to both the policy and the value modules — rather than only at action-sampling time — is the key architectural decision. If the mask were applied only during sampling, the value function would still produce incorrect value estimates for masked-out regions, leading to biased advantage computation. Injecting mask information into the value network as well ensures that both modules share a consistent view of the feasible action set, eliminating the policy–value inconsistency that would otherwise destabilize training.

The configurable gradient handling strategy (`zero` vs. `scale`) provides flexible control over constraint enforcement strength at different training stages: `scale` mode during early training retains faint gradient signals from masked dimensions to preserve learning flexibility; `zero` mode during late training enforces strict hard constraints. This mirrors the skeleton-switching mechanism in Skill 3 and jointly implements **dynamic constraint-strength scheduling** across the full training horizon.

---

---

## Cross-Skill Interface Contracts

```
Skill 1 → Skill 2:  local_subgraph.json    { nodes[], triples[], procedure_texts[] }
Skill 2 → Skill 3:  context_C.json         { subgraph, grid_state, resource_list, φ }
Skill 3 → Skill 4:  knowledge_mask.json    { m: int[], evidence: E[{r_id, C1, C2, C3, text}] }
Skill 3 → Skill 4:  fallback triggered     { m = ones(|A_RL|), warning_logged: true }
```

All interface formats are defined as JSON Schemas stored in the `schemas/` directory of each Skill, ensuring cross-language and cross-platform interoperability.

---

## Citation

If you use this Skill framework, please cite:

```
[Citation information to be added upon formal publication]
```

---

*Framework Version: 1.0.0 | License: Apache-2.0*
