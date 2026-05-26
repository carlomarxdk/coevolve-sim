# Agent Catalog Configuration

This directory defines how **agent populations** are specified and instantiated in experiments.

Agent catalogs are **declarative**:
YAML files describe *intent*, while Python code deterministically materializes the final list of agents at runtime
(see `build_catalog` in `utils_hydra.py`).

---

## Overview

Each experiment specifies a `catalog` with one of the following modes:

* **`procedural`** — generate many agents programmatically
* **`explicit`** — manually list agents

This allows us to scale populations while still supporting heterogeneous or special roles.

---

## Common Fields

```yaml
catalog:
  mode: procedural | explicit

  role_templates: {} 
  counts: {}
  explicit: []

  random_roles: true | false
  random_roles_spec: {}   # role -> count

name: str
```

---

## Field Descriptions

* **`mode`**
  Controls how agents are constructed.

* **`role_templates`**
  Reusable agent templates used in procedural generation.

* **`counts`**
  Number of agents to generate per role template.

* **`explicit`**
  Manually specified agents (used when `mode: explicit`).

* **`random_roles`**
  If `true`, agent roles are reassigned randomly at runtime.

* **`random_roles_spec`**
  A procedural specification of valid roles and their exact counts
  used when `random_roles: true`. It will essentially ignore the role specification in the `role_template`

---

## Example 1 — Procedural Catalog

Generate **32 identical human participants** using the same model and prompt.

```yaml
# @package agents

catalog:
  mode: procedural
  random_roles: false

  role_templates:
    LLM:
      name: llama-base
      role: "Human Participant"
      prompt: ${prompt}

  counts:
    LLM: 32

  explicit: []

  random_roles_spec: {}

name: example_procedural
```

---

## Example 2 — Explicit Catalog

Manually specify each agent.

```yaml
# @package agents

catalog:
  mode: explicit
  random_roles: false

  role_templates: {}
  counts: {}

  explicit:
    - id: 0
      name: llama-doc
      role: "Clinical Physician"
      prompt: ${prompt}
    - id: 1
      name: llama-base
      role: "Human Participant"
      prompt: ${prompt}
    - id: 2
      name: llama-hermes
      role: "Strategic Planner"
      prompt: ${prompt}
    
  random_roles_spec: {}

name: example_explicit
```

---

## Example 3 — Procedural with Multiple Role Templates

Generate a heterogeneous population using multiple templates.

```yaml
# @package agents

catalog:
  mode: procedural
  random_roles: false

  role_templates:
    Type_1:
      name: llama-base
      role: "Human Participant"
      prompt: ${prompt}

    Type_2:
      name: llama-doc
      role: "Clinical Physician"
      prompt: ${prompt}

  counts:
    Type_1: 3
    Type_2: 29

  explicit: []

  random_roles_spec: {}

name: example_multiple_procedurals
```

---

## Example 4 — Procedural Random Role Assignment

Roles are reassigned randomly **while preserving exact role counts**.

```yaml
# @package agents

catalog:
  mode: procedural
  random_roles: true

  role_templates:
    LLM:
      name: llama-base
      role: "Human Participant"   # initial role (overwritten)
      prompt: ${prompt}

  counts:
    LLM: 32

  explicit: []

  random_roles_spec:
    "Human Participant": 28
    "Clinical Physician": 2
    "Assistant": 2

name: example_random_roles
```

### Semantics

When `random_roles: true`:

* Roles are reassigned after catalog construction
* Assignment is deterministic given the experiment seed
* Only roles listed in `random_roles_spec` may be assigned
* The sum of all role counts **must equal the number of agents**

Violations cause catalog construction to fail.

---

## Important Notes

* **Agent IDs are assigned automatically and deterministically**
  Do **not** hard-code IDs unless strictly necessary.

* YAML files describe **what** the population should look like,
  not **how** it is constructed.

* Always log both:

  1. the **catalog specification** (YAML)
  2. the **materialized catalog** (runtime output)

This ensures experiments are reproducible and auditable.

