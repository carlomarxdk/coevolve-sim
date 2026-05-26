[![DOI](https://zenodo.org/badge/1113345522.svg)](https://doi.org/10.5281/zenodo.17875304)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Email](https://img.shields.io/badge/Email-g.savcisens@northeastern.edu-orange)](mailto:g.savcisens@northeastern.edu)

# CoevolveSim

Belief coevolution in social networks of generalist and specialist large language models.

CoevolveSim is an agent-based simulation framework for studying how beliefs coevolve in interacting LLM populations with different role and expertise assignments.

The codebase contains:
- simulation runs for belief-updating dynamics,
- notebook workflows for exploratory analysis and manuscript figure/result replication,
- surrogate-model fitting and evaluation pipelines.

## Table of Contents

- [CoevolveSim](#coevolvesim)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Environment and Installation](#environment-and-installation)
  - [Run Experiments](#run-experiments)
  - [Reproducibility Notes](#reproducibility-notes)
  - [Data and Outputs](#data-and-outputs)
  - [Notebooks and Analysis](#notebooks-and-analysis)
  - [Surrogate Models](#surrogate-models)
  - [Project Structure](#project-structure)
  - [Testing](#testing)
  - [Citation](#citation)
  - [License](#license)
  - [Authors](#authors)

## Overview

Each simulation run follows a simple loop:
1. agents form initial beliefs about a statement,
2. agents receive neighbor-belief summaries from the network,
3. agents update beliefs over rounds,
4. metrics and artifacts are saved for downstream analysis.

The framework is intended to support questions such as:
- how social interaction changes LLM belief dynamics compared with isolated inference,
- when specialist/expert agents stabilize or shift collective outcomes,
- how role identity and network structure affect influence and convergence.

## Environment and Installation

This project uses uv.

Install and sync dependencies:

```bash
uv sync
```

Optional development dependencies:

```bash
uv sync --group dev
```

## Run Experiments

Canonical entrypoint:

```bash
uv run python src/experiment.py
```

Example run with Hydra overrides:

```bash
uv run python src/experiment.py \
    catalog=random_roles \
    prompt=wR_L \
    network=erdos-renyi \
    network.params.n=48 \
    network.params.p=0.3 \
    statement=false_0 \
    probe=zeroshot \
    seed=814183 \
    experiment.max_rounds=10
```

Notes:
- Some model configurations require a Hugging Face access token in `src/configs/model/*.yaml`.
- A subset of analysis notebooks relies on R/rpy2 tooling (see notebooks/r_utils.py and notebook comments for details).

## Reproducibility Notes

- Prefer explicit seeds (for example, `seed=814183`) for reruns.
- Each run writes its realized configuration and outputs to timestamped folders.
- Completed matching runs are skipped automatically; incomplete matching runs are moved before rerun.

## Data and Outputs

Primary artifacts are under `data/outputs`.

Simulation outputs:
- data/outputs/runs/zeroshot

Dynamics/surrogate outputs:
- data/outputs/dynamics

Derived analysis summaries:
- data/analysis

## Notebooks and Analysis

- `notebooks/X1_data.ipynb`: run/agent data loading, validation, aggregation.
- `notebooks/X2_agent_analysis.ipynb`: agent-level analysis.
- `notebooks/X3_run_analysis.ipynb`: run-level analysis and aggregates.
- `notebooks/X4_manuscript_plots.ipynb`: manuscript plot replication.
- `notebooks/X5_surrogates.ipynb`: surrogate result analysis.
- `notebooks/20260101_maximin_selection.ipynb`: maximin selection of graphs/statements.

Paper replication path (brief):
1. Run experiments.
2. Aggregate and clean in `notebooks/X1_data.ipynb`.
3. Run statistical analysis in `notebooks/X2_agent_analysis.ipynb` and `notebooks/X3_run_analysis.ipynb`.
4. Regenerate figures in `notebooks/X4_manuscript_plots.ipynb`.


## Surrogate Models

Surrogate-model code is in:

- `src/analysis/dynamics_model_fitting`

Key scripts include transition fitting, full-trajectory fitting, and trajectory behavior evaluation.

## Project Structure

```text
coevolve-sim/
├── pyproject.toml
├── src/
│   ├── experiment.py
│   ├── core/
│   └── analysis/
│       └── dynamics_model_fitting/
├── notebooks/
├── data/
│   ├── outputs/
│   │   ├── runs/
│   │   └── dynamics/
│   └── analysis/
└── tests/
```

## Testing

```bash
uv run pytest
```

## Citation

```bibtex
TO BE ANNOUNCED
```

## License

MIT License. See [LICENSE](LICENSE).

## Authors

Germans Savcisens, Samantha Dies, Courtney Maynard, and Tina Eliassi-Rad.

Correspondence: [g.savcisens@northeastern.edu](mailto:g.savcisens@northeastern.edu)

