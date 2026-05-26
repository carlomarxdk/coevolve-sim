[![DOI](https://zenodo.org/badge/1113345522.svg)](https://doi.org/10.5281/zenodo.17875304)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Email](https://img.shields.io/badge/Email-g.savcisens@northeastern.edu-orange)](mailto:g.savcisens@northeastern.edu)

# CoevolveSim

This repository accompanies the  manuscript on belief diffusion in social networks of generalist and specialist large language models.

The codebase contains:
- simulation runs for belief-updating dynamics,
- notebook workflows for exploratory analysis and manuscript figure/result replication,
- surrogate-model fitting and evaluation pipelines.

## Quick Start

This project uses uv.

Install dependencies:

    uv sync

Run the main simulation entrypoint:

    uv run python src/experiment.py

Example run with Hydra overrides:

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

## Data Locations

- Simulation outputs: data/outputs/runs/zeroshot
- Dynamics/surrogate data and figures: data/outputs/dynamics
- Derived analysis summaries: data/analysis

## Notebooks (Paper and Exploration)

- notebooks/X1_data.ipynb: run/agent data loading, validation, aggregation.
- notebooks/X2_agent_analysis.ipynb: agent-level analysis.
- notebooks/X3_run_analysis.ipynb: run-level analysis and aggregates.
- notebooks/X4_manuscript_plots.ipynb: manuscript plot replication.
- notebooks/X5_surrogates.ipynb: surrogate result analysis.
- notebooks/20260101_maximin_selection.ipynb: maximin selection of graphs/statements.

## Surrogate Models

Surrogate-model code is in:

- src/analysis/dynamics_model_fitting

Key scripts include transition fitting, full-trajectory fitting, and trajectory behavior evaluation.

