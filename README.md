[![DOI](https://zenodo.org/badge/1113345522.svg)](https://doi.org/10.5281/zenodo.17875304)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Email](https://img.shields.io/badge/Email-g.savcisens@northeastern.edu-orange)](mailto:g.savcisens@northeastern.edu)

# CoevolveSim

Code and simulation data for **"Belief Coevolution in a Social Network of Generalist and Specialist Large Language Models"** TBA.

`CoevolveSim` is a framework for studying belief diffusion within networked LLM populations. Generalist and specialist LLM agents are placed on a social network (Erdős–Rényi or Watts–Strogatz) and exchange beliefs about medical-indication statements over several rounds, each agent revising its belief after observing a summary of its neighbors' beliefs. Across four scenarios:
-  **I.** baseline generalists, 
-  **II.** generalists with random social roles, 
-  **III.** specialists with random roles, 
-  **IV.** specialists with roles matched to their domain.

These simulations isolate the effects of persona-style role assignment, domain specialization (model heterogeneity), and role–specialization alignment on individual belief revision and population-level consensus. A hierarchy of classical opinion-dynamics surrogate models (M1–M4) is then fit to test which mechanisms (persistence, social belief composition, agent identity) are needed to reproduce the observed dynamics.

## Project structure

1. `src`:  simulation framework, agent/network configs, and analysis code (installed as an editable package).
2. `tests`: unit tests.
3. `data`: raw simulation output (`data/outputs/`) and derived analysis tables/figures (`data/analysis/`).
4. `notebooks`: cleaned, documented notebooks that reproduce every table and figure in the paper; see [Notebooks](#notebooks) below.

## Notebooks

`notebooks` contains the documented, reproducible pipeline behind the paper's results. Each notebook's own intro cell states exactly which figures/tables/sections it produces:

1. **`maximin_selection.ipynb`**: reproduces the maximin selection of the 16 network realizations and 20 discussion statements used across all runs.
2. **`sanity_check.ipynb`**: check that all data is in place
3. **`X1_data.ipynb`**: turns raw per-run simulation output (`data/outputs/runs/`) into the two aggregated tables every later notebook builds on (`agent_level_data.parquet`, `run_level_data.parquet`).
4. **`X2_agent_analysis.ipynb`**: agent-level analysis (`§What drives belief revisions?`): estimated marginal means and planned contrasts for *plasticity*, *directedness*, and *outgoing influence* across the four scenarios and two network types, plus the variance-decomposition/ICC analysis behind opinion leaders and followers.
5. **`X3_run_analysis.ipynb`**: population-level analysis: estimated marginal means and planned contrasts for *consensus change* across the four scenarios and two network types (Fig. 2C), plus per-scenario convergence rates.
6. **`X4_manuscript_plots.ipynb`**: combines the `X2`/`X3` outputs into the manuscript-ready tables and the combined contrast-forest figure.
7. **`X5_surrogates.ipynb`**: fits/pools the M1–M4 surrogate opinion-dynamics models (persistence; +global belief composition; +local neighborhood composition; +agent identity) and produces the final-state MCC and consensus-fidelity figures/tables (`§Can classical opinion-dynamics models explain these dynamics?`).


## Python environment

This project uses [uv](https://docs.astral.sh/uv/) for Python environment management. Run the setup script for initial configuration:

```sh
./setup.sh          # Install dependencies and configure environment
```

If you have `uv` installed, just run `uv sync` from this directory.

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

> [!NOTE]
> - Some model configurations require a Hugging Face access token in `src/configs/model/*.yaml`.
> - A subset of analysis notebooks relies on `R/rpy2` tooling (see notebooks/r_utils.py and notebook comments for details).

## 📃 Licenses

> [!IMPORTANT]
> This **code** is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

> [!WARNING]
> 1. This is research software. While we strive for correctness and reproducibility, please verify results for your specific use case.
> 2. GitHub Copilot and Claude Code contributed to code annotations, docstrings, and formatting. All algorithmic logic, methodological design, and scientific claims were developed and reviewed by the authors.

**Correspondence**: [g.savcisens@northeastern.edu](mailto:g.savcisens@northeastern.edu)
