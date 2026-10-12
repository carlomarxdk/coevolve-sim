[![DOI](https://zenodo.org/badge/1113345522.svg)](https://doi.org/10.5281/zenodo.17875304)
[![License: MIT](https://img.shields.io/badge/License-MIT-FF7873.svg)](LICENSE)
[![Email](https://img.shields.io/badge/Email-g.savcisens@northeastern.edu-E22F80)](mailto:g.savcisens@northeastern.edu)
[![arXiv](https://img.shields.io/badge/arXiv-2607.27512-8140DC.svg)](https://arxiv.org/abs/2607.27512)

# CoevolveSim

Code and simulation data for **[Belief Coevolution in a Social Network of Generalist and Specialist Large Language Models](https://arxiv.org/abs/2607.27512)**:

```bibtex
@misc{savcisens2026coevolvesim,
      title={Belief Coevolution in a Social Network of Generalist and Specialist Large Language Models}, 
      author={Germans Savcisens and Samantha Dies and Courtney Maynard and Tina Eliassi-Rad},
      year={2026},
      doi={10.48550/arXiv.2607.27512},
      publisher={arXiv},
      url={https://arxiv.org/abs/2607.27512}, 
}
```

`CoevolveSim` is a framework for studying belief diffusion within networked LLM populations. Generalist and specialist LLM agents are placed on a social network (Erdős–Rényi or Watts–Strogatz) and exchange beliefs about medical-indication statements over several rounds, each agent revising its belief after observing a summary of its neighbors' beliefs. Across four scenarios:

-  **I.** baseline generalists (`catalog=base_llms`), 
-  **II.** generalists with random social roles (`catalog=random_roles`), 
-  **III.** specialists with random roles (`catalog=random_experts`), 
-  **IV.** specialists with roles matched to their domain (`catalog=experts`).

These simulations isolate the effects of persona-style role assignment, domain specialization (model heterogeneity), and role–specialization alignment on individual belief revision and population-level consensus. A hierarchy of classical opinion-dynamics surrogate models (M1–M4) is then fit to test which mechanisms (persistence, social belief composition, agent identity) are needed to reproduce the observed dynamics.

## Project structure

1. `src`:  simulation framework, Hydra configs (`src/configs/`), and analysis code.
2. `data`: raw simulation output (`data/outputs/`) and derived analysis tables/figures (`data/analysis/`).
3. `notebooks`: cleaned, documented notebooks that reproduce every table and figure in the paper; see [Notebooks](#notebooks) below.

## Notebooks

`notebooks` contains the documented, reproducible pipeline behind the paper's results. Each notebook's own intro cell states exactly which figures/tables/sections it produces:

1. **`maximin_selection.ipynb`**: reproduces the maximin selection of the network realizations (16 per network structure, starting from the 8 realizations of each base config) and the 20 discussion statements.
2. **`sanity_check.ipynb`**: checks that all 2,560 runs are in place and that networks, roles and models are randomized as the design requires.
3. **`expanded/E1_data.ipynb`**: turns raw per-run simulation output (`data/outputs/runs/`) into the two aggregated tables every later notebook builds on (`data/outputs/aggregated_expanded/{agent_level_data,run_level_data}.parquet`).
4. **`expanded/E0_graph_configs.ipynb`**: structural features of the 32 network realizations (16 Erdős–Rényi, 16 Watts–Strogatz). Needs `E1`.
5. **`expanded/E2_agent_analysis.ipynb`**: agent-level analysis (§5.1 *What shapes belief revision?*): estimated marginal means and planned contrasts for *plasticity*, *directedness*, and *outgoing influence* across the four scenarios and network types, plus the variance-decomposition/ICC analysis behind opinion leaders and followers. Uses R via `rpy2`.
6. **`expanded/E3_run_analysis.ipynb`**: population-level analysis: estimated marginal means and planned contrasts for *consensus change* across the four scenarios and network types. Uses R via `rpy2`.
7. **`expanded/E4_surrogates.ipynb`**: M1–M4 surrogate opinion-dynamics models (M1 persistence; M2 + population-level belief composition; M3 + local neighborhood belief composition; M4 = M3 + agent identity): final-state MCC and consensus-fidelity figures/tables (§5.2 *Can opinion-dynamics surrogates reproduce these dynamics?*). First run:
   ```sh
   python src/analysis/dynamics_model_fitting/aggregate_trajectories.py
   python src/analysis/dynamics_model_fitting/surrogate_evaluation.py
   ```
8. **`expanded/E5_manuscript_plots.ipynb`**: combines the `E2`/`E3` outputs into the manuscript-ready tables and contrast-forest figures.
9. **`expanded/E6_round0_calibration.ipynb`**: round-0 accuracy and calibration appendix. Needs `E1`.

Analysis outputs are written to `data/analysis/aggregated_expanded/` (`E0` writes to `data/analysis/expanded/graphs/`).


## Python environment

This project uses [uv](https://docs.astral.sh/uv/) for Python environment management. Run the setup script for initial configuration:

```sh
./setup.sh          # Install dependencies and configure environment
```

If you have `uv` installed, just run `uv sync` from this directory.

## Run Experiments

Canonical entrypoint (run from the repository root):

```bash
uv run python -m src.experiment
```

Example run with Hydra overrides:

```bash
uv run python -m src.experiment \
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
> - Simulations need a GPU. Some model configurations require a Hugging Face access token in `src/configs/model/*.yaml` (`access.token`).
> - `E2`, `E3` and `E6` use R via `rpy2` (R with `lme4`, `lmerTest`, `emmeans`, `RLRsim`; see `notebooks/r_utils.py`).

## 📃 Licenses

> [!IMPORTANT]
> This **code** is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

> [!WARNING]
> 1. This is research software. While we strive for correctness and reproducibility, please verify results for your specific use case.
> 2. GitHub Copilot and Claude Code contributed to code annotations, docstrings, and formatting. All algorithmic logic, methodological design, and scientific claims were developed and reviewed by the authors.

**Correspondence**: [g.savcisens@northeastern.edu](mailto:g.savcisens@northeastern.edu)
