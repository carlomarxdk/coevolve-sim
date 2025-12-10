[![DOI](https://zenodo.org/badge/1113345522.svg)](https://doi.org/10.5281/zenodo.17875304)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Email](https://img.shields.io/badge/Email-g.savcisens@northeastern.edu-orange)](mailto:g.savcisens@northeastern.edu)

# CoevolveSim

**Belief Coevolution in a Social Network of Generalist and Specialist Large Language Models**

**CoevolveSim** is an agent-based simulation framework for studying how *beliefs coevolve* in a social network of large language models (LLMs) with varying **domain expertise** and **social cues**. The framework operationalizes coevolution as a multi-round interaction process in which LLM agents exchange beliefs, respond to neighbor summaries, and update their internal stance on factual statements.

CoevolveSim enables systematic evaluation of how expertise, roles and network structure jointly shape collective reasoning in LLM populations. The framework underlies all analyses in our paper **TO BE ANNOUNCED**.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Running Experiments](#running-experiments)
- [Analyzing Results](#analyzing-results)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

---

## Overview

This framework simulates belief dynamics in social networks where agents are LLM-based entities that:

1. **Initialize** with an independent assessment of a statement
2. **Observe** the beliefs of their network neighbors
3. **Update** their own beliefs through social influence
4. **Propagate** beliefs across the network over multiple rounds

The simulation uses neural probes to extract belief scores from LLM hidden states, enabling quantitative tracking of belief evolution without requiring explicit probabilistic outputs from the models.

### What CoevolveSim Models

Each agent in CoevolveSim is an LLM with:

- **Expertise**:  determined by its underlying model (e.g., biomedical, math, cybersecurity).
- **Role**:  a social label such as “Clinical Physician,” “Chemist,” or “Human Participant”.

Agents interact through three key steps:

1. **Initial belief formation**
   Agents receive a factual statement (e.g., “Methyl nicotinate is indicated for the treatment of aches”) and produce an initial classification via a zero-shot probe.
2. **Social exposure**
   On each subsequent round, agents receive *aggregated neighbor summaries* containing the expressed beliefs and roles of adjacent agents (see Fig. 1, page 2).
3. **Belief updating**
   Agents re-evaluate the statement conditioned on their own knowledge plus the neighborhood summary, yielding new belief scores across {true, false, neither}.

Over multiple rounds, belief states propagate through the network, enabling the study of convergence, cascades, fragmentation, and influence.

### Research Questions

CoevolveSim is designed to study questions such as:

- How do LLM beliefs change when agents interact through a network rather than in isolation?
- When do **expert** agents stabilize collective beliefs?
- How do **social identities** (roles) influence agent susceptibility or influence?
- Under what conditions do belief cascades emerge?
- Are there *persistent opinion leaders* in LLM networks?

---

## Installation

### Prerequisites

- **Python**: 3.10 or higher (tested on 3.10, 3.11, 3.12)
- **PyTorch**: 2.6.0 (for GPU support, install appropriate CUDA version)
- **GPU**: CUDA-capable GPU recommended (optional, CPU mode available)
- **Memory**: At least 16GB RAM (32GB+ recommended for larger models)

### Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/carlomarxdk/coevolve-sim.git
   cd coevolve-sim
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   **Note on GPU Support:** If you need CUDA support, ensure you install PyTorch with the correct CUDA version for your system. See [PyTorch installation guide](https://pytorch.org/get-started/locally/).

4. **Verify installation:**

   ```bash
   # Run tests to verify the installation
   pytest tests/ -v
   
   # Run a quick test experiment (uses dummy backend, no GPU required)
   python experiment.py backend=dummy experiment.max_rounds=2 network.params.n=5
   ```

### Alternative: Pipenv Setup

If you prefer Pipenv:

```bash
pipenv install
pipenv shell
```

See this [guide on Pipenv setup](https://medium.com/geekculture/setting-up-python-environment-in-macos-using-pyenv-and-pipenv-116293da8e72) for macOS users.

## Quick Start

The framework uses [Hydra](https://hydra.cc/) for configuration management. The main entry point is `experiment.py`.

You can run one of the default experiment (16 agents, Erdős–Rényi network, 10 rounds):

```bash
python experiment.py
```

Customize parameters using Hydra:

```bash
python experiment.py \
  network.params.n=20 \
  network.params.p=0.5 \
  experiment.max_rounds=15 \
  seed=42
```

Reproduce the configuration used in the paper:

```bash
python experiment.py \
  catalog=random_roles \
  prompt=wR_L \
  network.params.n=16 \
  network.params.p=0.3 \
  experiment.max_rounds=10 \
  seed=814183
```

> [!IMPORTANT]
> Some of the LLMs specified in the `configs/model` requires personal  HuggingFace token.
> For example, in case of `llama-base.yaml`, you need to update the `access:token:` field with a valid [Access Token](https://huggingface.co/settings/tokens).

---

## Project Structure

```
coevolve-sim/
├── configs/                  # Hydra configuration files
│   ├── experiment.yaml       # Main experiment configuration
│   ├── catalog/             # Agent role configurations
│   ├── network/             # Network topology settings
│   ├── probe/               # Belief probe configurations
│   ├── prompt/              # Prompt template definitions
│   └── statement/           # Evaluation statements
├── src/                     # Core source code
│   ├── Agent.py            # Agent classes (LLM, Expert)
│   ├── InferenceScheduler.py  # Model loading and caching
│   ├── Message.py          # Prompt generation system
│   ├── MetricsTracker.py   # Metrics collection
│   ├── Network.py          # Network topology management
│   ├── Probe.py            # Belief scoring probes
│   └── metrics/            # Metric computation modules
├── tests/                   # Comprehensive test suite
├── experiment.py            # Main experiment runner
├── plot_*.py               # Plotting and visualization scripts
├── utils.py                # Utility functions
└── requirements.txt         # Python dependencies
```

### Key Files

- **`experiment.py`**: Main entry point for running simulations
- **`utils.py`**: Experiment checkpointing, I/O management, and utilities
- **`plot_*.py`**: Various plotting scripts for visualizing results

### Configuration System

The experiment uses Hydra's compositional configuration with several modular groups:

#### 1. Network Configuration (`configs/network/`)

Defines the social network topology connecting agents.

- **`erdos-renyi`**: Random graph generator
  - `n`: Number of nodes/agents (default: 16)
  - `p`: Edge probability between any two nodes (default: 0.3)

```bash
# Create a network with 20 nodes and 50% edge probability
python experiment.py network.params.n=20 network.params.p=0.5
```

#### 2. Agent Catalog (`configs/catalog/`)

Defines the composition of agents in the simulation.

- **`simple`**: All agents have "LLM" role (default)
- **`only_llms`**: All agents have "LLM" role (same as simple)
- **`only_participants`**: All agents have "Human Participant" role  
- **`llm_and_participant`**: 50% LLM role, 50% Participant role
- **`random_roles`**: Random role assignment from diverse role pool
- **`experts`**: Expert agents with predefined knowledge
- **`random_experts`**: Mix of LLM and expert agents

```bash
python experiment.py catalog=random_roles
```

##### LLMs

These catalogs pull information for each single LLM from `configs/models`.

> [!IMPORTANT]
> Some of the LLMs specified in the `configs/model` requires personal HuggingFace token.
> For example, in case of `llama-base.yaml`, you need to update the `access:token:` field with a valid [Access Token](https://huggingface.co/settings/tokens).

#### 3. Prompt Templates (`configs/prompt/`)

Controls how agents are prompted and how neighbor information is presented.

- **`woR_C`**: Without role context, count-based neighbor aggregation
- **`wR_C`**: With role context, count-based neighbor aggregation
- **`wR_L`**: With role context, detailed list of neighbor beliefs

```bash
python experiment.py prompt=wR_L
```

#### 4. Statements (`configs/statement/`)

The claims that agents evaluate. Includes both true and false medical/scientific statements.

```bash
python experiment.py statement=true_doc_1
python experiment.py statement=false_rest_1
```

### Output Organization

Experiments are automatically saved in a hierarchical structure:

```
outputs/runs/
└── <catalog>/
    └── <prompt>/
        └── <statement>/
            └── YYYY-MM-DD_HH-MM-SS/
                ├── config.json              # Complete configuration
                ├── rounds/                  # Per-round data
                │   ├── round_0/
                │   │   ├── beliefs.jsonl
                │   │   ├── beliefs_detailed.json
                │   │   └── metrics.json
                │   └── ...
                └── results/                 # Final results
                    ├── agents_data.json
                    ├── network_edges.json
                    ├── final_metrics.json
                    └── per_round_metrics.json
```

## Testing

Run the test suite to verify your installation:

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_agent.py -v
pytest tests/test_network.py -v
pytest tests/test_probe.py -v
```

Test coverage includes:

- Agent initialization and belief tracking
- Network topology and edge operations
- Model loading and caching
- Probe scoring and calibration
- Message generation and prompting
- Metrics tracking and aggregation

## Ensuring Reproducible Experiments

1. **Always set a seed:**

   ```bash
   python experiment.py seed=814183
   ```

> [!Note]
> `Seed.txt` contains a list of seed we use for the reruns of the experiments.

2. **Document your configuration:**
   Each experiment automatically saves its complete configuration to `config.json` in the output directory.

3. **Version control your configs:**
   Custom configuration files in `configs/` should be version controlled.

4. **Check for completed experiments:**
   The framework automatically checks if an experiment with the same configuration and seed has already been completed.

## Citation

If you use this framework in your research, please cite our paper:

```bibtex
TO BE ANNOUNCED
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions involving:

- New expert LLMs
- Alternative network topologies (e.g., Watts–Strogatz, Barabási–Albert)
- Additional belief extraction methods (e.g., activation-based probes)
- Human–LLM coevolution extensions

Please open an issue before submitting a PR.

## Authors

Germans Savcisens\*, Samantha Dies, Courtney Maynard, and Tina Eliassi-Rad  
Khoury College of Computer Sciences, Northeastern University  
Santa Fe Institute (T.E.-R.)  

\*Correspondence about the code or paper: **[g.savcisens@northeastern.edu](mailto:g.savcisens@northeastern.edu)**

> [!NOTE]
>
> 1. This is research software. While we strive for correctness and reproducibility, please verify results for your specific use case.
> 2. GitHub Copilot contributed to code annotations, docstrings, and formatting. All algorithmic logic, methodological design, and scientific claims were developed and reviewed by the authors.
