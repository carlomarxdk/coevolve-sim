"""Loader utilities for extracting run metadata from outputs directory."""

from __future__ import annotations

import json
from pathlib import Path

from typing import Any

import numpy as np
import pandas as pd

from .utils import build_adjacency_matrix

VALID_SEEDS = [
    814183,
    1170252924,
    900911955,
    473392625,
    964669078,
    1265438423,
    597409993,
    1738238662,
    1866808230,
    13955984,
]

VALID_STATEMENT_IDS = [
    "false_0",
    "false_1",
    "false_2",
    "false_3",
    "false_4",
    "false_5",
    "false_6",
    "false_7",
    "false_8",
    "false_9",
    "true_0",
    "true_1",
    "true_2",
    "true_3",
    "true_4",
    "true_5",
    "true_6",
    "true_7",
    "true_8",
    "true_9",
]

VALID_PROMPTS = ["wR_L"]

VALID_PROBES = ["zeroshot"]

VALID_GRAPH_TYPES = ["erdos-renyi", "watts-strogatz"]

VALID_SETTINGS = ["experts", "random_roles", "random_experts", "base_llms"]


def load_runs_metadata(
    outputs_dir: Path | None = None,
) -> pd.DataFrame:
    """List all unique runs from outputs directory and extract metadata.

    Scans the outputs/runs directory structure and extracts metadata from each
    run. Expected directory structure:
    outputs/runs/{probe}/{setting}/{graph_type}/{prompt}/{statement_id}/{datetime}/

    For each run, extracts:
    - probe: The probe type (e.g., 'zeroshot')
    - setting: The setting/role configuration (e.g., 'experts', 'random_roles')
    - graph_type: Network topology (e.g., 'erdos-renyi', 'watts-strogatz')
    - prompt: Prompt configuration (e.g., 'wR_L', 'woR_C')
    - statement_id: Statement identifier (e.g., 'false_0')
    - datetime: Run timestamp in format YYYY-MM-DD_HH-MM-SS
    - seed: Random seed extracted from config.json

    Args:
        outputs_dir: Path to outputs directory. If None, looks for
            'outputs/runs' relative to current working directory.

    Returns:
        pd.DataFrame: DataFrame with columns [probe, setting, graph_type, prompt,
            statement_id, datetime, seed]. One row per run.

    Example:
        >>> df = load_runs_metadata()
        >>> df.head()
        >>> print(f"Total runs: {len(df)}")
    """
    if outputs_dir is None:
        outputs_dir = Path.cwd() / "outputs" / "runs"
    else:
        outputs_dir = Path(outputs_dir)

    if not outputs_dir.exists():
        raise ValueError(f"Outputs directory not found: {outputs_dir}")

    runs_data = []

    # Walk through the directory structure
    # Pattern: {outputs_dir}/{probe}/{setting}/{graph_type}/{prompt}/{statement_id}/{datetime}/
    for probe_dir in outputs_dir.iterdir():
        if not probe_dir.is_dir():
            continue
        probe = probe_dir.name

        for setting_dir in probe_dir.iterdir():
            if not setting_dir.is_dir():
                continue
            setting = setting_dir.name

            for graph_type_dir in setting_dir.iterdir():
                if not graph_type_dir.is_dir():
                    continue
                graph_type = graph_type_dir.name

                for prompt_dir in graph_type_dir.iterdir():
                    if not prompt_dir.is_dir():
                        continue
                    prompt = prompt_dir.name

                    for statement_dir in prompt_dir.iterdir():
                        if not statement_dir.is_dir():
                            continue
                        statement_id = statement_dir.name

                        for datetime_dir in statement_dir.iterdir():
                            if not datetime_dir.is_dir():
                                continue
                            datetime = datetime_dir.name

                            # Extract seed from config.json
                            config_path = datetime_dir / "config.json"
                            cmpltn_path = (
                                datetime_dir / "results" / "final_metrics.json"
                            )

                            complete_run = True if cmpltn_path.exists() else False
                            if config_path.exists():
                                try:
                                    with open(config_path) as f:
                                        config = json.load(f)
                                    seed = config.get("seed")

                                    runs_data.append(
                                        {
                                            "probe": probe,
                                            "setting": setting,
                                            "graph_type": graph_type,
                                            "prompt": prompt,
                                            "statement_id": statement_id,
                                            "datetime": datetime,
                                            "seed": seed,
                                            "run_path": str(datetime_dir),
                                            "completed": complete_run,
                                            "validated": seed in VALID_SEEDS
                                            and statement_id in VALID_STATEMENT_IDS
                                            and prompt in VALID_PROMPTS
                                            and probe in VALID_PROBES
                                            and graph_type in VALID_GRAPH_TYPES
                                            and setting in VALID_SETTINGS,
                                        }
                                    )
                                except (json.JSONDecodeError, KeyError) as e:
                                    print(
                                        f"Warning: Could not parse config at"
                                        f" {config_path}: {e}"
                                    )

    df = pd.DataFrame(runs_data)

    if df.empty:
        print(f"Warning: No runs found in {outputs_dir}. Check directory structure.")
        return df

    # Convert datetime to proper datetime type
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d_%H-%M-%S")

    # Sort by datetime
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def load_agents_data(run_path: Path | str) -> dict:
    """Load and process agents data from a simulation run.

    Loads agents_data.json. Unpacks nested belief structures into analyzable format.

    Args:
        run_path: Path to the run directory (containing results/ folder)

    Returns:
        dict:
          - belief_cat (np.ndarray): (N_agents, N_statements) matrix of categorical beliefs (0 or 1)
          - belief_con (np.ndarray): (N_agents, N_statements) matrix of P(True) confidence scores
          - belief_ful (np.ndarray): (N_agents, N_statements, 3) matrix of [P(True), P(False), P(Uncertain)]
          - roles (dict): Agent ID (int) -> role (str)
          - models (dict): Agent ID (int) -> model (str)
          - n_rounds (int): Number of rounds in the simulation
          - network_features (dict): Agent ID (int) -> network features dict (e.g. degree, centrality)

    Example:
        >>> data = load_agents_data("outputs/runs/.../2026-01-21_21-03-07/")
        >>> print(data["belief_cat"].shape)  # (30, 11) for 30 agents and 11 statements
        >>> print(data["roles"][0])  # 'Software Engineer'
    """
    run_path = Path(run_path)

    # Load agents data for roles and models
    file_path = run_path / "results" / "agents_data.json"
    with open(file_path) as f:
        raw_data = json.load(f)

    # Extract temporal beliefs
    belief_cat_list = []  # Categorical beliefs (0 or 1)
    belief_con_list = []  # Confidence scores (P(True))
    belief_ful_list = []  # Complete scores [P(True), P(False), P(Uncertain)]
    roles = {}
    models = {}

    for agent_id, agent_data in raw_data.items():
        agent_id = int(agent_id)
        roles[agent_id] = agent_data["role"]
        models[agent_id] = agent_data["model"]
        belief_cat_list.append([int(v) for _, v in agent_data["beliefs"].items()])
        belief_con_list.append([v for _, v in agent_data["belief_scores"].items()])
        complete_scores = [v for _, v in agent_data["complete_scores"].items()]
        belief_ful_list.append(np.array(complete_scores))

    belief_cat = np.vstack(belief_cat_list)
    belief_con = np.vstack(belief_con_list)
    belief_ful = np.array(belief_ful_list)

    file_path = run_path / "results" / "agent_manifest.json"
    with open(file_path) as f:
        raw_manifest = json.load(f)

    manifests = {}
    for agent_id, manifest in raw_manifest.items():
        agent_id = int(agent_id)
        manifests[agent_id] = manifest['network']


    output = {
        "belief_cat": belief_cat,
        "belief_con": belief_con,
        "belief_ful": belief_ful,
        "roles": roles,
        "models": models,
        "n_rounds": belief_cat.shape[-1],
        "network_features": manifests,
    }

    return output


def load_adjacency_matrix(run_path: Path | str) -> np.ndarray:
    """Load adjacency matrix from run data."""
    run_path = Path(run_path)
    file_path = run_path / "results" / "network_edges.json"
    with open(file_path) as f:
        raw_data = json.load(f)
    edges = raw_data["edges"]
    n_agents = raw_data["n"]
    return build_adjacency_matrix(edges, n_agents), edges # type: ignore


def load_network_data(run_path: Path | str) -> dict:
    """Load network data from a run."""
    run_path = Path(run_path)
    file_path = run_path / "results" / "network_manifest.json"
    with open(file_path) as f:
        raw_data = json.load(f)
    return raw_data


def load_run_data(run_path: Path) -> dict[str, Any] | None:
    """Load all necessary data from a single run."""

    run_path = Path(run_path)
    results_dir = run_path / "results"

    try:
        # Load agents data
        with open(results_dir / "agents_data.json") as f:
            agents_data = json.load(f)

        # Load final metrics
        with open(results_dir / "final_metrics.json") as f:
            final_metrics = json.load(f)

        # Load network edges
        with open(results_dir / "network_edges.json") as f:
            network_data = json.load(f)

        # Load config
        with open(run_path / "config.json") as f:
            config = json.load(f)

        with open(results_dir / "per_round_metrics.json") as f:
            rounds = json.load(f)

        return {
            "agents_data": agents_data,
            "final_metrics": final_metrics,
            "network_edges": network_data["edges"],
            "n_agents": network_data["n"],
            "rounds": rounds,
            "config": config,
            "network_manifest": load_network_data(run_path),
        }
    except Exception as e:
        print(f"Error loading {run_path}: {e}")
        return None
