import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _add_corr_box(ax, x_data: np.ndarray, y_data: np.ndarray) -> None:
    """Add Pearson r in a wheat box in the top-left, matching network_impact style."""
    x = pd.Series(x_data)
    y = pd.Series(y_data)
    # drop NaNs
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    corr = np.nan
    if len(x) > 1 and x.std() > 1e-10 and y.std() > 1e-10:
        corr = x.corr(y)
    if not np.isnan(corr):
        ax.text(
            0.05,
            0.95,
            f"r = {corr:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=7,
        )


def load_json(path: Path) -> Any:
    """Load a JSON file and return its contents."""
    with path.open("r") as f:
        return json.load(f)


def find_run_dirs(
    base_runs_dir: Path,
    probe: str,
    experiment_type: str,
    prompt_name: str,
    statement_type: str,
) -> List[Path]:
    """
    Return a list of run directories for a given probe/experiment/statement.

    Expected layout:
        {base_runs_dir}/{probe}/{experiment_type}/{prompt_name}/{statement_type}/{timestamp}/
    """
    root = base_runs_dir / probe / experiment_type / prompt_name / statement_type
    if not root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {root}")
    run_dirs = [p for p in root.iterdir() if p.is_dir()]
    run_dirs.sort()
    return run_dirs


def load_run_data(
    run_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load config, agent_manifest, agents_data, and network_edges for a single run.

    Returns:
        A tuple (config, agent_manifest, agents_data, network_edges).
    """
    config_path = run_dir / "config.json"
    results_dir = run_dir / "results"
    rounds_dir = run_dir / "rounds"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")

    config = load_json(config_path)

    manifest_path = results_dir / "agent_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing agent_manifest.json in {results_dir}")
    agent_manifest = load_json(manifest_path)

    # agents_data: prefer results/agents_data.json, fall back to rounds/agents_data.json
    agents_data_path = results_dir / "agents_data.json"
    if not agents_data_path.exists():
        alt_path = rounds_dir / "agents_data.json"
        if alt_path.exists():
            agents_data_path = alt_path
        else:
            raise FileNotFoundError(
                f"Missing agents_data.json in {results_dir} and {rounds_dir}"
            )
    agents_data = load_json(agents_data_path)

    edges_path = results_dir / "network_edges.json"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing network_edges.json in {results_dir}")
    network_edges = load_json(edges_path)

    return config, agent_manifest, agents_data, network_edges


def compute_downstream_influence(
    network_edges: Dict[str, Any],
    belief_scores: np.ndarray,
) -> np.ndarray:
    """
    Compute downstream influence for each agent based on neighbors' belief-score changes.

    Let s_j(t) be the belief score of agent j at round t.
    For agent i with neighbor set N_i, define:

        influence_i = 1 / ((T - 1) * |N_i|) * sum_{t=0}^{T-2} sum_{j in N_i} |s_j(t+1) - s_j(t)|

    If an agent has no neighbors or T < 2, its influence is set to 0.
    """
    n, T = belief_scores.shape

    # If fewer than 2 rounds, there are no temporal changes.
    if T < 2:
        return np.zeros(n, dtype=float)

    edges = network_edges["edges"]

    # Build undirected neighbor sets (excluding self).
    neighbors: List[set] = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Absolute per-round changes for each agent: shape (n, T-1)
    deltas = np.abs(np.diff(belief_scores, axis=1))

    influence = np.zeros(n, dtype=float)

    for i in range(n):
        neigh = neighbors[i]
        if not neigh:
            # No neighbors => no downstream influence (by this definition)
            continue

        neigh_indices = list(neigh)
        # Submatrix of neighbor deltas: shape (|N_i|, T-1)
        neigh_deltas = deltas[neigh_indices, :]

        # Average over neighbors and time:
        # (1 / ((T-1)*|N_i|)) * sum_{j in N_i, t} |s_j(t+1) - s_j(t)|
        influence[i] = neigh_deltas.mean()

    return influence


def build_belief_matrices(
    agents_data: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Build matrices of beliefs and belief_scores from agents_data.

    agents_data is a dict keyed by agent_id (as string), where each value has:
        - "beliefs": {round_idx: 0/1}
        - "belief_scores": {round_idx: float}

    Returns:
        beliefs:        (n_agents, T) binary beliefs.
        belief_scores:  (n_agents, T) continuous scores.
        agent_ids:      list of agent IDs in row order.
        rounds:         list of round indices in column order.
    """
    agent_ids = sorted(int(aid) for aid in agents_data.keys())
    sample_agent = agents_data[str(agent_ids[0])]
    round_keys = sorted(int(r) for r in sample_agent["beliefs"].keys())
    T = len(round_keys)
    n_agents = len(agent_ids)

    beliefs = np.zeros((n_agents, T), dtype=float)
    belief_scores = np.zeros((n_agents, T), dtype=float)

    for i, aid in enumerate(agent_ids):
        a_data = agents_data[str(aid)]
        for j, r in enumerate(round_keys):
            beliefs[i, j] = a_data["beliefs"][str(r)]
            belief_scores[i, j] = a_data["belief_scores"][str(r)]

    return beliefs, belief_scores, agent_ids, round_keys


def compute_plasticity_and_stubbornness(
    belief_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute plasticity and stubbornness from belief_scores.

    delta_i(t) = belief_scores[i, t+1] - belief_scores[i, t]
    plasticity_i = mean_t |delta_i(t)|
    stubbornness_i = 1 / (1 + plasticity_i)
    """
    if belief_scores.shape[1] < 2:
        n = belief_scores.shape[0]
        plasticity = np.zeros(n, dtype=float)
        stubbornness = np.ones(n, dtype=float)
        return plasticity, stubbornness

    deltas = np.diff(belief_scores, axis=1)  # (n_agents, T-1)
    plasticity = np.mean(np.abs(deltas), axis=1)
    stubbornness = 1.0 / (1.0 + plasticity)
    return plasticity, stubbornness


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    """
    Min–max normalize a 1D array to [0, 1].

    Returns zeros if the array is constant.
    """
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return np.zeros_like(x, dtype=float)
    return (x - x_min) / (x_max - x_min)


def compute_leadership_scores(
    influence: np.ndarray,
    plasticity: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Combine influence and plasticity into a leadership score.

    leadership_i = alpha * influence_norm_i - beta * plasticity_norm_i
    """
    influence_norm = minmax_normalize(influence)
    plasticity_norm = minmax_normalize(plasticity)
    return alpha * influence_norm - beta * plasticity_norm


def analyze_single_run(
    run_dir: Path,
    probe: str,
    experiment_type: str,
    statement_type: str,
    prompt_name: str = "wR_L",
    alpha: float = 1.0,
    beta: float = 1.0,
) -> pd.DataFrame:
    """
    Analyze a single run directory and return a per-agent metrics DataFrame.
    """
    run_timestamp = run_dir.name
    config, agent_manifest, agents_data, network_edges = load_run_data(run_dir)

    beliefs, belief_scores, agent_ids, rounds = build_belief_matrices(agents_data)

    # Influence from downstream changes in neighbors' belief scores
    influence = compute_downstream_influence(network_edges, belief_scores)

    # Plasticity / stubbornness from belief trajectories
    plasticity, stubbornness = compute_plasticity_and_stubbornness(belief_scores)

    # Leadership score
    leadership = compute_leadership_scores(
        influence, plasticity, alpha=alpha, beta=beta
    )

    # degree and eigenvector centrality per agent for this run
    n = int(network_edges["n"])
    edges = network_edges["edges"]

    # Degree
    degree = np.zeros(n, dtype=int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    # Eigenvector centrality on the undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    try:
        evc_dict = nx.eigenvector_centrality_numpy(G)
    except Exception:
        # Fallback: all zeros if it fails for any reason
        evc_dict = {i: 0.0 for i in range(n)}
    evc = np.array([float(evc_dict.get(i, 0.0)) for i in range(n)])

    records = []
    for idx, aid in enumerate(agent_ids):
        a_manifest = agent_manifest[str(aid)]
        role = a_manifest.get("role", None)
        model = a_manifest.get("model", None)

        records.append(
            {
                "probe": probe,
                "experiment_type": experiment_type,
                "statement_type": statement_type,
                "prompt_name": prompt_name,
                "run_timestamp": run_timestamp,
                "agent_id": aid,
                "role": role,
                "model": model,
                "influence": float(influence[aid]),
                "plasticity": float(plasticity[idx]),
                "stubbornness": float(stubbornness[idx]),
                "leadership": float(leadership[idx]),
                "degree": float(degree[aid]),
                "evc": float(evc[aid]),
            }
        )

    df = pd.DataFrame.from_records(records)
    return df


def aggregate_across_runs(per_run_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-run metrics across timestamps.
    """
    group_cols = [
        "probe",
        "experiment_type",
        "statement_type",
        "prompt_name",
        # "agent_id",
        "role",
        # "model",
    ]
    metric_cols = [
        "influence",
        "plasticity",
        "stubbornness",
        "leadership",
        "degree",
        "evc",
    ]

    grouped = per_run_df.groupby(group_cols, dropna=False, as_index=False)

    agg_dict = {}
    for m in metric_cols:
        agg_dict[m + "_mean"] = (m, "mean")
        agg_dict[m + "_std"] = (m, "std")

    summary_df = grouped.agg(**agg_dict)
    return summary_df


def compute_degrees_from_edges(network_edges: Dict[str, Any]) -> np.ndarray:
    """Degree of each node from network_edges['edges']."""
    n = int(network_edges["n"])
    edges = network_edges["edges"]
    deg = np.zeros(n, dtype=int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute influence, plasticity, and leadership scores for LLM opinion networks."
    )
    parser.add_argument(
        "--probe",
        required=True,
        help="Probe name, e.g. 'sawmil' or 'zeroshot' (matches directory under outputs/runs).",
    )
    parser.add_argument(
        "--experiment-type",
        required=True,
        choices=["experts", "only_llms", "random_experts", "random_roles"],
        help="Experiment type (matches directory under outputs/runs/{probe}/).",
    )
    parser.add_argument(
        "--statement-type",
        required=True,
        choices=["false_doc_1", "false_rest_1", "true_doc_1", "true_rest_1"],
        help="Statement type (matches directory under outputs/runs/{probe}/{experiment}/wR_L/).",
    )
    parser.add_argument(
        "--base-runs-dir",
        type=Path,
        default=Path("outputs") / "runs",
        help="Base directory for runs (default: outputs/runs).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs") / "plots",
        help="Base directory for analysis outputs (default: outputs/plots).",
    )
    parser.add_argument(
        "--prompt-name",
        type=str,
        default="wR_L",
        help="Prompt name / subdirectory under experiment-type (default: wR_L).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight on influence when computing leadership.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight on plasticity (subtracted) when computing leadership.",
    )
    parser.add_argument(
        "--doctor-role",
        type=str,
        default="Clinical Physician",
        help="Role name to treat as the doctor (default: 'Clinical Physician').",
    )
    parser.add_argument(
        "--doctor-model",
        type=str,
        default="llama-doc",
        help="Model name for the doctor agent (default: 'llama-doc').",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    probe = args.probe
    experiment_type = args.experiment_type
    statement_type = args.statement_type
    prompt_name = args.prompt_name

    try:
        run_dirs = find_run_dirs(
            base_runs_dir=args.base_runs_dir,
            probe=probe,
            experiment_type=experiment_type,
            prompt_name=prompt_name,
            statement_type=statement_type,
        )
    except FileNotFoundError as e:
        log.error("%s", e)
        return

    if not run_dirs:
        log.error("No run directories found. Check your paths and arguments.")
        return

    log.info(
        "Found %d run(s) for %s / %s / %s",
        len(run_dirs),
        probe,
        experiment_type,
        statement_type,
    )

    per_run_dfs: List[pd.DataFrame] = []
    for run_dir in run_dirs:
        log.info("Analyzing run: %s", run_dir)
        try:
            df_run = analyze_single_run(
                run_dir=run_dir,
                probe=probe,
                experiment_type=experiment_type,
                statement_type=statement_type,
                prompt_name=prompt_name,
                alpha=args.alpha,
                beta=args.beta,
            )
            per_run_dfs.append(df_run)
        except FileNotFoundError as e:
            log.warning("Skipping run %s due to missing file: %s", run_dir, e)
        except Exception as e:
            log.exception(
                "Failed to analyze run %s due to unexpected error: %s", run_dir, e
            )

    if not per_run_dfs:
        log.error("All runs failed; no output generated.")
        return

    per_run_df = pd.concat(per_run_dfs, ignore_index=True)

    # Match the plot script convention:
    # outputs/plots/{probe}/{experiment_type}/{prompt_name}/{statement_type}/
    out_root = args.out_dir / probe / experiment_type / prompt_name / statement_type
    out_root.mkdir(parents=True, exist_ok=True)

    per_run_path = out_root / "opinion_leaders_per_run.csv"
    per_run_df.to_csv(per_run_path, index=False)
    log.info("Saved per-run metrics to %s", per_run_path)

    summary_df = aggregate_across_runs(per_run_df)
    summary_path = out_root / "opinion_leaders_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("Saved aggregated summary to %s", summary_path)


if __name__ == "__main__":
    main()
