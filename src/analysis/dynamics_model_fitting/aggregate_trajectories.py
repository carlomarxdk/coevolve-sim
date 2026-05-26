import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass
class RunMetadata:
    """Metadata parsed from a single run directory path."""

    experiment_group: str
    graph_type: str
    statement_id: str
    timestamp: str
    run_dir: str
    run_id: str
    seed: int


@dataclass
class TrajectoryRow:
    """One row per agent trajectory within a run."""

    experiment_group: str
    graph_type: str
    statement_id: str
    timestamp: str
    run_dir: str
    run_id: str
    agent_id: str
    agent_role: str | None
    agent_model: str | None
    seed: int
    n_rounds: int
    beliefs_by_round: str
    belief_scores_by_round: str
    complete_scores_by_round: str


@dataclass
class RoundRow:
    """One row per agent per round within a run."""

    experiment_group: str
    graph_type: str
    statement_id: str
    timestamp: str
    run_dir: str
    run_id: str
    seed: int
    agent_id: str
    agent_role: str | None
    agent_model: str | None
    round_idx: int
    belief_label: float | int | None
    belief_score_true: float | None
    score_true: float | None
    score_false: float | None
    score_neither: float | None
    degree: int | None = None
    neighbor_ids: str | None = None


@dataclass
class TransitionRow:
    """One row per agent transition t -> t+1 within a run."""

    experiment_group: str
    graph_type: str
    statement_id: str
    timestamp: str
    run_dir: str
    run_id: str
    seed: int
    agent_id: str
    agent_role: str | None
    agent_model: str | None
    round_t: int
    round_t1: int
    belief_t: float | int | None
    belief_t1: float | int | None
    changed: int
    belief_score_true_t: float | None
    belief_score_true_t1: float | None
    score_true_t: float | None
    score_false_t: float | None
    score_neither_t: float | None
    score_true_t1: float | None
    score_false_t1: float | None
    score_neither_t1: float | None
    degree: int | None = None
    neighbor_ids: str | None = None


@dataclass
class RunRoundMetricsRow:
    """One row per run per round from per_round_metrics.json."""

    experiment_group: str
    graph_type: str
    statement_id: str
    timestamp: str
    run_dir: str
    run_id: str
    seed: int
    round_idx: int
    entropy: float | None
    multiclass_entropy: float | None
    magnetization: float | None
    polarity: float | None
    consensus_fraction: float | None
    flip_rate: float | None
    change_l1: float | None
    change_l2: float | None
    agreement_mean: float | None
    edge_disagreement: float | None
    assortativity: float | None
    assortativity_con: float | None
    cross_belief_fraction_cat: float | None
    cross_belief_fraction_con: float | None
    modularity: float | None
    belief_label_mean: float | None
    belief_label_var: float | None
    belief_label_median: float | None
    belief_label_mode: float | int | None
    belief_score_mean: float | None
    belief_score_var: float | None
    belief_score_median: float | None
    belief_score_mode: float | None
    neighbor_mean_by_agent: str | None


@dataclass
class EdgeRow:
    """One row per undirected edge within a run."""

    experiment_group: str
    graph_type: str
    statement_id: str
    timestamp: str
    run_dir: str
    run_id: str
    seed: int
    source_agent_id: str
    target_agent_id: str


@dataclass
class AgentNetworkRow:
    """One row per agent with static graph features within a run."""

    experiment_group: str
    graph_type: str
    statement_id: str
    timestamp: str
    run_dir: str
    run_id: str
    seed: int
    agent_id: str
    degree: int
    neighbor_ids: str


class AggregationError(RuntimeError):
    """Raised when a run directory cannot be parsed correctly."""


class SeedMetadataError(AggregationError):
    """Raised when run seed metadata is missing or invalid."""


EXPECTED_RESULTS_FILES = (
    "config.json",
    "results/agents_data.json",
    "results/per_round_metrics.json",
    "results/network_edges.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate opinion-dynamics experiment outputs into trajectory, round, "
            "transition, and run-level metric tables."
        )
    )
    project_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--input_root",
        type=Path,
        default=project_root / "data/outputs/runs/zeroshot",
        help="Root directory containing zeroshot run outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "src/analysis/dynamics_model_fitting/data",
        help="Directory where aggregated parquet/csv files will be written.",
    )
    parser.add_argument(
        "--write_csv",
        action="store_true",
        help="Also write CSV copies in addition to parquet.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if a run cannot be parsed.",
    )
    return parser.parse_args()


def find_run_dirs(input_root: Path) -> list[Path]:
    """Return run directories that appear to contain the required result files."""
    run_dirs: list[Path] = []
    for agents_path in input_root.glob("*/*/*/*/*/results/agents_data.json"):
        run_dir = agents_path.parents[1]
        if all((run_dir / rel_path).exists() for rel_path in EXPECTED_RESULTS_FILES):
            run_dirs.append(run_dir)
    return sorted(set(run_dirs))


def parse_run_metadata(run_dir: Path, input_root: Path) -> RunMetadata:
    """Parse metadata encoded in the directory structure."""
    rel_parts = run_dir.relative_to(input_root).parts
    if len(rel_parts) < 5:
        raise AggregationError(f"Run directory has unexpected structure: {run_dir}")

    experiment_group = rel_parts[0]
    graph_type = rel_parts[1]
    statement_id = rel_parts[3]
    timestamp = rel_parts[4]
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise SeedMetadataError(f"Missing config.json in run directory: {run_dir}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SeedMetadataError(
            f"Failed to load config.json for {run_dir}: {exc}"
        ) from exc

    raw_seed = config.get("seed")
    if raw_seed is None:
        raise SeedMetadataError(f"Missing 'seed' in config.json for run: {run_dir}")

    try:
        seed = int(raw_seed)
    except (ValueError, TypeError) as exc:
        raise SeedMetadataError(
            f"Invalid seed value in config.json for run {run_dir}: {raw_seed}"
        ) from exc
        
        
    run_id = f"{experiment_group}__{graph_type}__{statement_id}__{seed}"

    return RunMetadata(
        experiment_group=experiment_group,
        graph_type=graph_type,
        statement_id=statement_id,
        timestamp=timestamp,
        run_dir=str(run_dir),
        seed=seed,
        run_id=run_id,
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_json_dumps(obj: Any) -> str:
    """Serialize nested dict/list objects for storage in a tabular column."""
    return json.dumps(obj, sort_keys=True)


def sorted_round_keys(mapping: dict[str, Any]) -> list[str]:
    return sorted(mapping.keys(), key=lambda x: int(x))


def build_edge_rows(
    network_edges: dict[str, Any],
    meta: RunMetadata,
) -> list[EdgeRow]:
    rows: list[EdgeRow] = []
    for edge in network_edges.get("edges", []):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        source, target = edge
        rows.append(
            EdgeRow(
                **asdict(meta),
                source_agent_id=str(source),
                target_agent_id=str(target),
            )
        )
    return rows


def build_agent_network_rows(
    network_edges: dict[str, Any],
    meta: RunMetadata,
    agent_ids: Iterable[str] | None = None,
) -> list[AgentNetworkRow]:
    n_agents = network_edges.get("n")
    adjacency: dict[str, set[str]] = {}

    if n_agents is not None:
        for agent_idx in range(int(n_agents)):
            adjacency[str(agent_idx)] = set()

    if agent_ids is not None:
        for agent_id in agent_ids:
            adjacency.setdefault(str(agent_id), set())

    for edge in network_edges.get("edges", []):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        source, target = str(edge[0]), str(edge[1])
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    rows: list[AgentNetworkRow] = []
    for agent_id in sorted(adjacency.keys(), key=int):
        neighbor_ids = sorted(adjacency[agent_id], key=int)
        rows.append(
            AgentNetworkRow(
                **asdict(meta),
                agent_id=agent_id,
                degree=len(neighbor_ids),
                neighbor_ids=safe_json_dumps(neighbor_ids),
            )
        )
    return rows


def build_trajectory_rows(
    agents_data: dict[str, dict[str, Any]],
    meta: RunMetadata,
) -> list[TrajectoryRow]:
    rows: list[TrajectoryRow] = []

    for agent_id, agent_payload in agents_data.items():
        beliefs = agent_payload.get("beliefs", {})
        belief_scores = agent_payload.get("belief_scores", {})
        complete_scores = agent_payload.get("complete_scores", {})

        rows.append(
            TrajectoryRow(
                **asdict(meta),
                agent_id=str(agent_id),
                agent_role=agent_payload.get("role"),
                agent_model=agent_payload.get("model"),
                n_rounds=len(beliefs),
                beliefs_by_round=safe_json_dumps(beliefs),
                belief_scores_by_round=safe_json_dumps(belief_scores),
                complete_scores_by_round=safe_json_dumps(complete_scores),
            )
        )

    return rows


def build_round_rows(
    agents_data: dict[str, dict[str, Any]],
    meta: RunMetadata,
    agent_network_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[RoundRow]:
    rows: list[RoundRow] = []

    for agent_id, agent_payload in agents_data.items():
        beliefs = agent_payload.get("beliefs", {})
        belief_scores = agent_payload.get("belief_scores", {})
        complete_scores = agent_payload.get("complete_scores", {})
        network_info = (agent_network_lookup or {}).get(str(agent_id), {})

        round_keys = sorted(
            set(beliefs) | set(belief_scores) | set(complete_scores),
            key=int,
        )

        for round_key in round_keys:
            full_scores = complete_scores.get(round_key)
            score_true, score_false, score_neither = _unpack_complete_scores(full_scores)

            rows.append(
                RoundRow(
                    **asdict(meta),
                    agent_id=str(agent_id),
                    agent_role=agent_payload.get("role"),
                    agent_model=agent_payload.get("model"),
                    round_idx=int(round_key),
                    belief_label=beliefs.get(round_key),
                    belief_score_true=belief_scores.get(round_key),
                    score_true=score_true,
                    score_false=score_false,
                    score_neither=score_neither,
                    degree=network_info.get("degree"),
                    neighbor_ids=network_info.get("neighbor_ids"),
                )
            )

    return rows


def build_transition_rows(
    agents_data: dict[str, dict[str, Any]],
    meta: RunMetadata,
    agent_network_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[TransitionRow]:
    rows: list[TransitionRow] = []

    for agent_id, agent_payload in agents_data.items():
        beliefs = agent_payload.get("beliefs", {})
        belief_scores = agent_payload.get("belief_scores", {})
        complete_scores = agent_payload.get("complete_scores", {})
        network_info = (agent_network_lookup or {}).get(str(agent_id), {})

        round_keys = sorted_round_keys(beliefs)

        for current_key, next_key in zip(round_keys[:-1], round_keys[1:]):
            current_complete = complete_scores.get(current_key)
            next_complete = complete_scores.get(next_key)

            current_true, current_false, current_neither = _unpack_complete_scores(current_complete)
            next_true, next_false, next_neither = _unpack_complete_scores(next_complete)

            belief_t = beliefs.get(current_key)
            belief_t1 = beliefs.get(next_key)

            rows.append(
                TransitionRow(
                    **asdict(meta),
                    agent_id=str(agent_id),
                    agent_role=agent_payload.get("role"),
                    agent_model=agent_payload.get("model"),
                    round_t=int(current_key),
                    round_t1=int(next_key),
                    belief_t=belief_t,
                    belief_t1=belief_t1,
                    changed=int(belief_t != belief_t1),
                    belief_score_true_t=belief_scores.get(current_key),
                    belief_score_true_t1=belief_scores.get(next_key),
                    score_true_t=current_true,
                    score_false_t=current_false,
                    score_neither_t=current_neither,
                    score_true_t1=next_true,
                    score_false_t1=next_false,
                    score_neither_t1=next_neither,
                    degree=network_info.get("degree"),
                    neighbor_ids=network_info.get("neighbor_ids"),
                )
            )

    return rows


def build_run_round_metrics_rows(
    per_round_metrics: dict[str, Any],
    meta: RunMetadata,
) -> list[RunRoundMetricsRow]:
    rows: list[RunRoundMetricsRow] = []

    for round_payload in per_round_metrics.get("per_round", []):
        belief_label = ((round_payload.get("belief") or {}).get("label") or {})
        belief_score = ((round_payload.get("belief") or {}).get("score") or {})
        consensus = round_payload.get("consensus") or {}
        temporal = round_payload.get("temporal") or {}
        local = round_payload.get("local") or {}

        rows.append(
            RunRoundMetricsRow(
                **asdict(meta),
                round_idx=round_payload.get("round"),
                entropy=round_payload.get("entropy"),
                multiclass_entropy=round_payload.get("multiclass_entropy"),
                magnetization=consensus.get("magnetization"),
                polarity=consensus.get("polarity"),
                consensus_fraction=consensus.get("consensus_fraction"),
                flip_rate=temporal.get("flip_rate"),
                change_l1=temporal.get("change_l1"),
                change_l2=temporal.get("change_l2"),
                agreement_mean=local.get("agreement_mean"),
                edge_disagreement=local.get("edge_disagreement"),
                assortativity=local.get("assortativity"),
                assortativity_con=local.get("assortativity_con"),
                cross_belief_fraction_cat=local.get("cross_belief_fraction_cat"),
                cross_belief_fraction_con=local.get("cross_belief_fraction_con"),
                modularity=local.get("modularity"),
                belief_label_mean=belief_label.get("mean"),
                belief_label_var=belief_label.get("var"),
                belief_label_median=belief_label.get("median"),
                belief_label_mode=belief_label.get("mode"),
                belief_score_mean=belief_score.get("mean"),
                belief_score_var=belief_score.get("var"),
                belief_score_median=belief_score.get("median"),
                belief_score_mode=belief_score.get("mode"),
                neighbor_mean_by_agent=safe_json_dumps(local.get("neighbor_mean")),
            )
        )

    return rows


def _unpack_complete_scores(
    scores: list[float] | tuple[float, float, float] | None,
) -> tuple[float | None, float | None, float | None]:
    if scores is None or len(scores) != 3:
        return None, None, None
    return scores[0], scores[1], scores[2]


def rows_to_dataframe(rows: Iterable[Any]) -> pd.DataFrame:
    return pd.DataFrame([asdict(row) for row in rows])


def add_global_prevalence_features(transition_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-run, per-round global state prevalence features based on belief_t."""
    prevalence_df = (
        transition_df.groupby(["run_dir", "round_t"])["belief_t"]
        .value_counts(normalize=True)
        .rename("global_fraction")
        .reset_index()
        .pivot(
            index=["run_dir", "round_t"],
            columns="belief_t",
            values="global_fraction",
        )
        .reset_index()
    )

    rename_map = {
        -1: "global_frac_neg1_t",
        0: "global_frac_0_t",
        1: "global_frac_1_t",
    }
    prevalence_df = prevalence_df.rename(columns=rename_map)

    for col in ["global_frac_neg1_t", "global_frac_0_t", "global_frac_1_t"]:
        if col not in prevalence_df.columns:
            prevalence_df[col] = 0.0

    prevalence_df["n_agents_in_run"] = (
        transition_df.groupby(["run_dir", "round_t"])["agent_id"]
        .nunique()
        .values
    )

    return transition_df.merge(
        prevalence_df[
            [
                "run_dir",
                "round_t",
                "global_frac_neg1_t",
                "global_frac_0_t",
                "global_frac_1_t",
                "n_agents_in_run",
            ]
        ],
        on=["run_dir", "round_t"],
        how="left",
    )


def add_neighbor_prevalence_features(
    transition_df: pd.DataFrame,
    round_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add per-agent, per-round neighbor state fractions based on belief_label at round t.

    Adds:
        - neighbor_frac_neg1_t
        - neighbor_frac_0_t
        - neighbor_frac_1_t
        - n_neighbors_with_observed_belief_t

    Assumes:
        - transition_df has columns: run_dir, round_t, agent_id, neighbor_ids
        - round_df has columns: run_dir, round_idx, agent_id, belief_label
        - neighbor_ids is a JSON-serialized list of agent IDs
    """
    if transition_df.empty:
        return transition_df.copy()

    # Keep only the belief information we need from the round table, aligned to round_t.
    _round_cols = ["run_dir", "round_idx", "agent_id", "belief_label"]
    if "belief_score_true" in round_df.columns:
        _round_cols.append("belief_score_true")
    round_beliefs = round_df[_round_cols].copy()
    round_beliefs = round_beliefs.rename(
        columns={
            "round_idx": "round_t",
            "agent_id": "neighbor_agent_id",
            "belief_label": "neighbor_belief_t",
            "belief_score_true": "neighbor_belief_score_true_t",
        }
    )
    round_beliefs["neighbor_agent_id"] = round_beliefs["neighbor_agent_id"].astype(str)

    # Work on a copy so we do not mutate upstream dataframes.
    neighbor_df = transition_df[
        ["run_dir", "round_t", "agent_id", "neighbor_ids"]
    ].copy()

    # Parse neighbor_ids from JSON string to Python list.
    neighbor_df["neighbor_ids"] = neighbor_df["neighbor_ids"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else ([] if pd.isna(x) else x)
    )

    # Expand one row per (agent, neighbor, round_t).
    neighbor_df = neighbor_df.explode("neighbor_ids", ignore_index=False)
    neighbor_df = neighbor_df.rename(columns={"neighbor_ids": "neighbor_agent_id"})
    neighbor_df["neighbor_agent_id"] = neighbor_df["neighbor_agent_id"].astype("string")

    # Merge in each neighbor's belief at round t.
    neighbor_df = neighbor_df.merge(
        round_beliefs,
        on=["run_dir", "round_t", "neighbor_agent_id"],
        how="left",
    )

    # Count fractions of neighbors in each state.
    # Drop rows where a neighbor is missing entirely after explode (e.g., isolated nodes).
    valid_neighbor_rows = neighbor_df.dropna(subset=["neighbor_agent_id"]).copy()

    if valid_neighbor_rows.empty:
        result = transition_df.copy()
        result["neighbor_frac_neg1_t"] = 0.0
        result["neighbor_frac_0_t"] = 0.0
        result["neighbor_frac_1_t"] = 0.0
        result["n_neighbors_with_observed_belief_t"] = 0
        if "neighbor_belief_score_true_t" in valid_neighbor_rows.columns:
            result["neighbor_mean_belief_score_true_t"] = float("nan")
        return result

    frac_df = (
        valid_neighbor_rows.groupby(["run_dir", "round_t", "agent_id"])["neighbor_belief_t"]
        .value_counts(normalize=True)
        .rename("neighbor_fraction")
        .reset_index()
        .pivot(
            index=["run_dir", "round_t", "agent_id"],
            columns="neighbor_belief_t",
            values="neighbor_fraction",
        )
        .reset_index()
    )

    frac_df = frac_df.rename(
        columns={
            -1: "neighbor_frac_neg1_t",
            0: "neighbor_frac_0_t",
            1: "neighbor_frac_1_t",
        }
    )

    for col in ["neighbor_frac_neg1_t", "neighbor_frac_0_t", "neighbor_frac_1_t"]:
        if col not in frac_df.columns:
            frac_df[col] = 0.0

    count_df = (
        valid_neighbor_rows.groupby(["run_dir", "round_t", "agent_id"])["neighbor_belief_t"]
        .count()
        .reset_index(name="n_neighbors_with_observed_belief_t")
    )

    neighbor_features_df = frac_df.merge(
        count_df,
        on=["run_dir", "round_t", "agent_id"],
        how="left",
    )

    # Compute continuous neighbor mean P(true) if available.
    _extra_merge_cols: list[str] = []
    if "neighbor_belief_score_true_t" in valid_neighbor_rows.columns:
        mean_score_df = (
            valid_neighbor_rows.groupby(["run_dir", "round_t", "agent_id"])[
                "neighbor_belief_score_true_t"
            ]
            .mean()
            .reset_index(name="neighbor_mean_belief_score_true_t")
        )
        neighbor_features_df = neighbor_features_df.merge(
            mean_score_df, on=["run_dir", "round_t", "agent_id"], how="left"
        )
        _extra_merge_cols = ["neighbor_mean_belief_score_true_t"]

    result = transition_df.merge(
        neighbor_features_df[
            [
                "run_dir",
                "round_t",
                "agent_id",
                "neighbor_frac_neg1_t",
                "neighbor_frac_0_t",
                "neighbor_frac_1_t",
                "n_neighbors_with_observed_belief_t",
                *_extra_merge_cols,
            ]
        ],
        on=["run_dir", "round_t", "agent_id"],
        how="left",
    )

    # Fill isolated nodes / missing cases with 0.
    result["neighbor_frac_neg1_t"] = result["neighbor_frac_neg1_t"].fillna(0.0)
    result["neighbor_frac_0_t"] = result["neighbor_frac_0_t"].fillna(0.0)
    result["neighbor_frac_1_t"] = result["neighbor_frac_1_t"].fillna(0.0)
    result["n_neighbors_with_observed_belief_t"] = (
        result["n_neighbors_with_observed_belief_t"].fillna(0).astype(int)
    )

    return result


def write_dataframe(df: pd.DataFrame, output_path: Path, write_csv: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    if write_csv:
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)


def main() -> None:
    args = parse_args()

    run_dirs = find_run_dirs(args.input_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {args.input_root}.")

    trajectory_rows: list[TrajectoryRow] = []
    round_rows: list[RoundRow] = []
    transition_rows: list[TransitionRow] = []
    run_round_metrics_rows: list[RunRoundMetricsRow] = []
    edge_rows: list[EdgeRow] = []
    agent_network_rows: list[AgentNetworkRow] = []
    failures: list[str] = []

    for run_dir in run_dirs:
        try:
            meta = parse_run_metadata(run_dir, args.input_root)
            agents_data = load_json(run_dir / "results/agents_data.json")
            per_round_metrics = load_json(run_dir / "results/per_round_metrics.json")
            network_edges = load_json(run_dir / "results/network_edges.json")

            current_edge_rows = build_edge_rows(network_edges, meta)
            current_agent_network_rows = build_agent_network_rows(
                network_edges,
                meta,
                agent_ids=agents_data.keys(),
            )
            current_agent_network_lookup = {
                row.agent_id: {
                    "degree": row.degree,
                    "neighbor_ids": row.neighbor_ids,
                }
                for row in current_agent_network_rows
            }

            edge_rows.extend(current_edge_rows)
            agent_network_rows.extend(current_agent_network_rows)
            trajectory_rows.extend(build_trajectory_rows(agents_data, meta))
            round_rows.extend(
                build_round_rows(
                    agents_data,
                    meta,
                    agent_network_lookup=current_agent_network_lookup,
                )
            )
            transition_rows.extend(
                build_transition_rows(
                    agents_data,
                    meta,
                    agent_network_lookup=current_agent_network_lookup,
                )
            )
            run_round_metrics_rows.extend(
                build_run_round_metrics_rows(per_round_metrics, meta)
            )
        except SeedMetadataError:
            raise
        except Exception as exc:  # noqa: BLE001
            message = f"Failed to parse {run_dir}: {exc}"
            if args.strict:
                raise
            failures.append(message)
            print(message)

    trajectory_df = rows_to_dataframe(trajectory_rows)
    round_df = rows_to_dataframe(round_rows)
    transition_df = rows_to_dataframe(transition_rows)

    # add additional model-specific features
    transition_df = add_global_prevalence_features(transition_df)
    transition_df = add_neighbor_prevalence_features(transition_df, round_df)

    run_round_metrics_df = rows_to_dataframe(run_round_metrics_rows)
    edge_df = rows_to_dataframe(edge_rows)
    agent_network_df = rows_to_dataframe(agent_network_rows)

    write_dataframe(
        trajectory_df,
        args.output_dir / "trajectories.parquet",
        write_csv=args.write_csv,
    )
    write_dataframe(
        round_df,
        args.output_dir / "rounds.parquet",
        write_csv=args.write_csv,
    )
    write_dataframe(
        transition_df,
        args.output_dir / "transitions.parquet",
        write_csv=args.write_csv,
    )
    write_dataframe(
        run_round_metrics_df,
        args.output_dir / "run_round_metrics.parquet",
        write_csv=args.write_csv,
    )
    write_dataframe(
        edge_df,
        args.output_dir / "edges.parquet",
        write_csv=args.write_csv,
    )
    write_dataframe(
        agent_network_df,
        args.output_dir / "agent_network.parquet",
        write_csv=args.write_csv,
    )

    manifest = {
        "input_root": str(args.input_root),
        "output_dir": str(args.output_dir),
        "n_runs": len(run_dirs),
        "n_failures": len(failures),
        "failures": failures,
        "tables": {
            "trajectories": len(trajectory_df),
            "rounds": len(round_df),
            "transitions": len(transition_df),
            "run_round_metrics": len(run_round_metrics_df),
            "edges": len(edge_df),
            "agent_network": len(agent_network_df),
        },
        "transition_feature_columns_added": [
            "global_frac_neg1_t",
            "global_frac_0_t",
            "global_frac_1_t",
            "n_agents_in_run",
            "neighbor_frac_neg1_t",
            "neighbor_frac_0_t",
            "neighbor_frac_1_t",
            "n_neighbors_with_observed_belief_t",
        ],
    }

    manifest_path = args.output_dir / "aggregation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()