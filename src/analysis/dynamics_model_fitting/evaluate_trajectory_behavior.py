"""
To fit on all transitions:
    python evaluate_trajectory_behavior.py \
      --fit_mode single_transitions \
      --model m1 \
      --experiment_group experts \
      --graph_type erdos-renyi \
      --exclude_round0_to_1 false \
      --match_mode latest \
      --save_level summary

To fit on all transitions except round 0 to 1:
    python evaluate_trajectory_behavior.py \
      --fit_mode single_transitions \
      --model m1 \
      --experiment_group experts \
      --graph_type erdos-renyi \
      --exclude_round0_to_1 true \
      --match_mode latest \
      --save_level summary

Note: to use trajectory models instead of single-transition models, use --fit_mode full_trajectories
"""

import argparse
import json
from pathlib import Path
from tkinter import N
from typing import Any
import warnings
import numpy as np
import pandas as pd

from simulation import simulate_many_runs
from trajectory_utils import (
    get_eval_rounds_for_saved_model,
    get_initial_states_by_run,
    load_saved_model_bundle,
)
from utils import OUTPUT_DIR


ROLLOUT_MODES = ["deterministic", "stochastic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fit_mode",
        type=str,
        required=True,
        choices=["single_transitions", "full_trajectories"],
        help="Which fitted-model directory to search.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["m1", "m2", "m2-5", "m3", "m4"],
        help="Model family to evaluate.",
    )
    parser.add_argument(
        "--experiment_group",
        type=str,
        default="all",
        help=(
            "Experiment group to match in saved model metadata: one of "
            "{base_llms, experts, random_experts, random_roles, all}."
        ),
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="all",
        help="Graph type to match in saved model metadata: {erdos-renyi, watts-strogatz, all}.",
    )
    parser.add_argument(
        "--exclude_round0_to_1",
        type=str,
        default="all",
        choices=["true", "false", "all"],
        help=(
            "Which trained-model regime to match. "
            "'false' means model trained including 0->1 transitions; "
            "'true' means model trained excluding them; "
            "'all' matches both."
        ),
    )
    parser.add_argument(
        "--match_mode",
        type=str,
        default="latest",
        choices=["latest", "all"],
        help="Whether to evaluate only the latest matching saved model or all matching saved models.",
    )
    parser.add_argument(
        "--save_level",
        type=str,
        default="summary",
        choices=["summary", "full"],
        help=(
            "'summary' saves only one summary parquet with two rows per matched model "
            "(deterministic/stochastic). "
            "'full' additionally saves simulated/comparison parquet files for each rollout mode."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which saved split to evaluate.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Base random state used for stochastic rollouts.",
    )
    parser.add_argument(
        "--start_round",
        type=int,
        default=None,
        help=(
            "Optional manual override. If omitted, start_round is inferred from the "
            "saved model metadata: 0 if exclude_round0_to_1=False, 1 if True."
        ),
    )

    return parser.parse_args()


def summarize_round_behavior(df: pd.DataFrame, belief_col: str) -> pd.DataFrame:
    """Compute round-level global summary metrics."""
    summary = (
        df.groupby(["run_dir", "round_idx"])[belief_col]
        .agg(["mean", "var"])
        .reset_index()
        .rename(columns={"mean": "belief_mean", "var": "belief_var"})
    )

    state_frac = (
        df.groupby(["run_dir", "round_idx"])[belief_col]
        .value_counts(normalize=True)
        .rename("fraction")
        .reset_index()
        .pivot(index=["run_dir", "round_idx"], columns=belief_col, values="fraction")
        .reset_index()
    )

    state_frac = state_frac.rename(
        columns={
            -1: "frac_neg1",
            0: "frac_0",
            1: "frac_1",
        }
    )

    for col in ["frac_neg1", "frac_0", "frac_1"]:
        if col not in state_frac.columns:
            state_frac[col] = 0.0

    out = summary.merge(state_frac, on=["run_dir", "round_idx"], how="left")
    out["consensus_fraction"] = out[["frac_neg1", "frac_0", "frac_1"]].max(axis=1)
    out["polarity"] = (out["frac_neg1"] + out["frac_1"]).fillna(0.0)

    return out


def compare_empirical_vs_simulated(
    empirical_df: pd.DataFrame,
    simulated_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare empirical and simulated trajectories at the run-round level."""
    empirical_summary = summarize_round_behavior(empirical_df, belief_col="belief_label")
    empirical_summary = empirical_summary.add_prefix("emp_")
    empirical_summary = empirical_summary.rename(
        columns={"emp_run_dir": "run_dir", "emp_round_idx": "round_idx"}
    )

    simulated_summary = summarize_round_behavior(simulated_df, belief_col="belief_t")
    simulated_summary = simulated_summary.add_prefix("sim_")
    simulated_summary = simulated_summary.rename(
        columns={"sim_run_dir": "run_dir", "sim_round_idx": "round_idx"}
    )

    comparison = empirical_summary.merge(
        simulated_summary,
        on=["run_dir", "round_idx"],
        how="inner",
    )

    metric_names = [
        "belief_mean",
        "belief_var",
        "frac_neg1",
        "frac_0",
        "frac_1",
        "consensus_fraction",
        "polarity",
    ]
    for metric in metric_names:
        comparison[f"{metric}_abs_error"] = (
            comparison[f"emp_{metric}"] - comparison[f"sim_{metric}"]
        ).abs()

    return comparison


def get_nontrivial_run_ids(empirical_df: pd.DataFrame) -> list[str]:
    """Return run_ids with at least one empirical state transition over time."""
    sorted_df = empirical_df.sort_values(["run_dir", "agent_id", "round_idx"]).copy()
    sorted_df["prev_belief"] = (
        sorted_df.groupby(["run_dir", "agent_id"])["belief_label"].shift(1)
    )
    sorted_df["changed"] = (
        sorted_df["prev_belief"].notna()
        & (sorted_df["belief_label"] != sorted_df["prev_belief"])
    )

    run_change_flags = sorted_df.groupby("run_dir")["changed"].any()
    return run_change_flags[run_change_flags].index.tolist()

def summarize_comparison_metrics(
    comparison_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05,
) -> dict[str, dict[str, float] | None]:
    """Compute mean absolute errors with bootstrap CIs, resampling over runs.

    Each run is treated as an atomic unit — rows within a run share a
    simulated trajectory and are not independent.

    Args:
        comparison_df: Output of ``compare_empirical_vs_simulated``.
        n_bootstrap: Number of bootstrap resamples over runs.
        random_state: RNG seed.
        alpha: Significance level; produces a (1-alpha) CI.

    Returns:
        Mapping from metric column name to dict with keys ``point``,
        ``ci_lower``, ``ci_upper``, or ``None`` if uncomputable.
    """
    metric_cols = [c for c in comparison_df.columns if c.endswith("_abs_error")]

    if comparison_df.empty:
        return {col: None for col in metric_cols}

    # Compute per-run means first — this is what we resample over
    per_run = (
        comparison_df.groupby("run_dir")[metric_cols]
        .mean()
    )  # (n_runs, k)

    run_vals = per_run.to_numpy(dtype=float)  # (n_runs, k)
    n_runs = len(run_vals)

    rng = np.random.default_rng(random_state)
    boot_idx = rng.integers(0, n_runs, size=(n_bootstrap, n_runs))  # (B, n_runs)

    # All bootstrap means at once: (B, k)
    boot_means = np.nanmean(run_vals[boot_idx], axis=1)

    lower_p = 100 * alpha / 2
    upper_p = 100 * (1 - alpha / 2)

    results: dict[str, dict[str, float] | None] = {}
    point_means = np.nanmean(run_vals, axis=0)  # (k,)

    for i, col in enumerate(metric_cols):
        pt = point_means[i]
        if np.isnan(pt):
            results[col] = None
            continue
        valid = boot_means[:, i]
        valid = valid[~np.isnan(valid)]
        results[col] = {
            "point":    float(pt),
            "ci_lower": float(np.percentile(valid, lower_p)),
            "ci_upper": float(np.percentile(valid, upper_p)),
        }

    return results


def selector_tag(args: argparse.Namespace) -> str:
    return (
        f"exp-{args.experiment_group}"
        f"__graph-{args.graph_type}"
        f"__trans-{args.exclude_round0_to_1}"
        f"__split-{args.split}"
        f"__match-{args.match_mode}"
    )


def output_dir_for_args(args: argparse.Namespace) -> Path:
    return OUTPUT_DIR / args.fit_mode / args.model / "trajectory_behavior"


def saved_models_dir(args: argparse.Namespace) -> Path:
    return OUTPUT_DIR / args.fit_mode / args.model / "saved_models"


def normalize_bool_selector(value: str) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def bundle_matches_selector(bundle: dict[str, Any], args: argparse.Namespace) -> bool:
    if bundle.get("model_name") != args.model:
        return False

    if args.experiment_group != "all":
        if bundle.get("experiment_group") != args.experiment_group:
            return False

    if args.graph_type != "all":
        if bundle.get("graph_type") != args.graph_type:
            return False

    exclude_selector = normalize_bool_selector(args.exclude_round0_to_1)
    if exclude_selector is not None:
        if bundle.get("exclude_round0_to_1") != exclude_selector:
            return False

    return True


def find_matching_model_paths(args: argparse.Namespace) -> list[Path]:
    model_dir = saved_models_dir(args)
    if not model_dir.exists():
        raise FileNotFoundError(f"Saved model directory does not exist: {model_dir}")

    candidate_paths = sorted(model_dir.glob("*.joblib"))
    matches: list[tuple[Path, dict[str, Any]]] = []

    for path in candidate_paths:
        try:
            bundle = load_saved_model_bundle(path)
        except Exception:
            continue

        if bundle_matches_selector(bundle, args):
            matches.append((path, bundle))

    if not matches:
        raise FileNotFoundError(
            "No matching saved models found for selector: "
            f"fit_mode={args.fit_mode}, model={args.model}, "
            f"experiment_group={args.experiment_group}, "
            f"graph_type={args.graph_type}, "
            f"exclude_round0_to_1={args.exclude_round0_to_1}."
        )

    matches = sorted(matches, key=lambda x: x[0].name)

    if args.match_mode == "latest":
        return [matches[-1][0]]

    return [path for path, _ in matches]


def infer_start_round(bundle: dict[str, Any], start_round_override: int | None) -> int:
    if start_round_override is not None:
        return start_round_override

    exclude_round0_to_1 = bundle.get("exclude_round0_to_1")
    if exclude_round0_to_1 is True:
        return 1
    return 0


def flatten_summary_row(
    *,
    model_path: Path,
    bundle: dict[str, Any],
    split: str,
    start_round: int,
    rollout_mode: str,
    random_state: int,
    n_runs_evaluated: int,
    n_nontrivial_runs: int,
    all_run_metrics: dict[str, dict[str, float] | None],
    nontrivial_metrics: dict[str, dict[str, float] | None],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_path": str(model_path),
        "model_name": bundle.get("model_name"),
        "fit_timestamp": bundle.get("timestamp"),
        "experiment_group": bundle.get("experiment_group"),
        "graph_type": bundle.get("graph_type"),
        "exclude_round0_to_1": bundle.get("exclude_round0_to_1"),
        "split": split,
        "start_round": start_round,
        "rollout_mode": rollout_mode,
        "random_state": random_state,
        "n_runs_evaluated": n_runs_evaluated,
        "n_nontrivial_runs": n_nontrivial_runs,
    }

    def _add_metrics(metrics: dict, prefix: str) -> None:
        for key, value in metrics.items():
            if isinstance(value, dict):
                # Flatten point/ci_lower/ci_upper into separate columns
                row[f"{prefix}{key}__point"]    = value.get("point")
                row[f"{prefix}{key}__ci_lower"] = value.get("ci_lower")
                row[f"{prefix}{key}__ci_upper"] = value.get("ci_upper")
            else:
                row[f"{prefix}{key}"] = value

    _add_metrics(all_run_metrics, prefix="all_runs__")
    _add_metrics(nontrivial_metrics, prefix="nontrivial_runs_only__")

    return row


def evaluate_one_bundle(
    *,
    model_path: Path,
    bundle: dict[str, Any],
    split: str,
    start_round: int,
    rollout_mode: str,
    random_state: int,
    save_level: str,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame | None, pd.DataFrame | None]:
    model = bundle["model"]

    empirical_rounds_df = get_eval_rounds_for_saved_model(
        bundle=bundle,
        split_name=split,
    )
    empirical_rounds_df = empirical_rounds_df[
        empirical_rounds_df["round_idx"] >= start_round
    ].copy()

    initial_states_by_run = get_initial_states_by_run(
        empirical_rounds_df,
        start_round=start_round,
    )

    if empirical_rounds_df.empty or not initial_states_by_run:
        raise ValueError(
            "No empirical rounds or initial states found for the requested split/start_round."
        )

    max_round = int(empirical_rounds_df["round_idx"].max())
    n_steps = max_round - start_round

    simulated_df = simulate_many_runs(
        model=model,
        initial_states_by_run=initial_states_by_run,
        n_steps=n_steps,
        rollout_mode=rollout_mode,
        random_state=random_state,
    )

    comparison_df = compare_empirical_vs_simulated(
        empirical_df=empirical_rounds_df,
        simulated_df=simulated_df,
    )

    nontrivial_run_ids = get_nontrivial_run_ids(empirical_rounds_df)
    comparison_nontrivial_df = comparison_df[
        comparison_df["run_dir"].isin(nontrivial_run_ids)
    ].copy()

    summary_row = flatten_summary_row(
        model_path=model_path,
        bundle=bundle,
        split=split,
        start_round=start_round,
        rollout_mode=rollout_mode,
        random_state=random_state,
        n_runs_evaluated=int(comparison_df["run_dir"].nunique()),
        n_nontrivial_runs=int(len(nontrivial_run_ids)),
        all_run_metrics=summarize_comparison_metrics(comparison_df, n_bootstrap=1000, random_state=random_state, alpha=0.05),
        nontrivial_metrics=summarize_comparison_metrics(comparison_nontrivial_df, n_bootstrap=1000, random_state=random_state, alpha=0.05),
    )

    if save_level == "full":
        stem = model_path.stem
        base = (
            f"{stem}"
            f"__split-{split}"
            f"__start-{start_round}"
            f"__mode-{rollout_mode}"
        )

        simulated_path = output_dir / f"{base}__simulated_rounds.parquet"
        comparison_path = output_dir / f"{base}__comparison.parquet"

        simulated_df.to_parquet(simulated_path, index=False)
        comparison_df.to_parquet(comparison_path, index=False)

        summary_row["saved_simulated_rounds_parquet"] = str(simulated_path)
        summary_row["saved_comparison_parquet"] = str(comparison_path)

    return summary_row, simulated_df if save_level == "full" else None, comparison_df if save_level == "full" else None


def main() -> None:
    warnings.filterwarnings("ignore")

    args = parse_args()

    outdir = output_dir_for_args(args)
    outdir.mkdir(parents=True, exist_ok=True)

    matched_model_paths = find_matching_model_paths(args)

    summary_rows: list[dict[str, Any]] = []

    for model_idx, model_path in enumerate(matched_model_paths):
        bundle = load_saved_model_bundle(model_path)
        start_round = infer_start_round(bundle, args.start_round)

        for rollout_idx, rollout_mode in enumerate(ROLLOUT_MODES):
            summary_row, _, _ = evaluate_one_bundle(
                model_path=model_path,
                bundle=bundle,
                split=args.split,
                start_round=start_round,
                rollout_mode=rollout_mode,
                random_state=args.random_state + model_idx * 100 + rollout_idx,
                save_level=args.save_level,
                output_dir=outdir,
            )
            summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)

    summary_path = outdir / (
        f"{args.model}"
        f"__{selector_tag(args)}"
        f"__trajectory_behavior_summary.parquet"
    )
    summary_df.to_parquet(summary_path, index=False)

    print(summary_df)


if __name__ == "__main__":
    main()
