"""
To fit on all transitions:
    python fit_full_trajectories.py --model m1 --experiment_group experts --graph_type erdos-renyi --save_model

To fit on all transitions except round 0 to 1:
    python fit_full_trajectories.py --model m1 --experiment_group experts --graph_type erdos-renyi --save_model --exclude_round0_to_1

With uv:
uv run python src/analysis/dynamics_model_fitting/fit_full_trajectories.py  --model m1  --experiment_group random_experts  --graph_type erdos-renyi  --save_model  --exclude_round0_to_1
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from models.m1_persistence import M1PersistenceModel
from models.m2_global_drift import M2GlobalDriftModel
from models.m3_multinomial_logit import M3MultinomialLogitSocialInfluenceModel
from models.m25_global_drift_persistence import M25GlobalDriftPersistenceModel
from models.m4_social_influence_identity import M4SocialInfluenceIdentityModel

from utils import (
    DATA_DIR,
    OUTPUT_DIR,
    evaluate_predictions,
    evaluate_trajectory_fit_predictions,
    load_run_round_metrics,
    load_transitions,
    train_val_test_split_by_run,
)

MODEL_REGISTRY = {
    "m1": M1PersistenceModel,
    "m2": M2GlobalDriftModel,
    "m2-5": M25GlobalDriftPersistenceModel,
    "m3": M3MultinomialLogitSocialInfluenceModel,
    "m4": M4SocialInfluenceIdentityModel
}

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for full-trajectory fitting.

    Returns:
        Parsed namespace with model choice, filters, output options,
        random seed, and round-exclusion configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR / "full_trajectories",
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="If set, save the trained model to disk.",
    )
    parser.add_argument(
        "--experiment_group",
        type=str,
        default="all",
        help=(
            "Experiment group to use: one of "
            "{base_llms, experts, random_experts, random_roles, all}."
        ),
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="all",
        help=(
            "Graph type to use: one of "
            "{erdos-renyi, watts-strogatz, all}."
        ),
    )
    parser.add_argument(
        "--exclude_round0_to_1",
        action="store_true",
        help=(
            "If set, drop round 0 from the round-level data and exclude the "
            "0->1 transition regime from the transition-level data."
        ),
    )
    return parser.parse_args()


def load_rounds() -> pd.DataFrame:
    """Load round-level data and add trajectory identifiers.

    Returns:
        Round-level dataframe with an added `trajectory_id` column formed as
        `run_dir__agent_id`.
    """
    df = pd.read_parquet(DATA_DIR / "rounds.parquet")
    df["trajectory_id"] = df["run_dir"] + "__" + df["agent_id"].astype(str)
    return df


def merge_global_features(
    transitions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge run-level round metrics onto transition rows at round t.

    Args:
        transitions_df: Transition-level rows keyed by (`run_dir`, `round_t`).
        metrics_df: Run-round metrics keyed by (`run_dir`, `round_idx`).

    Returns:
        Transition dataframe augmented with global metrics aligned to round t.
    """
    metrics_t = metrics_df.rename(
        columns={
            "round_idx": "round_t",
            "entropy": "entropy_t",
            "multiclass_entropy": "multiclass_entropy_t",
            "magnetization": "magnetization_t",
            "polarity": "polarity_t",
            "consensus_fraction": "consensus_fraction_t",
            "belief_label_mean": "belief_label_mean_t",
            "belief_score_mean": "belief_score_mean_t",
        }
    )

    merge_cols = ["run_dir", "round_t"]
    keep_cols = [
        *merge_cols,
        "entropy_t",
        "multiclass_entropy_t",
        "magnetization_t",
        "polarity_t",
        "consensus_fraction_t",
        "belief_label_mean_t",
        "belief_score_mean_t",
    ]

    return transitions_df.merge(metrics_t[keep_cols], on=merge_cols, how="left")


def filter_transition_dataset(
    df: pd.DataFrame,
    experiment_group: str = "all",
    graph_type: str = "all",
    exclude_round0_to_1: bool = False,
) -> pd.DataFrame:
    """Filter the transition dataframe according to user options.

    Args:
        df: Transition-level dataframe.
        experiment_group: Experiment subset to keep, or "all".
        graph_type: Graph topology subset to keep, or "all".
        exclude_round0_to_1: Whether to drop transitions from round 0 to 1.

    Returns:
        Filtered transition dataframe.
    """
    filtered_df = df.copy()

    if experiment_group != "all":
        filtered_df = filtered_df[
            filtered_df["experiment_group"] == experiment_group
        ].copy()

    if graph_type != "all":
        filtered_df = filtered_df[
            filtered_df["graph_type"] == graph_type
        ].copy()

    if exclude_round0_to_1:
        filtered_df = filtered_df[
            ~(
                (filtered_df["round_t"] == 0)
                & (filtered_df["round_t1"] == 1)
            )
        ].copy()

    return filtered_df


def filter_round_dataset(
    df: pd.DataFrame,
    experiment_group: str = "all",
    graph_type: str = "all",
    exclude_round0_to_1: bool = False,
) -> pd.DataFrame:
    """Filter the round dataframe according to user options.

    Args:
        df: Round-level dataframe.
        experiment_group: Experiment subset to keep, or "all".
        graph_type: Graph topology subset to keep, or "all".
        exclude_round0_to_1: Whether to remove round 0 rows.

    Returns:
        Filtered round dataframe.
    """
    filtered_df = df.copy()

    if experiment_group != "all":
        filtered_df = filtered_df[
            filtered_df["experiment_group"] == experiment_group
        ].copy()

    if graph_type != "all":
        filtered_df = filtered_df[
            filtered_df["graph_type"] == graph_type
        ].copy()

    if exclude_round0_to_1:
        filtered_df = filtered_df[filtered_df["round_idx"] >= 1].copy()

    return filtered_df


def subset_df_to_runs(df: pd.DataFrame, run_ids: set[str]) -> pd.DataFrame:
    """Subset a dataframe to a set of run IDs.

    Args:
        df: Input dataframe with a `run_dir` column.
        run_ids: Set of run identifiers to keep.

    Returns:
        Subset dataframe restricted to the selected runs.
    """
    return df[df["run_dir"].isin(run_ids)].copy()


def build_run_tag(args: argparse.Namespace) -> str:
    """Build a descriptive tag for output artifacts.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Stable run tag encoding model, experiment filters, graph type,
        and transition-regime choice.
    """
    experiment_tag = args.experiment_group
    graph_tag = args.graph_type
    transition_tag = "exclude01" if args.exclude_round0_to_1 else "alltrans"

    return (
        f"{args.model}"
        f"__exp-{experiment_tag}"
        f"__graph-{graph_tag}"
        f"__trans-{transition_tag}"
    )


def save_split_manifest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    run_tag: str,
    timestamp: str,
    model_name: str,
) -> Path:
    """Save a trajectory-level split manifest.

    Args:
        train_df: Train transition dataframe.
        val_df: Validation transition dataframe.
        test_df: Test transition dataframe.
        output_dir: Root output directory.
        run_tag: Descriptive run tag.
        timestamp: Run timestamp string.
        model_name: Model name used in path construction.

    Returns:
        Path to the saved CSV manifest containing trajectory split labels.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    split_path = output_dir / model_name / f"{run_tag}__{timestamp}__splits.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)

    split_df[["trajectory_id", "split"]].drop_duplicates().to_csv(
        split_path,
        index=False,
    )
    return split_path


def build_trajectory_dict(round_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build a dictionary of trajectories keyed by trajectory_id.

    Args:
        round_df: Round-level dataframe containing `trajectory_id` and
            `round_idx`.

    Returns:
        Mapping from trajectory id to round-sorted trajectory dataframe.
    """
    trajectory_dict: dict[str, pd.DataFrame] = {}

    grouped = round_df.sort_values(["trajectory_id", "round_idx"]).groupby("trajectory_id")
    for trajectory_id, traj_df in grouped:
        trajectory_dict[str(trajectory_id)] = traj_df.copy()

    return trajectory_dict


def build_train_payload(
    train_transitions_df: pd.DataFrame,
    val_transitions_df: pd.DataFrame,
    test_transitions_df: pd.DataFrame,
    train_rounds_df: pd.DataFrame,
    val_rounds_df: pd.DataFrame,
    test_rounds_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build the standardized payload passed to fit_full_trajectories.

    Args:
        train_transitions_df: Train split transition rows.
        val_transitions_df: Validation split transition rows.
        test_transitions_df: Test split transition rows.
        train_rounds_df: Train split round rows.
        val_rounds_df: Validation split round rows.
        test_rounds_df: Test split round rows.

    Returns:
        Payload containing transition splits, round splits, and pre-grouped
        trajectory dictionaries for each split.
    """
    return {
        "train_transitions_df": train_transitions_df,
        "val_transitions_df": val_transitions_df,
        "test_transitions_df": test_transitions_df,
        "train_rounds_df": train_rounds_df,
        "val_rounds_df": val_rounds_df,
        "test_rounds_df": test_rounds_df,
        "train_trajectories": build_trajectory_dict(train_rounds_df),
        "val_trajectories": build_trajectory_dict(val_rounds_df),
        "test_trajectories": build_trajectory_dict(test_rounds_df),
    }


def compute_split_metrics(
    model: object,
    split_df: pd.DataFrame,
) -> dict[str, dict[str, float | None]]:
    """Compute transition and trajectory-fit metrics for one split.

    Args:
        model: Fitted dynamics model implementing prediction methods.
        split_df: Transition rows for one split.

    Returns:
        Dict with `transition_metrics` and `trajectory_fit_metrics`.
    """
    y_true = split_df["belief_t1"]
    y_pred = model.predict_next(split_df)

    try:
        y_proba = model.predict_proba_next(split_df)
    except NotImplementedError:
        y_proba = None

    transition_metrics = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    trajectory_fit_metrics = evaluate_trajectory_fit_predictions(
        df=split_df,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        trajectory_col="trajectory_id",
        round_col="round_t1",
    )

    return {
        "transition_metrics": transition_metrics,
        "trajectory_fit_metrics": trajectory_fit_metrics,
    }


def fit_and_evaluate_model(
    model_name: str,
    train_transitions_df: pd.DataFrame,
    val_transitions_df: pd.DataFrame,
    test_transitions_df: pd.DataFrame,
    train_rounds_df: pd.DataFrame,
    val_rounds_df: pd.DataFrame,
    test_rounds_df: pd.DataFrame,
) -> tuple[object, dict[str, dict[str, dict[str, float | None]]]]:
    """Fit the requested model on full trajectories and evaluate on splits.

    Args:
        model_name: Registry key for the model.
        train_transitions_df: Train split transition rows.
        val_transitions_df: Validation split transition rows.
        test_transitions_df: Test split transition rows.
        train_rounds_df: Train split round rows.
        val_rounds_df: Validation split round rows.
        test_rounds_df: Test split round rows.

    Returns:
        Tuple of fitted model instance and nested metrics per split.
    """
    model = MODEL_REGISTRY[model_name]()

    train_payload = build_train_payload(
        train_transitions_df=train_transitions_df,
        val_transitions_df=val_transitions_df,
        test_transitions_df=test_transitions_df,
        train_rounds_df=train_rounds_df,
        val_rounds_df=val_rounds_df,
        test_rounds_df=test_rounds_df,
    )

    model.fit_full_trajectories(train_payload)

    split_results = {
        "train": compute_split_metrics(model, train_transitions_df),
        "val": compute_split_metrics(model, val_transitions_df),
        "test": compute_split_metrics(model, test_transitions_df),
    }

    return model, split_results


def main() -> None:
    """Run the full trajectory-fitting and evaluation pipeline.

    Pipeline overview:
        1. Load transitions, round metrics, and round-level trajectories.
        2. Merge global metrics into transition rows.
        3. Apply user-selected experiment/graph/round filters.
        4. Split by run id into train/val/test.
        5. Build trajectory payload and fit chosen model.
        6. Evaluate transition-level and trajectory-fit metrics.
        7. Optionally save model artifacts and always save JSON results.
    """
    warnings.filterwarnings("ignore")

    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    transitions_df = load_transitions()
    metrics_df = load_run_round_metrics()
    rounds_df = load_rounds()
    transitions_df = merge_global_features(transitions_df, metrics_df)
    transitions_df = filter_transition_dataset(
        transitions_df,
        experiment_group=args.experiment_group,
        graph_type=args.graph_type,
        exclude_round0_to_1=args.exclude_round0_to_1,
    )
    if transitions_df["seed"].isna().any():
        missing_seed_runs = transitions_df.loc[
            transitions_df["seed"].isna(), "run_dir"
        ].dropna().astype(str).unique()
        example_missing = ", ".join(sorted(missing_seed_runs)[:3])
        raise ValueError(
            "Seed cannot be null in filtered transitions. "
            f"Found {len(missing_seed_runs)} run(s) with missing seed. "
            f"Example run_dir values: {example_missing}"
        )
    transitions_df = transitions_df.sort_values(
        ["run_dir", "agent_id", "round_t"]
    ).reset_index(drop=True)

    rounds_df = filter_round_dataset(
        rounds_df,
        experiment_group=args.experiment_group,
        graph_type=args.graph_type,
        exclude_round0_to_1=args.exclude_round0_to_1,
    )
    rounds_df = rounds_df.sort_values(
        ["run_dir", "agent_id", "round_idx"]
    ).reset_index(drop=True)

    if transitions_df.empty:
        raise ValueError("Filtered transition dataframe is empty. Check your filter arguments.")
    if rounds_df.empty:
        raise ValueError("Filtered round dataframe is empty. Check your filter arguments.")

    train_transitions_df, val_transitions_df, test_transitions_df = train_val_test_split_by_run(
        transitions_df,
        random_state=args.random_state,
    )

    train_run_ids = set(train_transitions_df["run_dir"].unique())
    val_run_ids = set(val_transitions_df["run_dir"].unique())
    test_run_ids = set(test_transitions_df["run_dir"].unique())

    train_rounds_df = subset_df_to_runs(rounds_df, train_run_ids)
    val_rounds_df = subset_df_to_runs(rounds_df, val_run_ids)
    test_rounds_df = subset_df_to_runs(rounds_df, test_run_ids)

    run_tag = build_run_tag(args)

    split_path = save_split_manifest(
        train_df=train_transitions_df,
        val_df=val_transitions_df,
        test_df=test_transitions_df,
        output_dir=args.output_dir,
        run_tag=run_tag,
        timestamp=timestamp,
        model_name=args.model,
    )

    # print(train_transitions_df.head()['seed'])
    # print(train_rounds_df.head())
    # raise Exception("Debug stop - check the loaded dataframes before fitting the model.")

    model, split_results = fit_and_evaluate_model(
        model_name=args.model,
        train_transitions_df=train_transitions_df,
        val_transitions_df=val_transitions_df,
        test_transitions_df=test_transitions_df,
        train_rounds_df=train_rounds_df,
        val_rounds_df=val_rounds_df,
        test_rounds_df=test_rounds_df,
    )

    if args.save_model:
        model_dir = args.output_dir / args.model / "saved_models"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{run_tag}__{timestamp}.joblib"
        joblib.dump(
            {
                "model": model,
                "model_name": args.model,
                "timestamp": timestamp,
                "random_state": args.random_state,
                "config": model.get_params(),
                "split_path": str(split_path),
                "experiment_group": args.experiment_group,
                "graph_type": args.graph_type,
                "exclude_round0_to_1": args.exclude_round0_to_1,
                "fit_objective": "full_trajectories",
            },
            model_path,
        )
        print(f"Saved model to: {model_path}")

    results_payload = {
        "model": args.model,
        "timestamp": timestamp,
        "random_state": args.random_state,
        "experiment_group": args.experiment_group,
        "graph_type": args.graph_type,
        "exclude_round0_to_1": args.exclude_round0_to_1,
        "fit_objective": "full_trajectories",
        "n_transition_rows_total": len(transitions_df),
        "n_round_rows_total": len(rounds_df),
        "n_train_transition_rows": len(train_transitions_df),
        "n_val_transition_rows": len(val_transitions_df),
        "n_test_transition_rows": len(test_transitions_df),
        "n_train_round_rows": len(train_rounds_df),
        "n_val_round_rows": len(val_rounds_df),
        "n_test_round_rows": len(test_rounds_df),
        "n_train_runs": len(train_run_ids),
        "n_val_runs": len(val_run_ids),
        "n_test_runs": len(test_run_ids),
        "n_train_trajectories": train_rounds_df["trajectory_id"].nunique(),
        "n_val_trajectories": val_rounds_df["trajectory_id"].nunique(),
        "n_test_trajectories": test_rounds_df["trajectory_id"].nunique(),
        "metrics": split_results,
    }

    outpath = args.output_dir / args.model / f"{run_tag}__{timestamp}__results.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)

    with outpath.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(json.dumps(results_payload, indent=2))


if __name__ == "__main__":
    main()
