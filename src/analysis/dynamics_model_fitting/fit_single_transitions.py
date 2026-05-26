"""
To fit on all transitions:
    python fit_single_transitions.py --model m1 --experiment_group experts --graph_type erdos-renyi --save_model

To fit on all transitions except round 0 to 1:
    python fit_single_transitions.py --model m1 --experiment_group experts --graph_type erdos-renyi --save_model --exclude_round0_to_1
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from models.m1_persistence import M1PersistenceModel
from models.m2_global_drift import M2GlobalDriftModel
from models.m25_global_drift_persistence import M25GlobalDriftPersistenceModel
from models.m3_multinomial_logit import M3MultinomialLogitSocialInfluenceModel
from models.m4_social_influence_identity import M4SocialInfluenceIdentityModel
from utils import (
    OUTPUT_DIR,
    evaluate_predictions,
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR / "single_transitions",
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
        help="If set, exclude transitions where round_t=0 and round_t1=1.",
    )
    return parser.parse_args()


def merge_global_features(
    transitions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge run-level round metrics onto transition rows at round t."""
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
    keep_cols = merge_cols + [
        "entropy_t",
        "multiclass_entropy_t",
        "magnetization_t",
        "polarity_t",
        "consensus_fraction_t",
        "belief_label_mean_t",
        "belief_score_mean_t",
    ]

    return transitions_df.merge(metrics_t[keep_cols], on=merge_cols, how="left")


def filter_dataset(
    df: pd.DataFrame,
    experiment_group: str = "all",
    graph_type: str = "all",
    exclude_round0_to_1: bool = False,
) -> pd.DataFrame:
    """Filter the transition dataframe according to user options."""
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


def build_run_tag(args: argparse.Namespace) -> str:
    """Build a descriptive tag for outputs from this run."""
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
    """Save a trajectory-level split manifest."""
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


def fit_and_evaluate_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[object, dict[str, dict[str, float]]]:
    """Fit the requested model and evaluate on train/val/test."""
    model = MODEL_REGISTRY[model_name]()
    model.fit(train_df)

    split_results: dict[str, dict[str, float]] = {}
    split_frames = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for split_name, split_df in split_frames.items():
        y_true = split_df["belief_t1"]
        y_pred = model.predict_next(split_df)

        try:
            y_proba = model.predict_proba_next(split_df)
        except NotImplementedError:
            y_proba = None

        split_results[split_name] = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        )

    return model, split_results


def main() -> None:
    """Run the full transition-based fitting pipeline."""
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    transitions_df = load_transitions()
    metrics_df = load_run_round_metrics()

    df = merge_global_features(transitions_df, metrics_df)
    df = filter_dataset(
        df,
        experiment_group=args.experiment_group,
        graph_type=args.graph_type,
        exclude_round0_to_1=args.exclude_round0_to_1,
    )
    df = df.sort_values(["run_dir", "agent_id", "round_t"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Filtered dataframe is empty. Check your filter arguments.")

    train_df, val_df, test_df = train_val_test_split_by_run(
        df,
        random_state=args.random_state,
    )

    run_tag = build_run_tag(args)

    split_path = save_split_manifest(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=args.output_dir,
        run_tag=run_tag,
        timestamp=timestamp,
        model_name=args.model,
    )

    model, split_results = fit_and_evaluate_model(
        model_name=args.model,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
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
        "n_rows_total": len(df),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_train_trajectories": train_df["trajectory_id"].nunique(),
        "n_val_trajectories": val_df["trajectory_id"].nunique(),
        "n_test_trajectories": test_df["trajectory_id"].nunique(),
        "metrics": split_results,
    }

    outpath = args.output_dir / args.model / f"{run_tag}__{timestamp}__results.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)

    with outpath.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(json.dumps(results_payload, indent=2))


if __name__ == "__main__":
    main()
