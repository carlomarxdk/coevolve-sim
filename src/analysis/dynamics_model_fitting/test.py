'''
from pathlib import Path
from trajectory_utils import (
    load_saved_model_bundle,
    get_eval_rounds_for_saved_model,
    get_initial_states_by_run,
    load_rounds,
)
from simulation import simulate_one_run
from evaluate_trajectory_behavior import summarize_round_behavior, compare_empirical_vs_simulated

print('\n\n\n===LOAD TEST===\n\n\n')

bundle = load_saved_model_bundle(Path("outputs/single_transitions/m1/saved_models/m1__exp-random_experts__graph-erdos-renyi__trans-alltrans__20260415_163328.joblib"))
eval_rounds_df = get_eval_rounds_for_saved_model(bundle=bundle, split_name="test")

print(eval_rounds_df.shape)
print(eval_rounds_df.columns.tolist())
print(eval_rounds_df["round_idx"].value_counts().sort_index())

initial_states_by_run = get_initial_states_by_run(eval_rounds_df, start_round=0)
print(len(initial_states_by_run))

first_run = next(iter(initial_states_by_run))
print(first_run)
print(initial_states_by_run[first_run].head())

print('\n\n\n===SIMULATE TEST===\n\n\n')

model = bundle["model"]

one_run_dir = next(iter(initial_states_by_run))
one_run_init = initial_states_by_run[one_run_dir]

sim_df = simulate_one_run(
    model=model,
    initial_state_df=one_run_init,
    n_steps=10,
)

print(sim_df.shape)
print(sim_df.columns.tolist())
print(sim_df["round_idx"].value_counts().sort_index())
print(sim_df.head())

print('\n\n\n===COMPARE TO EMPIRICAL===\n\n\n')

emp_one_run = eval_rounds_df[eval_rounds_df["run_dir"] == one_run_dir].copy()

print(emp_one_run.shape)
print(emp_one_run["agent_id"].nunique())
print(emp_one_run["round_idx"].value_counts().sort_index())

print(emp_one_run[["round_idx", "belief_label"]].head(50))
print(emp_one_run.groupby("round_idx")["belief_label"].nunique())

rounds_df = load_rounds()

test_run = one_run_dir
full_run = rounds_df[rounds_df["run_dir"] == test_run]

print(full_run.groupby("round_idx")["belief_label"].mean())

print('\n\n\n===EVALUATE TEST===\n\n\n')

emp_summary = summarize_round_behavior(emp_one_run, belief_col="belief_label")
sim_summary = summarize_round_behavior(sim_df, belief_col="belief_t")

print(emp_summary.head())
print(sim_summary.head())

comparison = compare_empirical_vs_simulated(
    empirical_df=emp_one_run,
    simulated_df=sim_df,
)

print(comparison.head())
print(comparison.filter(like="_abs_error").mean())
'''

from pathlib import Path
import json

import pandas as pd


DATA_DIR = Path("data")

IDENTITY_CANDIDATE_COLS = [
    "agent_id",
    "model_name",
    "model",
    "llm_name",
    "llm",
    "agent_model",
    "role",
    "role_label",
    "agent_role",
    "agent_type",
    "identity",
    "condition",
    "experiment_group",
]


FILES_TO_INSPECT = [
    "agent_network.parquet",
    "edges.parquet",
    "rounds.parquet",
    "run_round_metrics.parquet",
    "trajectories.parquet",
    "transitions.parquet",
]


def inspect_parquet_file(path: Path) -> None:
    print(f"\n\n{'=' * 80}")
    print(f"FILE: {path}")
    print(f"{'=' * 80}")

    if not path.exists():
        print("MISSING")
        return

    df = pd.read_parquet(path)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    present_identity_cols = [
        col for col in IDENTITY_CANDIDATE_COLS if col in df.columns
    ]

    print("\nCandidate identity columns present:")
    print(present_identity_cols if present_identity_cols else "NONE")

    if present_identity_cols:
        print("\nUnique counts:")
        for col in present_identity_cols:
            print(f"  {col}: {df[col].nunique(dropna=False)}")

        print("\nSample values:")
        for col in present_identity_cols:
            vals = df[col].dropna().astype(str).unique()[:20]
            print(f"  {col}: {list(vals)}")

        print("\nMissing counts:")
        for col in present_identity_cols:
            print(f"  {col}: {df[col].isna().sum()}")

    if "agent_id" in df.columns and present_identity_cols:
        identity_cols = [
            col for col in present_identity_cols if col != "agent_id"
        ]

        if identity_cols:
            print("\nIdentity stability within agent_id:")
            for col in identity_cols:
                per_agent_unique = df.groupby("agent_id")[col].nunique(dropna=False)
                print(
                    f"  {col}: max unique values per agent_id = "
                    f"{per_agent_unique.max()}"
                )

            display_cols = ["agent_id", *identity_cols]
            print("\nFirst agent_id / identity mappings:")
            print(df[display_cols].drop_duplicates().head(50))


def inspect_manifest(path: Path) -> None:
    print(f"\n\n{'=' * 80}")
    print(f"FILE: {path}")
    print(f"{'=' * 80}")

    if not path.exists():
        print("MISSING")
        return

    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(json.dumps(manifest, indent=2)[:5000])


def main() -> None:
    for filename in FILES_TO_INSPECT:
        inspect_parquet_file(DATA_DIR / filename)

    inspect_manifest(DATA_DIR / "aggregation_manifest.json")


if __name__ == "__main__":
    main()
