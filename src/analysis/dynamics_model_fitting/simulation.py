import json
from typing import Any

import numpy as np
import pandas as pd


def compute_global_features_from_state(state_df: pd.DataFrame) -> pd.DataFrame:
    """Compute global prevalence features from the current simulated state."""
    prevalence_df = (
        state_df.groupby(["run_dir", "round_idx"])["belief_t"]
        .value_counts(normalize=True)
        .rename("global_fraction")
        .reset_index()
        .pivot(
            index=["run_dir", "round_idx"],
            columns="belief_t",
            values="global_fraction",
        )
        .reset_index()
    )

    prevalence_df = prevalence_df.rename(
        columns={
            -1: "global_frac_neg1_t",
            0: "global_frac_0_t",
            1: "global_frac_1_t",
        }
    )

    for col in ["global_frac_neg1_t", "global_frac_0_t", "global_frac_1_t"]:
        if col not in prevalence_df.columns:
            prevalence_df[col] = 0.0

    n_agents_df = (
        state_df.groupby(["run_dir", "round_idx"])["agent_id"]
        .nunique()
        .reset_index(name="n_agents_in_run")
    )

    prevalence_df = prevalence_df.merge(
        n_agents_df,
        on=["run_dir", "round_idx"],
        how="left",
    )

    return prevalence_df


def compute_neighbor_features_from_state(state_df: pd.DataFrame) -> pd.DataFrame:
    """Compute neighbor-state fractions from the current simulated state."""
    lookup_df = state_df[["run_dir", "round_idx", "agent_id", "belief_t"]].rename(
        columns={
            "agent_id": "neighbor_agent_id",
            "belief_t": "neighbor_belief_t",
        }
    )
    lookup_df["neighbor_agent_id"] = lookup_df["neighbor_agent_id"].astype(str)

    neighbor_df = state_df[["run_dir", "round_idx", "agent_id", "neighbor_ids"]].copy()
    neighbor_df["neighbor_ids"] = neighbor_df["neighbor_ids"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else ([] if pd.isna(x) else x)
    )
    neighbor_df = neighbor_df.explode("neighbor_ids", ignore_index=False)
    neighbor_df = neighbor_df.rename(columns={"neighbor_ids": "neighbor_agent_id"})
    neighbor_df["neighbor_agent_id"] = neighbor_df["neighbor_agent_id"].astype("string")

    merged = neighbor_df.merge(
        lookup_df,
        on=["run_dir", "round_idx", "neighbor_agent_id"],
        how="left",
    )

    valid = merged.dropna(subset=["neighbor_agent_id"]).copy()

    if valid.empty:
        result = state_df[["run_dir", "round_idx", "agent_id"]].copy()
        result["neighbor_frac_neg1_t"] = 0.0
        result["neighbor_frac_0_t"] = 0.0
        result["neighbor_frac_1_t"] = 0.0
        result["n_neighbors_with_observed_belief_t"] = 0
        return result

    frac_df = (
        valid.groupby(["run_dir", "round_idx", "agent_id"])["neighbor_belief_t"]
        .value_counts(normalize=True)
        .rename("neighbor_fraction")
        .reset_index()
        .pivot(
            index=["run_dir", "round_idx", "agent_id"],
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
        valid.groupby(["run_dir", "round_idx", "agent_id"])["neighbor_belief_t"]
        .count()
        .reset_index(name="n_neighbors_with_observed_belief_t")
    )

    result = frac_df.merge(
        count_df,
        on=["run_dir", "round_idx", "agent_id"],
        how="left",
    )
    result["n_neighbors_with_observed_belief_t"] = (
        result["n_neighbors_with_observed_belief_t"].fillna(0).astype(int)
    )

    return result


def build_model_input_for_round(state_df: pd.DataFrame) -> pd.DataFrame:
    """Construct model input rows for the current simulated round."""
    model_input = state_df.copy()
    model_input = model_input.rename(columns={"round_idx": "round_t"})

    global_df = compute_global_features_from_state(state_df)
    global_df = global_df.rename(columns={"round_idx": "round_t"})
    model_input = model_input.merge(
        global_df,
        on=["run_dir", "round_t"],
        how="left",
    )

    neighbor_df = compute_neighbor_features_from_state(state_df)
    neighbor_df = neighbor_df.rename(columns={"round_idx": "round_t"})
    model_input = model_input.merge(
        neighbor_df,
        on=["run_dir", "round_t", "agent_id"],
        how="left",
    )

    return model_input


def sample_next_from_proba(
    proba_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.Series:
    """Sample a next state from class probabilities for each row."""
    sampled = []
    classes = list(proba_df.columns)

    for _, row in proba_df.iterrows():
        probs = row.to_numpy(dtype=float)
        total = probs.sum()
        if total <= 0:
            probs = np.ones(len(classes), dtype=float) / len(classes)
        else:
            probs = probs / total
        sampled.append(rng.choice(classes, p=probs))

    return pd.Series(sampled, index=proba_df.index)


def simulate_one_run(
    model: Any,
    initial_state_df: pd.DataFrame,
    n_steps: int,
    rollout_mode: str = "deterministic",
    random_state: int = 42,
) -> pd.DataFrame:
    """Simulate one run forward for n_steps starting from initial state."""
    current_state = initial_state_df.copy()
    current_state = current_state.rename(columns={"belief_label": "belief_t"})
    current_state["agent_id"] = current_state["agent_id"].astype(str)

    all_states = [current_state.copy()]
    rng = np.random.default_rng(random_state)

    for _ in range(n_steps):
        model_input = build_model_input_for_round(current_state)

        if rollout_mode == "deterministic":
            next_beliefs = model.predict_next(model_input)
        elif rollout_mode == "stochastic":
            proba_df = model.predict_proba_next(model_input)
            next_beliefs = sample_next_from_proba(proba_df, rng=rng)
        else:
            raise ValueError(
                f"Unknown rollout_mode='{rollout_mode}'. "
                "Expected one of {'deterministic', 'stochastic'}."
            )

        next_state = current_state.copy()
        next_state["belief_t"] = next_beliefs.values
        next_state["round_idx"] = next_state["round_idx"] + 1

        all_states.append(next_state.copy())
        current_state = next_state

    return pd.concat(all_states, ignore_index=True)


def simulate_many_runs(
    model: Any,
    initial_states_by_run: dict[str, pd.DataFrame],
    n_steps: int,
    rollout_mode: str = "deterministic",
    random_state: int = 42,
) -> pd.DataFrame:
    """Simulate multiple runs and concatenate outputs."""
    outputs = []

    for run_idx, (_, initial_state_df) in enumerate(initial_states_by_run.items()):
        simulated_run_df = simulate_one_run(
            model=model,
            initial_state_df=initial_state_df,
            n_steps=n_steps,
            rollout_mode=rollout_mode,
            random_state=random_state + run_idx,
        )
        outputs.append(simulated_run_df)

    if not outputs:
        return pd.DataFrame(
            columns=["run_dir", "agent_id", "round_idx", "belief_t", "degree", "neighbor_ids"]
        )

    return pd.concat(outputs, ignore_index=True)
