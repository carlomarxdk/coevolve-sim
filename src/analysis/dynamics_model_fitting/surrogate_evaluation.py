"""Surrogate evaluation for E4: in-sample model comparison plus cross-validated behaviour.

For every model x scenario x graph family cell (320 runs: 160 original + 160 on the new seeds):

Model comparison, one fit on all runs (information criteria are in-sample by construction):
    log-likelihood, AIC, BIC  one-step transitions; n = number of transitions, k = fitted parameters
                              (transition-table entries for M1, logit coefficients + intercepts else).
                              Only comparable between models within one scenario x graph family.
    McFadden pseudo-R2        1 - LL / LL_null, LL_null = the cell's marginal next-belief frequencies;
                              scale-free, so comparable across scenarios and graph families.
    The logits use scikit-learn's default L2 penalty (C = 1), negligible at ~10^5 transitions.

Behaviour, 5-fold cross-validation (every run is scored once, by a model that never saw it):
    Runs are split with `StratifiedGroupKFold` (group = run, stratified by graph type, so each fold gets
    the same mix of original and new-seed graphs); fold assignment is shared across models.
    consensus fidelity  1 - |empirical - simulated| consensus fraction, stochastic rollout from
                        round 0, averaged over rounds within a run, then over runs
    final-state MCC     one-step prediction of each agent's final belief from the true previous state
    CIs bootstrap over networks (1000 resamples, 95% percentile): a cell's runs share 16 network
    realizations (graph type x seed, 20 runs each) and runs on the same network are correlated.

Writes to `data/outputs/surrogates/evaluation/` without absolute paths:

    summary.csv     one row per model x scenario x graph family
    cv_runs.csv     one row per model x held-out run (fold, consensus error, final-state counts)
    manifest.json   design, seeds, versions

    uv run python src/analysis/dynamics_model_fitting/surrogate_evaluation.py

To redo only the CI aggregation from an existing cv_runs.csv (no refitting; fit statistics are kept
from the existing summary.csv):

    uv run python src/analysis/dynamics_model_fitting/surrogate_evaluation.py --resummarize
"""

import argparse
import json
import platform
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

from evaluate_trajectory_behavior import compare_empirical_vs_simulated
from fit_full_trajectories import MODEL_REGISTRY, merge_global_features
from simulation import simulate_many_runs
from trajectory_utils import get_initial_states_by_run
from utils import DATA_DIR, PROJECT_ROOT

MODELS = ["m1", "m2-5", "m3", "m4"]
SCENARIOS = ["base_llms", "experts", "random_experts", "random_roles"]
GRAPH_FAMILIES = ["erdos-renyi", "watts-strogatz"]
KEYS = ["model", "scenario", "graph_family"]
N_FOLDS = 5
SEED = 42  # fold assignment and bootstrap
ROLLOUT_SEED = 43  # fold k uses ROLLOUT_SEED + 10_000 * k (+ run index inside simulate_many_runs)
N_BOOTSTRAP = 1000
LABELS = [-1, 0, 1]
CM_COLS = [f"cm_{t}_{p}" for t in LABELS for p in LABELS]
EXPORT_DIR = PROJECT_ROOT / "data" / "outputs" / "surrogates" / "evaluation"


def load_cell(scenario: str, graph_family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transitions (with global features) and rounds for one scenario x graph family."""
    filters = [("experiment_group", "==", scenario), ("graph_family", "==", graph_family)]
    transitions = pd.read_parquet(DATA_DIR / "transitions.parquet", filters=filters)
    metrics = pd.read_parquet(DATA_DIR / "run_round_metrics.parquet",
                              filters=[("run_dir", "in", transitions["run_dir"].unique().tolist())])
    rounds = pd.read_parquet(DATA_DIR / "rounds.parquet", filters=filters)
    for df in (transitions, rounds):
        df["trajectory_id"] = df["run_dir"] + "__" + df["agent_id"].astype(str)
    transitions = merge_global_features(transitions, metrics)
    transitions = transitions.sort_values(["run_dir", "agent_id", "round_t"]).reset_index(drop=True)
    rounds = rounds.sort_values(["run_dir", "agent_id", "round_idx"]).reset_index(drop=True)
    return transitions, rounds


def fit(model_name: str, transitions: pd.DataFrame):
    model = MODEL_REGISTRY[model_name]()
    model.fit_full_trajectories({"train_transitions_df": transitions})
    return model


# --- model comparison (all runs) -------------------------------------------------------------

def n_parameters(model) -> int:
    """Free parameters: transition-table entries for M1, logit coefficients + intercepts otherwise."""
    table = getattr(model, "transition_probs", None)
    if table is not None:
        return table.shape[0] * (table.shape[1] - 1)
    clf = model.model.named_steps["classifier"]
    return clf.coef_.size + clf.intercept_.size


def log_likelihood(model, transitions: pd.DataFrame) -> float:
    proba = model.predict_proba_next(transitions)
    col = pd.Index(proba.columns.astype(float)).get_indexer(transitions["belief_t1"].astype(float))
    assert (col >= 0).all(), "observed next state missing from the model's classes"
    p = proba.to_numpy()[np.arange(len(proba)), col]
    return float(np.log(np.clip(p, 1e-12, None)).sum())


def null_log_likelihood(transitions: pd.DataFrame) -> float:
    """Log-likelihood of predicting every next belief with the cell's marginal frequencies."""
    counts = transitions["belief_t1"].value_counts().to_numpy(dtype=float)
    return float((counts * np.log(counts / counts.sum())).sum())


def fit_statistics(model_name: str, transitions: pd.DataFrame) -> dict:
    model = fit(model_name, transitions)
    n, k, ll = len(transitions), n_parameters(model), log_likelihood(model, transitions)
    ll_null = null_log_likelihood(transitions)
    return {"n_transitions": n, "n_params": k, "log_lik": ll, "log_lik_null": ll_null,
            "pseudo_r2": 1 - ll / ll_null, "aic": 2 * k - 2 * ll, "bic": k * np.log(n) - 2 * ll}


# --- behaviour (5-fold CV) -------------------------------------------------------------------

def assign_folds(rounds: pd.DataFrame) -> pd.Series:
    """Fold index per run_dir, identical for every model in the cell."""
    runs = rounds.drop_duplicates("run_dir").reset_index(drop=True)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = pd.Series(-1, index=runs["run_dir"])
    for k, (_, test_idx) in enumerate(cv.split(runs, runs["graph_type"], groups=runs["run_dir"])):
        fold.iloc[test_idx] = k
    return fold


def evaluate_fold(model_name: str, transitions: pd.DataFrame, rounds: pd.DataFrame,
                  test_runs: list[str], fold: int) -> pd.DataFrame:
    """Fit on the other folds; per held-out run: consensus error and final-state confusion counts."""
    model = fit(model_name, transitions[~transitions["run_dir"].isin(test_runs)])

    test_rounds = rounds[rounds["run_dir"].isin(test_runs)]
    simulated = simulate_many_runs(model, get_initial_states_by_run(test_rounds, start_round=0),
                                   n_steps=int(test_rounds["round_idx"].max()), rollout_mode="stochastic",
                                   random_state=ROLLOUT_SEED + 10_000 * fold)
    comparison = compare_empirical_vs_simulated(test_rounds, simulated)
    per_run = comparison.groupby("run_dir")["consensus_fraction_abs_error"].mean().rename("consensus_abs_error")

    test_tr = transitions[transitions["run_dir"].isin(test_runs)]
    final = test_tr.loc[test_tr.groupby("trajectory_id")["round_t1"].idxmax()]
    final = final.assign(belief_t1=final["belief_t1"].astype(int),
                         final_pred=model.predict_next(final).to_numpy().astype(int))
    counts = (final.groupby(["run_dir", "belief_t1", "final_pred"]).size()
              .unstack(["belief_t1", "final_pred"], fill_value=0))
    counts = counts.reindex(columns=pd.MultiIndex.from_product([LABELS, LABELS]), fill_value=0)
    counts.columns = CM_COLS
    return per_run.to_frame().join(counts).assign(fold=fold)


def mcc_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Multiclass MCC (as in sklearn) from confusion matrices of shape (..., k, k)."""
    t, p = cm.sum(-1), cm.sum(-2)
    c, s = np.trace(cm, axis1=-2, axis2=-1), cm.sum((-2, -1))
    num = c * s - (t * p).sum(-1)
    den = np.sqrt((s**2 - (p * p).sum(-1)) * (s**2 - (t * t).sum(-1)))
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def network_id(runs: pd.DataFrame) -> pd.Series:
    """One network realization = graph type x seed (the base configs hold 8 seeds under one graph type)."""
    return runs["graph_type"] + "__" + runs["seed"].astype(str)


def summarize_behaviour(runs: pd.DataFrame) -> dict:
    """Point estimates and network-bootstrap CIs for consensus fidelity and final-state MCC."""
    by_network = runs.assign(fidelity=1 - runs["consensus_abs_error"], n=1).groupby(network_id(runs))
    fidelity_sum, n_runs = by_network["fidelity"].sum().to_numpy(), by_network["n"].sum().to_numpy()
    cms = by_network[CM_COLS].sum().to_numpy().reshape(-1, len(LABELS), len(LABELS))
    boot = np.random.default_rng(SEED).integers(0, len(n_runs), size=(N_BOOTSTRAP, len(n_runs)))

    row = {"n_runs": len(runs), "n_runs_new": int(runs["graph_type"].str.contains(r"-\d+$").sum()),
           "n_statements": runs["statement_id"].nunique(), "n_networks": len(n_runs),
           "n_agents": int(cms.sum())}
    for name, point, dist in [
        ("consensus_fidelity", fidelity_sum.sum() / n_runs.sum(),
         fidelity_sum[boot].sum(1) / n_runs[boot].sum(1)),
        ("final_state_mcc", mcc_from_confusion(cms.sum(0)), mcc_from_confusion(cms[boot].sum(1))),
    ]:
        row |= {f"{name}__point": float(point),
                f"{name}__ci_lower": float(np.percentile(dist, 2.5)),
                f"{name}__ci_upper": float(np.percentile(dist, 97.5))}
    return row


def evaluate_cell(model_name: str, scenario: str, graph_family: str) -> tuple[dict, pd.DataFrame]:
    t0 = time.time()
    warnings.filterwarnings("ignore")
    transitions, rounds = load_cell(scenario, graph_family)
    cell = {"model": model_name, "scenario": scenario, "graph_family": graph_family}

    stats = {**cell, **fit_statistics(model_name, transitions)}
    folds = assign_folds(rounds)
    runs = pd.concat([
        evaluate_fold(model_name, transitions, rounds, folds.index[folds == k].tolist(), k)
        for k in range(N_FOLDS)
    ])
    info = rounds.drop_duplicates("run_dir").set_index("run_dir")[["run_id", "statement_id", "graph_type", "seed"]]
    runs = runs.join(info).reset_index(drop=True).assign(**cell)
    print(f"{model_name} {scenario} {graph_family}: {len(runs)} runs, k={stats['n_params']} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return stats, runs


BOOTSTRAP_NOTE = (f"{N_BOOTSTRAP} resamples over network realizations (graph type x seed; clusters of runs), "
                  "95% percentile CI")


def resummarize() -> None:
    """Recompute the behaviour columns of summary.csv from cv_runs.csv, keeping the fit statistics."""
    runs = pd.read_csv(EXPORT_DIR / "cv_runs.csv")
    old = pd.read_csv(EXPORT_DIR / "summary.csv")
    behaviour = pd.DataFrame([{**dict(zip(KEYS, k)), **summarize_behaviour(g)}
                              for k, g in runs.groupby(KEYS, sort=False)])
    fit_cols = [c for c in old.columns if c in KEYS or c not in behaviour.columns]
    summary = old[fit_cols].merge(behaviour, on=KEYS, validate="one_to_one")
    assert len(summary) == len(old) and (summary["n_runs"] == 320).all()
    summary.to_csv(EXPORT_DIR / "summary.csv", index=False)

    manifest = json.loads((EXPORT_DIR / "manifest.json").read_text())
    manifest["bootstrap"] = BOOTSTRAP_NOTE
    (EXPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Rewrote", (EXPORT_DIR / "summary.csv").relative_to(PROJECT_ROOT), "from cv_runs.csv")


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = list(product(MODELS, SCENARIOS, GRAPH_FAMILIES))
    results = Parallel(n_jobs=-1)(delayed(evaluate_cell)(*task) for task in tasks)

    id_cols = KEYS + ["fold", "run_id", "statement_id", "graph_type", "seed"]
    runs = pd.concat([r for _, r in results], ignore_index=True)
    runs = runs[id_cols + [c for c in runs.columns if c not in id_cols]]
    behaviour = pd.DataFrame([{**dict(zip(KEYS, k)), **summarize_behaviour(g)}
                              for k, g in runs.groupby(KEYS, sort=False)])
    summary = pd.DataFrame([stats for stats, _ in results]).merge(behaviour, on=KEYS)
    per_cell = summary.groupby(["scenario", "graph_family"])
    summary["delta_aic"] = summary["aic"] - per_cell["aic"].transform("min")
    summary["delta_bic"] = summary["bic"] - per_cell["bic"].transform("min")

    for df, name in [(runs, "cv_runs"), (summary, "summary")]:
        text = df.astype(str).to_csv(index=False)
        assert "/Users/" not in text and "/home/" not in text, f"{name} contains an absolute path"
    assert len(summary) == len(tasks) and (summary["n_runs"] == 320).all()

    summary.to_csv(EXPORT_DIR / "summary.csv", index=False)
    runs.to_csv(EXPORT_DIR / "cv_runs.csv", index=False)
    manifest = {
        "model_comparison": "one fit per model x scenario x graph family on all runs; AIC = 2k - 2LL, "
                            "BIC = k ln(n) - 2LL, n = one-step transitions; delta = difference to the best "
                            "model in the scenario x graph family; pseudo_r2 = McFadden 1 - LL / LL_null",
        "behaviour": f"{N_FOLDS}-fold StratifiedGroupKFold per scenario x graph family (group = run, "
                     "stratified by graph type); fold assignment shared across models; each run scored "
                     "once by the model fitted on the other folds",
        "fold_seed": SEED,
        "rollout": f"stochastic from round 0, seed {ROLLOUT_SEED} + 10000 * fold + run index",
        "bootstrap": BOOTSTRAP_NOTE,
        "consensus_fidelity": "1 - mean over runs of the round-averaged |empirical - simulated| consensus fraction",
        "final_state_mcc": "pooled over agents; one-step prediction of the final round from the true previous state",
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "rows": {"summary.csv": len(summary), "cv_runs.csv": len(runs)},
    }
    (EXPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Wrote", EXPORT_DIR.relative_to(PROJECT_ROOT), manifest["rows"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resummarize", action="store_true",
                        help="only recompute CIs from the existing cv_runs.csv (no refitting)")
    if parser.parse_args().resummarize:
        resummarize()
    else:
        main()
