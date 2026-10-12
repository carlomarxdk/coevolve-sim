"""Re-evaluate the selected full-trajectory surrogate fits and export anonymised results.

For every model x scenario x graph family cell, selects the latest `trans-alltrans` fit trained on
all runs, rolls it out on its test split (deterministic and stochastic, with the seeds of the
reported evaluation) and writes to `data/outputs/surrogates/`:

    surrogate_summary.csv     one row per cell x rollout mode: consensus/behaviour errors with
                              run-bootstrap CIs, final-state MCC from the fit, test-set composition
    surrogate_run_rounds.parquet  per test run x round empirical vs simulated behaviour
    manifest.json             selection rule, package versions, row counts

No absolute paths or user names are written; runs are identified by `run_id`.

The saved models were pickled with scikit-learn 1.6.1. Newer versions unpickle them but return
different class probabilities, which changes every stochastic rollout, so the script refuses to
run under any other version:

    uv run --no-project --python 3.12 --with scikit-learn==1.6.1 --with pandas --with pyarrow \
        --with joblib python src/analysis/dynamics_model_fitting/export_surrogate_evaluation.py
"""

import json
import platform
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from evaluate_trajectory_behavior import (
    compare_empirical_vs_simulated,
    get_nontrivial_run_ids,
    summarize_comparison_metrics,
)
from simulation import simulate_many_runs
from trajectory_utils import (
    get_eval_rounds_for_saved_model,
    get_initial_states_by_run,
    load_saved_model_bundle,
    load_split_manifest,
)
from utils import OUTPUT_DIR, PROJECT_ROOT

REQUIRED_SKLEARN = "1.6.1"
MODELS = ["m1", "m2-5", "m3", "m4"]
FIT_DIR = OUTPUT_DIR / "full_trajectories"
EXPORT_DIR = PROJECT_ROOT / "data" / "outputs" / "surrogates"
# Seeds of the reported evaluation: random_state + model_idx * 100 + rollout_idx, model_idx = 0.
ROLLOUTS = {"deterministic": 42, "stochastic": 43}
FIT_RE = re.compile(r"__exp-(?P<scenario>.+?)__graph-(?P<graph_family>[a-z-]+?)__trans-alltrans__(?P<ts>\d{8}_\d{6})__results\.json$")
NEW_SEED_RE = re.compile(r"-\d+$")


def select_fits() -> pd.DataFrame:
    """Latest alltrans fit per cell among those trained on original and new-seed runs together."""
    rows = []
    for model in MODELS:
        for path in sorted((FIT_DIR / model).glob("*__trans-alltrans__*__results.json")):
            m = FIT_RE.search(path.name)
            results = json.loads(path.read_text())
            graph_types = results.get("graph_types_in_fit") or []
            is_new = [bool(NEW_SEED_RE.search(g)) for g in graph_types]
            if not (any(is_new) and not all(is_new)):
                continue
            mcc = results["metrics"]["test"]["trajectory_fit_metrics"]["final_state_mcc"]
            rows.append({
                "model": model, "scenario": m["scenario"], "graph_family": m["graph_family"],
                "fit_timestamp": m["ts"], "n_runs_in_fit": results["n_runs_in_fit"],
                "final_state_mcc__point": mcc["point"],
                "final_state_mcc__ci_lower": mcc["ci_lower"],
                "final_state_mcc__ci_upper": mcc["ci_upper"],
                "stem": path.name.removesuffix("__results.json"),
            })
    fits = pd.DataFrame(rows).sort_values("fit_timestamp")
    return fits.drop_duplicates(["model", "scenario", "graph_family"], keep="last").reset_index(drop=True)


def evaluate_fit(fit: pd.Series) -> tuple[list[dict], pd.DataFrame]:
    bundle = load_saved_model_bundle(FIT_DIR / fit.model / "saved_models" / f"{fit.stem}.joblib")
    split = load_split_manifest(Path(bundle["split_path"]))
    test_runs = split[split["split"] == "test"].drop_duplicates("run_dir")
    run_info = test_runs.set_index("run_dir")[["run_id", "graph_type", "seed"]]

    empirical = get_eval_rounds_for_saved_model(bundle=bundle, split_name="test")
    initial_states = get_initial_states_by_run(empirical, start_round=0)
    n_steps = int(empirical["round_idx"].max())
    nontrivial = set(get_nontrivial_run_ids(empirical))

    cell = fit[["model", "scenario", "graph_family", "fit_timestamp"]].to_dict()
    summary_rows, run_rounds = [], []
    for mode, seed in ROLLOUTS.items():
        simulated = simulate_many_runs(bundle["model"], initial_states, n_steps=n_steps,
                                       rollout_mode=mode, random_state=seed)
        comparison = compare_empirical_vs_simulated(empirical, simulated)
        row = {**cell, "rollout_mode": mode, "random_state": seed,
               "n_test_runs": comparison["run_dir"].nunique(), "n_nontrivial_runs": len(nontrivial)}
        for prefix, df in [("all_runs", comparison),
                           ("nontrivial_runs", comparison[comparison["run_dir"].isin(nontrivial)])]:
            for metric, stats in summarize_comparison_metrics(df, n_bootstrap=1000, random_state=seed).items():
                for key in ("point", "ci_lower", "ci_upper"):
                    row[f"{prefix}__{metric}__{key}"] = None if stats is None else stats[key]
        summary_rows.append(row)

        comparison = comparison.join(run_info, on="run_dir")
        comparison["nontrivial_run"] = comparison["run_dir"].isin(nontrivial)
        run_rounds.append(comparison.drop(columns="run_dir").assign(**cell, rollout_mode=mode))

    is_new = test_runs["graph_type"].str.contains(NEW_SEED_RE)
    for row in summary_rows:
        row.update(n_test_old=int((~is_new).sum()), n_test_new=int(is_new.sum()),
                   n_runs_in_fit=int(fit.n_runs_in_fit),
                   **{c: fit[c] for c in fit.index if c.startswith("final_state_mcc")})
    return summary_rows, pd.concat(run_rounds, ignore_index=True)


def assert_anonymous(df: pd.DataFrame, name: str) -> None:
    text = df.astype(str).to_csv(index=False)
    assert "/Users/" not in text and "/home/" not in text, f"{name} contains an absolute path"


def main() -> None:
    if sklearn.__version__ != REQUIRED_SKLEARN:
        raise RuntimeError(f"scikit-learn {REQUIRED_SKLEARN} required (found {sklearn.__version__}); see module docstring.")
    warnings.filterwarnings("ignore")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    fits = select_fits()
    assert len(fits) == 32, f"expected 32 cells, selected {len(fits)}"

    summary_rows, run_rounds = [], []
    for i, fit in fits.iterrows():
        rows, rr = evaluate_fit(fit)
        summary_rows += rows
        run_rounds.append(rr)
        print(f"[{i + 1}/{len(fits)}] {fit.model} {fit.scenario} {fit.graph_family} {fit.fit_timestamp}", flush=True)

    id_cols = ["model", "scenario", "graph_family", "fit_timestamp", "rollout_mode"]
    summary = pd.DataFrame(summary_rows)
    summary = summary[id_cols + [c for c in summary.columns if c not in id_cols]]
    run_rounds = pd.concat(run_rounds, ignore_index=True)
    run_rounds = run_rounds[id_cols + [c for c in run_rounds.columns if c not in id_cols]]
    for df, name in [(summary, "summary"), (run_rounds, "run_rounds")]:
        assert_anonymous(df, name)

    summary.to_csv(EXPORT_DIR / "surrogate_summary.csv", index=False)
    run_rounds.to_parquet(EXPORT_DIR / "surrogate_run_rounds.parquet", index=False)
    manifest = {
        "selection": "per model x scenario x graph family: latest trans-alltrans fit whose training "
                     "runs include original and new-seed graphs; evaluated on its test split",
        "fits": fits[["model", "scenario", "graph_family", "fit_timestamp"]].to_dict("records"),
        "rollouts": ROLLOUTS,
        "bootstrap": "1000 resamples over test runs, 95% percentile CI",
        "final_state_mcc": "copied from the fit's results.json (bootstrap over test agents)",
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "rows": {"surrogate_summary.csv": len(summary), "surrogate_run_rounds.parquet": len(run_rounds)},
    }
    (EXPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Wrote", EXPORT_DIR.relative_to(PROJECT_ROOT), manifest["rows"])


if __name__ == "__main__":
    main()
