from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import entropy


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "src/analysis/dynamics_model_fitting/data"
OUTPUT_DIR = PROJECT_ROOT / "src/analysis/dynamics_model_fitting/outputs"


def _resolve_config_path(run_dir: str) -> Path | None:
    """Resolve a run's config path, including cross-machine path remapping.

    Args:
        run_dir: Run directory string stored in aggregated tables.

    Returns:
        Path to config.json if resolvable, otherwise None.
    """
    direct_config = Path(run_dir) / "config.json"
    if direct_config.exists():
        return direct_config

    anchor = "/data/outputs/runs/"
    if anchor in run_dir:
        suffix = run_dir.split(anchor, 1)[1]
        remapped_config = (
            PROJECT_ROOT / "data" / "outputs" / "runs" / suffix / "config.json"
        )
        if remapped_config.exists():
            return remapped_config

    return None


def bootstrap_metric(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.DataFrame | None = None,
    metric_func: Callable = accuracy_score,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05,
    n_jobs: int = -1,
    **metric_kwargs: Any,
) -> tuple[float, float, float]:
    """Estimate bootstrap confidence intervals.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_proba: Class probability matrix; required when ``metric_func`` is
            ``log_loss``.
        metric_func: Scikit-learn-compatible scoring function.
        n_bootstrap: Number of bootstrap resamples.
        random_state: Seed for the random-number generator.
        alpha: Significance level; produces a ``(1 - alpha)`` CI.
        n_jobs: Number of parallel workers passed to joblib (``1`` disables
            parallelism).
        **metric_kwargs: Extra keyword arguments forwarded to ``metric_func``.

    Returns:
        A three-tuple ``(point, ci_lower, ci_upper)`` where *point* is the
        metric on the full sample and the remaining two values are the
        lower and upper percentile bounds of the bootstrap distribution.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_proba_arr = np.asarray(y_proba) if y_proba is not None else None

    uses_proba = metric_func == log_loss

    # Pre-generate all indices at once: shape (n_bootstrap, n)
    all_idx = rng.integers(0, n, size=(n_bootstrap, n))

    def _one_sample(idx: np.ndarray) -> float:
        if uses_proba and y_proba_arr is not None:
            return metric_func(y_true_arr[idx], y_proba_arr[idx], **metric_kwargs)
        return metric_func(y_true_arr[idx], y_pred_arr[idx], **metric_kwargs)

    if n_jobs != 1:
        from joblib import Parallel, delayed

        metrics = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one_sample)(idx) for idx in all_idx
        )
    else:
        metrics = [_one_sample(idx) for idx in all_idx]

    metrics = np.array(metrics)

    point = (
        metric_func(y_true_arr, y_proba_arr, **metric_kwargs)
        if uses_proba and y_proba_arr is not None
        else metric_func(y_true_arr, y_pred_arr, **metric_kwargs)
    )

    lower = 100 * (alpha / 2)
    upper = 100 * (1 - alpha / 2)
    return (
        point,
        float(np.percentile(metrics, lower)),
        float(np.percentile(metrics, upper)),
    )

def bootstrap_all_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.DataFrame | None = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05,
) -> dict[str, dict[str, float] | None]:
    """Compute accuracy, macro-F1, MCC, and log-loss with shared bootstrap indices."""
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    yproba = np.asarray(y_proba) if y_proba is not None else None

    # One index matrix, reused for all metrics
    all_idx = rng.integers(0, n, size=(n_bootstrap, n))  # (B, n)

    # Vectorized accuracy — no loop at all
    boot_true = yt[all_idx]   # (B, n)
    boot_pred = yp[all_idx]   # (B, n)
    acc_boot = (boot_true == boot_pred).mean(axis=1)  # (B,)

    # F1 and MCC still need a loop (no clean vectorization)
    f1_boot, mcc_boot = [], []
    for idx in all_idx:
        try:
            f1_boot.append(f1_score(yt[idx], yp[idx], average="macro"))
        except Exception:
            f1_boot.append(np.nan)
        try:
            mcc_boot.append(matthews_corrcoef(yt[idx], yp[idx]))
        except Exception:
            mcc_boot.append(np.nan)

    f1_boot = np.array(f1_boot)
    mcc_boot = np.array(mcc_boot)

    lower_p = 100 * alpha / 2
    upper_p = 100 * (1 - alpha / 2)

    def _ci(arr: np.ndarray, point: float) -> dict[str, float]:
        valid = arr[~np.isnan(arr)]
        return {
            "point": point,
            "ci_lower": float(np.percentile(valid, lower_p)),
            "ci_upper": float(np.percentile(valid, upper_p)),
        }

    results: dict[str, dict[str, float] | None] = {
        "accuracy": _ci(acc_boot, float(accuracy_score(yt, yp))),
        "macro_f1": _ci(f1_boot, float(f1_score(yt, yp, average="macro"))),
    }

    try:
        results["mcc"] = _ci(mcc_boot, float(matthews_corrcoef(yt, yp)))
    except Exception:
        results["mcc"] = None

    results["log_loss"] = None
    if yproba is not None and y_proba is not None:
        class_order = sorted(pd.Series(yt).dropna().unique())
        common_cols = [c for c in class_order if c in y_proba.columns]
        if common_cols:
            col_idx = [list(y_proba.columns).index(c) for c in common_cols]
            ll_boot = []
            for idx in all_idx:
                try:
                    ll_boot.append(
                        log_loss(yt[idx], yproba[np.ix_(idx, col_idx)], labels=common_cols)
                    )
                except Exception:
                    ll_boot.append(np.nan)
            ll_boot = np.array(ll_boot)
            try:
                results["log_loss"] = _ci(
                    ll_boot, float(log_loss(yt, yproba[:, col_idx], labels=common_cols))
                )
            except Exception:
                pass

    return results


def load_transitions() -> pd.DataFrame:
    """Load transition-level rows and append derived identifiers/features.

    Returns:
        Transition dataframe with `trajectory_id` and per-run `seed` columns.
    """
    df = pd.read_parquet(DATA_DIR / "transitions.parquet")
    df["trajectory_id"] = df["run_dir"] + "__" + df["agent_id"].astype(str)
    return df


def load_run_round_metrics() -> pd.DataFrame:
    """Load per-run, per-round aggregate metrics from disk.

    Returns:
        DataFrame indexed by run and round with precomputed metric columns.
    """
    return pd.read_parquet(DATA_DIR / "run_round_metrics.parquet")


def train_val_test_split_by_group(
    df: pd.DataFrame,
    group_col: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/val/test sets stratified by a group column.

    Rows that share the same group value are never split across partitions,
    preventing data leakage between splits.

    Args:
        df: Input dataframe.
        group_col: Column name whose values define the grouping units.
        train_size: Fraction of groups allocated to the training set.
        val_size: Fraction of groups allocated to the validation set.
        test_size: Fraction of groups allocated to the test set.
        random_state: Seed for reproducibility.

    Returns:
        A three-tuple ``(train_df, val_df, test_df)``.

    Raises:
        ValueError: If the three size fractions do not sum to 1.
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-8:
        raise ValueError("train/val/test sizes must sum to 1.")

    df = df.copy()
    groups = df[group_col]

    gss1 = GroupShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=random_state,
    )
    train_idx, temp_idx = next(gss1.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    relative_val_size = val_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(
        n_splits=1,
        train_size=relative_val_size,
        random_state=random_state,
    )
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[group_col]))

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    return train_df, val_df, test_df


def train_val_test_split_by_trajectory(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by ``trajectory_id`` so all steps of a trajectory stay together.

    Args:
        df: Input dataframe containing a ``trajectory_id`` column.
        train_size: Fraction of trajectories for training.
        val_size: Fraction of trajectories for validation.
        test_size: Fraction of trajectories for testing.
        random_state: Seed for reproducibility.

    Returns:
        A three-tuple ``(train_df, val_df, test_df)``.
    """
    return train_val_test_split_by_group(
        df=df,
        group_col="trajectory_id",
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )


def train_val_test_split_by_run(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by ``run_dir`` so that all agents from a run stay together.

    Args:
        df: Input dataframe containing a ``run_dir`` column.
        train_size: Fraction of runs for training.
        val_size: Fraction of runs for validation.
        test_size: Fraction of runs for testing.
        random_state: Seed for reproducibility.

    Returns:
        A three-tuple ``(train_df, val_df, test_df)``.
    """
    return train_val_test_split_by_group(
        df=df,
        group_col="run_dir",
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.DataFrame | None = None,
) -> dict[str, dict[str, float] | float]:
    """Compute pooled one-step prediction metrics with bootstrap CIs.

    Evaluates accuracy, macro-F1, MCC, and optionally log-loss over the
    entire prediction set (no per-trajectory splitting).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_proba: Class probability matrix used for log-loss; skipped if
            ``None``.

    Returns:
        Mapping from metric name to a dict with keys ``point``, ``ci_lower``,
        and ``ci_upper``, or ``None`` when the metric could not be computed.
    """
    results: dict[str, dict[str, float] | float] = {}

    acc_point, acc_lower, acc_upper = bootstrap_metric(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=accuracy_score,
        n_bootstrap=1000,
    )
    results["accuracy"] = {
        "point": acc_point,
        "ci_lower": acc_lower,
        "ci_upper": acc_upper,
    }

    f1_point, f1_lower, f1_upper = bootstrap_metric(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=f1_score,
        average="macro",
        n_bootstrap=1000,
    )
    results["macro_f1"] = {
        "point": f1_point,
        "ci_lower": f1_lower,
        "ci_upper": f1_upper,
    }

    try:
        bal_acc_point, bal_acc_lower, bal_acc_upper = bootstrap_metric(
            y_true=y_true,
            y_pred=y_pred,
            metric_func=balanced_accuracy_score,
            n_bootstrap=1000,
        )
        results["balanced_accuracy"] = {
            "point": bal_acc_point,
            "ci_lower": bal_acc_lower,
            "ci_upper": bal_acc_upper,
        }
    except Exception:
        results["balanced_accuracy"] = None # type: ignore

    try:
        kappa_point, kappa_lower, kappa_upper = bootstrap_metric(
            y_true=y_true,
            y_pred=y_pred,
            metric_func=cohen_kappa_score,
            n_bootstrap=1000,
        )
        results["cohen_kappa"] = {
            "point": kappa_point,
            "ci_lower": kappa_lower,
            "ci_upper": kappa_upper,
        }
    except Exception:
        results["cohen_kappa"] = None # type: ignore

    # MCC with bootstrap CI
    try:
        mcc_point, mcc_lower, mcc_upper = bootstrap_metric(
            y_true=y_true,
            y_pred=y_pred,
            metric_func=matthews_corrcoef,
            n_bootstrap=1000,
        )
        results["mcc"] = {
            "point": mcc_point,
            "ci_lower": mcc_lower,
            "ci_upper": mcc_upper,
        }
    except Exception:
        results["mcc"] = None # type: ignore

    # Log loss with bootstrap CI
    if y_proba is not None:
        class_order = sorted(y_true.dropna().unique())
        common_cols = [c for c in class_order if c in y_proba.columns]
        if common_cols:
            try:
                ll_point, ll_lower, ll_upper = bootstrap_metric(
                    y_true=y_true,
                    y_pred=y_pred,
                    y_proba=y_proba[common_cols],
                    metric_func=log_loss,
                    labels=common_cols,
                    n_bootstrap=1000,
                )
                results["log_loss"] = {
                    "point": ll_point,
                    "ci_lower": ll_lower,
                    "ci_upper": ll_upper,
                }
            except Exception:
                results["log_loss"] = None # type: ignore

    return results


def _safe_metric_dict_from_arrays(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.DataFrame | None = None,
) -> dict[str, float | None]:
    """Compute point estimates only. Used per-trajectory."""
    if len(y_true) == 0:
        return {"accuracy": None, "macro_f1": None, "mcc": None, "log_loss": None}

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    metrics: dict[str, float | None] = {}
    metrics["accuracy"] = float(accuracy_score(yt, yp))

    try:
        metrics["macro_f1"] = float(f1_score(yt, yp, average="macro"))
    except Exception:
        metrics["macro_f1"] = None

    try:
        metrics["mcc"] = float(matthews_corrcoef(yt, yp))
    except Exception:
        metrics["mcc"] = None

    metrics["log_loss"] = None
    if y_proba is not None and not y_proba.empty:
        class_order = sorted(y_true.dropna().unique())
        common_cols = [c for c in class_order if c in y_proba.columns]
        if common_cols:
            try:
                metrics["log_loss"] = float(
                    log_loss(yt, np.asarray(y_proba[common_cols]), labels=common_cols)
                )
            except Exception:
                pass

    return metrics


def evaluate_trajectory_fit_predictions(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.DataFrame | None = None,
    trajectory_col: str = "trajectory_id",
    n_bootstrap: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05,
    round_col: str = "round_t1",
) -> dict[str, float | None]:
    """Compute trajectory-aware ML fit metrics.

    Metrics are computed both averaged across all per-trajectory step
    predictions and on final-state predictions only.

    Args:
        df: DataFrame containing at least ``trajectory_col`` and ``round_col``
            columns, aligned with ``y_true`` / ``y_pred``.
        y_true: Ground-truth labels aligned with ``df``.
        y_pred: Predicted labels aligned with ``df``.
        y_proba: Class probability matrix aligned with ``df``; used only for
            log-loss.  Skipped if ``None``.
        trajectory_col: Column identifying individual trajectories.
        round_col: Column containing the round/time-step index used to sort
            steps within a trajectory.

    Returns:
        Dictionary with keys ``mean_trajectory_step_accuracy``,
        ``mean_trajectory_step_macro_f1``, ``mean_trajectory_step_mcc``,
        ``mean_trajectory_step_log_loss``, ``final_state_accuracy``,
        ``final_state_macro_f1``, ``final_state_mcc``, and
        ``exact_trajectory_match_rate``.  Values are ``None`` when the metric
        cannot be computed.
    """
    work_df = df[[trajectory_col, round_col]].copy().reset_index(drop=True)
    work_df["y_true"] = y_true.values
    work_df["y_pred"] = y_pred.values

    proba_cols: list[str] = []
    if y_proba is not None:
        proba_df = y_proba.copy().reset_index(drop=True)
        proba_df.columns = [f"proba__{c}" for c in proba_df.columns]
        proba_cols = list(proba_df.columns)
        work_df = pd.concat([work_df, proba_df], axis=1)

    def _traj_stats(traj_df: pd.DataFrame) -> dict[str, Any]:
        traj_df = traj_df.sort_values(round_col)
        yt = traj_df["y_true"].values
        yp = traj_df["y_pred"].values

        traj_proba = None
        if proba_cols:
            traj_proba_df = traj_df[proba_cols].copy()
            traj_proba_df.columns = [c.replace("proba__", "") for c in proba_cols]
            traj_proba = traj_proba_df

        m = _safe_metric_dict_from_arrays(
            y_true=pd.Series(yt),
            y_pred=pd.Series(yp),
            y_proba=traj_proba,
        )
        return {
            "step_accuracy":  m["accuracy"],
            "step_macro_f1":  m["macro_f1"],
            "step_mcc":       m["mcc"],
            "step_log_loss":  m["log_loss"],
            "exact_match":    float((yt == yp).all()),
            "flip_rate_true": float((np.diff(yt) != 0).mean()) if len(yt) > 1 else np.nan,
            "flip_rate_pred": float((np.diff(yp) != 0).mean()) if len(yp) > 1 else np.nan,
            "lcs_rate":       _lcs_rate(yt, yp),
            "final_true":     yt[-1],
            "final_pred":     yp[-1],
        }

    grouped = work_df.groupby(trajectory_col, sort=False)
    traj_ids = list(grouped.groups.keys())
    traj_stats = {tid: _traj_stats(tdf) for tid, tdf in grouped}
    per_traj_df = pd.DataFrame(traj_stats.values(), index=traj_ids)

    rng = np.random.default_rng(random_state)
    n_traj = len(traj_ids)
    boot_idx = rng.integers(0, n_traj, size=(n_bootstrap, n_traj))

    # Scalar-per-trajectory columns we want to aggregate with mean
    mean_cols = [
        "step_accuracy", "step_macro_f1", "step_mcc", "step_log_loss",
        "exact_match", "lcs_rate",
    ]
    # For flip rate we want |true_rate - pred_rate| per bootstrap
    stats_arr = per_traj_df[mean_cols].to_numpy(dtype=float)       # (n_traj, k)
    flip_true_arr = per_traj_df["flip_rate_true"].to_numpy(float)  # (n_traj,)
    flip_pred_arr = per_traj_df["flip_rate_pred"].to_numpy(float)  # (n_traj,)
    final_true_arr = per_traj_df["final_true"].to_numpy()          # (n_traj,)
    final_pred_arr = per_traj_df["final_pred"].to_numpy()          # (n_traj,)

    # Boot distributions: (n_bootstrap, k) for mean cols
    boot_means = np.nanmean(stats_arr[boot_idx], axis=1)            # (n_bootstrap, k)

    boot_flip_error = np.abs(
        np.nanmean(flip_true_arr[boot_idx], axis=1)
        - np.nanmean(flip_pred_arr[boot_idx], axis=1)
    )  # (n_bootstrap,)

    # Final-state accuracy per bootstrap resample
    boot_final_acc = (
        final_true_arr[boot_idx] == final_pred_arr[boot_idx]
    ).mean(axis=1)  # (n_bootstrap,)

    # Final-state KL per bootstrap resample
    labels = sorted(np.unique(np.concatenate([final_true_arr, final_pred_arr])))
    boot_final_kl = np.array([
        _final_state_kl(final_true_arr[idx], final_pred_arr[idx], labels)
        for idx in boot_idx
    ])  # (n_bootstrap,) — loop unavoidable; KL needs frequency counts


    lower_p = 100 * alpha / 2
    upper_p = 100 * (1 - alpha / 2)

    def _ci(point: float, boot_dist: np.ndarray) -> dict[str, float]:
        valid = boot_dist[~np.isnan(boot_dist)]
        return {
            "point":    float(point),
            "ci_lower": float(np.percentile(valid, lower_p)),
            "ci_upper": float(np.percentile(valid, upper_p)),
        }

    point_means = np.nanmean(stats_arr, axis=0)  # (k,)
    point_flip_error = abs(
        float(np.nanmean(flip_true_arr)) - float(np.nanmean(flip_pred_arr))
    )
    point_final_acc = float((final_true_arr == final_pred_arr).mean())
    point_final_kl = _final_state_kl(final_true_arr, final_pred_arr, labels)

    col_idx = {c: i for i, c in enumerate(mean_cols)}

    def _mean_ci(col: str) -> dict[str, float] | None:
        i = col_idx[col]
        pt = point_means[i]
        if np.isnan(pt):
            return None
        return _ci(pt, boot_means[:, i])



    # Final-state CI via bootstrap_all_metrics (row-level bootstrap is fine
    # for a single-point-per-trajectory series)
    final_state_full = bootstrap_all_metrics(
        y_true=pd.Series(final_true_arr),
        y_pred=pd.Series(final_pred_arr),
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        alpha=alpha,
    )

    return {
        # Step-level means across trajectories
        "mean_trajectory_step_accuracy":  _mean_ci("step_accuracy"),
        "mean_trajectory_step_macro_f1":  _mean_ci("step_macro_f1"),
        "mean_trajectory_step_mcc":       _mean_ci("step_mcc"),
        "mean_trajectory_step_log_loss":  _mean_ci("step_log_loss"),
        "exact_trajectory_match_rate":    _mean_ci("exact_match"),
        "mean_lcs_rate":                  _mean_ci("lcs_rate"),
        # Dynamics-specific
        "flip_rate_error":  _ci(point_flip_error, boot_flip_error),
        "final_state_kl":   _ci(point_final_kl, boot_final_kl),
        # Final-state classification metrics (row-bootstrap over finals)
        "final_state_accuracy":  final_state_full.get("accuracy"),
        "final_state_macro_f1":  final_state_full.get("macro_f1"),
        "final_state_mcc":       final_state_full.get("mcc"),
        "final_state_balanced_accuracy": _ci(
            point_final_acc,
            boot_final_acc,
        ),
    } # type: ignore

def _lcs_rate(seq_true: np.ndarray, seq_pred: np.ndarray) -> float:
    """Edit Distance. Compute the longest common subsequence (LCS) rate between two sequences."""

    n, m = len(seq_true), len(seq_pred)
    if n == 0:
        return np.nan
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = (
                dp[i - 1, j - 1] + 1
                if seq_true[i - 1] == seq_pred[j - 1]
                else max(dp[i - 1, j], dp[i, j - 1])
            )
    return float(dp[n, m] / n)


def _final_state_kl(
    true_finals: np.ndarray,
    pred_finals: np.ndarray,
    labels: list,
    epsilon: float = 1e-8,
) -> float:
    """Compute KL divergence between the empirical distributions of true and predicted final states."""
    n = len(true_finals)
    p = np.array([(true_finals == l).sum() for l in labels], dtype=float) + epsilon
    q = np.array([(pred_finals == l).sum() for l in labels], dtype=float) + epsilon
    p /= p.sum()
    q /= q.sum()
    return float(entropy(p, q))

def _safe_mean(series: pd.Series) -> float | None:
    """Return the mean of non-null values, or None if none exist."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    return float(non_null.mean())
