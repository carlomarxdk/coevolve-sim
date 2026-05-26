from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from utils import DATA_DIR, PROJECT_ROOT


def _install_sklearn_pickle_compat() -> None:
    """Install lightweight aliases for loading older scikit-learn pickles."""
    try:
        import sklearn.compose._column_transformer as ct
        from sklearn.impute import SimpleImputer
    except Exception:
        return

    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Compatibility alias for models serialized with older sklearn."""

        ct._RemainderColsList = _RemainderColsList

    # Some older pickles do not include this newer attribute. Reconstruct it
    # from the fitted dtype when possible so transform() can proceed.
    if not hasattr(SimpleImputer, "_compat_transform_patched"):
        original_transform = SimpleImputer.transform

        def _compat_transform(self: Any, X: Any) -> Any:
            if not hasattr(self, "_fill_dtype") and hasattr(self, "_fit_dtype"):
                self._fill_dtype = self._fit_dtype
            return original_transform(self, X)

        SimpleImputer.transform = _compat_transform
        SimpleImputer._compat_transform_patched = True


def load_saved_model_bundle(model_path: Path) -> dict:
    """Load a saved model bundle."""
    _install_sklearn_pickle_compat()
    return joblib.load(model_path)


def _resolve_split_manifest_path(split_path: Path) -> Path | None:
    """Resolve split manifest path, including cross-machine path remapping."""
    if split_path.exists():
        return split_path

    # Saved bundles may contain absolute paths from a different machine.
    parts = split_path.parts
    if "outputs" in parts:
        outputs_idx = parts.index("outputs")
        suffix = Path(*parts[outputs_idx + 1 :])
        remapped = (
            PROJECT_ROOT
            / "src"
            / "analysis"
            / "dynamics_model_fitting"
            / "outputs"
            / suffix
        )
        if remapped.exists():
            return remapped

    return None


def load_split_manifest(split_path: Path) -> pd.DataFrame:
    """Load saved trajectory split manifest."""
    resolved = _resolve_split_manifest_path(split_path)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not resolve split manifest path: {split_path}"
        )
    return pd.read_csv(resolved)


def load_rounds() -> pd.DataFrame:
    """Load empirical round-level data."""
    df = pd.read_parquet(DATA_DIR / "rounds.parquet")
    df["trajectory_id"] = df["run_dir"] + "__" + df["agent_id"].astype(str)
    return df


def get_eval_rounds_for_saved_model(
    bundle: dict,
    split_name: str = "test",
) -> pd.DataFrame:
    """Load empirical rounds for the requested split from a saved model bundle."""
    split_df = load_split_manifest(Path(bundle["split_path"]))
    split_df = split_df[split_df["split"] == split_name].copy()

    rounds_df = load_rounds()
    eval_df = rounds_df.merge(
        split_df[["trajectory_id"]],
        on="trajectory_id",
        how="inner",
    )
    return eval_df


def get_initial_states_by_run(
    eval_rounds_df: pd.DataFrame,
    start_round: int = 0,
) -> dict[str, pd.DataFrame]:
    """Build a dictionary of initial states, one dataframe per run.

    Preserves agent identity columns needed by rollout models such as M4,
    which conditions on `agent_model` and `agent_role` at prediction time.
    """
    init_df = eval_rounds_df[eval_rounds_df["round_idx"] == start_round].copy()

    required_cols = [
        "run_dir",
        "agent_id",
        "round_idx",
        "seed",
        "belief_label",
        "degree",
        "neighbor_ids",
    ]

    optional_identity_cols = [
        col for col in ["agent_model", "agent_role"] if col in init_df.columns
    ]
    init_df = init_df[required_cols + optional_identity_cols].copy()

    return {
        run_dir: run_df.copy()
        for run_dir, run_df in init_df.groupby("run_dir")
    }
