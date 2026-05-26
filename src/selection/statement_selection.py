"""Medical statement selection using prediction diversity criteria."""

from pathlib import Path

import numpy as np
import polars as pl
from diversity import maximin_select_balanced
from sklearn.preprocessing import StandardScaler


def load_statement_data(
    data_dir: Path,
    doctor_model: str = "llama-doc",
    other_models: list[str] | None = None,
    filter_negation: bool = True,
    filter_real_object: bool = True,
) -> tuple[pl.DataFrame, list[int]]:
    """Load and preprocess medical statement predictions from multiple LLMs.

    Loads predictions from a doctor LLM and aggregates predictions from other
    models. Validates data consistency across all models.

    Args:
        data_dir: Directory containing prediction CSV files.
        doctor_model: Name of the doctor/expert model.
        other_models: List of other model names. If None, uses default set.
        filter_negation: If True, exclude negated statements.
        filter_real_object: If True, exclude statements without real objects.

    Returns:
        Tuple of (df_preds, valid_idx) where:
            - df_preds: DataFrame with combined predictions and computed metrics.
            - valid_idx: List of valid statement indices after filtering.

    Raises:
        AssertionError: If index/statement mismatches found across models.

    Example:
        >>> df, idx = load_statement_data(Path("resources/data/predictions/"))
        >>> print(df.columns)
    """
    if other_models is None:
        other_models = [
            "llama-assistant",
            "llama-base",
            "llama-biomed",
            "llama-chemist",
            "llama-coder",
            "llama-cyber",
            "llama-finance",
            "llama-hermes",
            "llama-law",
            "llama-lexicographer",
            "llama-linguist",
            "llama-openmath",
            "llama-roleplay",
            "llama-scholar",
            "llama-user",
        ]

    # Load and filter base data from doctor model
    df_doc = (
        pl.read_csv(data_dir / f"zs_med_test_split_{doctor_model}.csv")
        .filter(
            pl.col("negation") == (0 if filter_negation else pl.col("negation")),
            pl.col("real_object")
            == (1 if filter_real_object else pl.col("real_object")),
        )
        .select(
            pl.col("").alias("init_idx"),
            "statement",
            pl.col("correct").alias("label_ground_truth"),
            pl.col("predicted_label").alias("doc_predicted_label"),
            pl.col("prob_true").alias("doc_prob"),
        )
        .with_row_index("idx")
    )

    valid_idx = df_doc["init_idx"].to_list()

    # Load predictions from other models
    dfs_models = [
        pl.read_csv(data_dir / f"zs_med_test_split_{model}.csv")
        .filter(pl.col("").is_in(valid_idx))
        .rename({"": "init_idx"})
        .select("init_idx", "predicted_label", "prob_true", "statement")
        .with_columns(
            (pl.col("predicted_label") == 1).cast(pl.Float32).alias("predicted_label")
        )
        for model in other_models
    ]

    # Validate consistency
    for i, df_model in enumerate(dfs_models):
        model_indices = df_model["init_idx"].to_list()
        model_statements = df_model["statement"].to_list()
        assert model_indices == valid_idx, (
            f"Index mismatch in {other_models[i]}: "
            f"model indices do not match doctor model order"
        )
        assert len(model_statements) == len(set(model_statements)), (
            f"Duplicate statements found in {other_models[i]}"
        )

    # Aggregate predictions
    df_agg = (
        pl.concat(dfs_models)
        .group_by("init_idx")
        .agg(
            pl.col("predicted_label").mean().alias("other_frac_predicted_label"),
            pl.col("prob_true").median().alias("other_median_prob"),
        )
        .with_row_index("idx")
    )

    # Join and compute additional metrics
    df_preds = df_doc.join(df_agg, on="idx", how="inner").with_columns(
        (1 - (pl.col("doc_prob") - pl.col("other_median_prob")).abs())
        .cast(pl.Float64)
        .alias("consensus_score"),
        (
            1
            - (
                pl.col("label_ground_truth") - pl.col("other_frac_predicted_label")
            ).abs()
        )
        .cast(pl.Float64)
        .alias("other_accuracy"),
        (1 - (pl.col("label_ground_truth") - pl.col("doc_prob")).abs())
        .cast(pl.Float64)
        .alias("doc_accuracy"),
    )

    # Validation
    assert len(df_preds) == len(df_doc), (
        f"Join mismatch: expected {len(df_doc)} rows, got {len(df_preds)}"
    )

    return df_preds, valid_idx


def select_diverse_statements(
    df: pl.DataFrame,
    K: int,
    feature_cols: list[str],
    init_statements: list[str] | None = None,
) -> tuple[list[int], pl.DataFrame, np.ndarray]:
    """Select K diverse statements using balanced maximin criterion.

    Selects statements to maximize diversity in feature space while maintaining
    balanced representation of true/false labels (K/2 each).

    Args:
        df: DataFrame with statement features including "statement" and "label_ground_truth".
        K: Even number of statements to select.
        feature_cols: Column names for diversity features.
        init_statements: Optional list of statements to pre-select.

    Returns:
        Tuple of (selected_ids, df_selected, Xz) where:
            - selected_ids: Indices of selected statements.
            - df_selected: DataFrame with selected rows and selection_order column.
            - Xz: Z-score normalized feature matrix.

    Raises:
        ValueError: If K is odd or init_statements not found.
        AssertionError: If init_statements not all present in df.

    Example:
        >>> features = ["doc_accuracy", "other_accuracy", "consensus_score"]
        >>> selected, df_sel, Xz = select_diverse_statements(
        ...     df, K=30, feature_cols=features
        ... )
        >>> len(selected)
        30
    """
    # Find initial statement indices if provided
    init_idxs = None
    if init_statements:
        init_idxs = df.filter(pl.col("statement").is_in(init_statements))[
            "idx"
        ].to_list()
        found = len(init_idxs)
        expected = len(init_statements)
        assert found == expected, (
            f"Missing initial statements: found {found}/{expected}"
        )

    # Extract features and labels
    X = df.select(pl.col(feature_cols)).to_numpy().astype(float)
    labels = (
        X[:, 0]
        if "label_ground_truth" in feature_cols
        else df["label_ground_truth"].to_numpy()
    )

    # Standardize features
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    # Balanced maximin selection
    selected_ids = maximin_select_balanced(
        Xz, K=K, labels=labels, init_selected=init_idxs
    )

    # Create DataFrame with selection order
    df_order = pl.DataFrame(
        {"idx": selected_ids, "selection_order": range(len(selected_ids))}
    )

    df_selected = (
        df.filter(pl.col("idx").is_in(selected_ids))
        .join(df_order, on="idx", how="inner")
        .sort("selection_order")
    )

    return selected_ids, df_selected, Xz
