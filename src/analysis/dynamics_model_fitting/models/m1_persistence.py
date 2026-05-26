from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseDynamicsModel, get_train_transitions_from_payload


class M1PersistenceModel(BaseDynamicsModel):
    """Empirical first-order persistence model.

    This model estimates the transition probabilities
    P(X_{t+1} | X_t) directly from frequency counts in the training data.
    """

    def __init__(self) -> None:
        """Initialize the persistence model state."""
        super().__init__()
        self.transition_probs: pd.DataFrame | None = None
        self.majority_next_by_current: dict[float, float] | None = None
        self.classes_: list[float] | None = None
        self.feature_cols = ["belief_t"]

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit transition probabilities from observed transitions.

        Args:
            train_df: Training dataframe containing at least `belief_t`
                and `belief_t1` columns.
        """
        counts = pd.crosstab(train_df["belief_t"], train_df["belief_t1"])
        probs = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)

        self.transition_probs = probs
        self.classes_ = [float(c) for c in probs.columns]
        self.majority_next_by_current = probs.idxmax(axis=1).to_dict()

    def fit_full_trajectories(self, train_payload: dict[str, Any]) -> None:
        """Fit by maximizing full trajectory likelihood.

        For this first-order Markov persistence model, the trajectory likelihood
        factorizes over transitions, so fitting on full trajectories reduces to
        counting all observed training transitions implied by those trajectories.
        """
        train_df = get_train_transitions_from_payload(train_payload)
        self.fit(train_df)

    def predict_next(self, df: pd.DataFrame) -> pd.Series:
        """Predict the most likely next belief label for each row.

        Args:
            df: Input dataframe containing `belief_t`.

        Returns:
            Series of predicted labels aligned to `df.index`.

        Raises:
            RuntimeError: If the model has not been fit.
        """
        if self.majority_next_by_current is None:
            raise RuntimeError("Model must be fit before prediction.")

        return df["belief_t"].map(self.majority_next_by_current)

    def predict_proba_next(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict next-label probabilities for each row.

        Args:
            df: Input dataframe containing `belief_t`.

        Returns:
            DataFrame where each row contains class probabilities for the
            next belief label, indexed by `df.index`.

        Raises:
            RuntimeError: If the model has not been fit.
        """
        if self.transition_probs is None:
            raise RuntimeError("Model must be fit before prediction.")

        rows = []
        for belief in df["belief_t"]:
            if belief in self.transition_probs.index:
                rows.append(self.transition_probs.loc[belief].to_dict())
            else:
                rows.append({c: 0.0 for c in self.transition_probs.columns})

        return pd.DataFrame(rows, index=df.index)
