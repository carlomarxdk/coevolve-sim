from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from sklearn.linear_model import LogisticRegression
import inspect

def _make_logreg(max_iter: int, random_state: int) -> LogisticRegression:
    """Build LogisticRegression with cross-version compatible kwargs.

    Args:
        max_iter: Maximum optimizer iterations.
        random_state: Random seed for reproducibility.

    Returns:
        Configured LogisticRegression instance.
    """
    kwargs: dict[str, Any] = {
        "solver": "lbfgs",
        "max_iter": max_iter,
        "random_state": random_state,
    }
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        kwargs["multi_class"] = "multinomial"
    return LogisticRegression(**kwargs)


def get_train_transitions_from_payload(train_payload: dict[str, Any]) -> pd.DataFrame:
    """Extract the training transition dataframe from a trajectory payload."""
    if "train_transitions_df" not in train_payload:
        raise KeyError("train_payload is missing 'train_transitions_df'.")

    train_df = train_payload["train_transitions_df"]

    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("'train_transitions_df' must be a pandas DataFrame.")

    if train_df.empty:
        raise ValueError("'train_transitions_df' is empty.")

    return train_df


class BaseDynamicsModel(ABC):
    """Abstract base class for transition-based opinion dynamics models."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit model parameters using training transitions."""

    @abstractmethod
    def fit_full_trajectories(self, train_payload: dict[str, Any]) -> None:
        """Fit the model using full trajectory information."""

    @abstractmethod
    def predict_next(self, df: pd.DataFrame) -> pd.Series:
        """Predict the next belief label for each row."""

    def predict_proba_next(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optionally return class probabilities."""
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        return dict(self.config)