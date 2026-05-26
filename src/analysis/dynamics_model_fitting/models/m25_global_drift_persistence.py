from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import BaseDynamicsModel, get_train_transitions_from_payload, _make_logreg


class M25GlobalDriftPersistenceModel(BaseDynamicsModel):
    """Mean-field model with persistence via current agent state.

    This model combines:
    1) persistence through `belief_t` as a categorical feature, and
    2) global drift through population prevalence features at time t.
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        """Initialize the M2.5 global-drift-plus-persistence model.

        Args:
            feature_cols: Optional feature column list. If None, defaults to
                `belief_t` and the three global prevalence fractions.
            max_iter: Maximum solver iterations for logistic regression.
            random_state: Random seed for reproducible fitting.
        """
        super().__init__(
            feature_cols=feature_cols,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.feature_cols = feature_cols or [
            "belief_t",
            "global_frac_neg1_t",
            "global_frac_0_t",
            "global_frac_1_t",
        ]
        self.model: Pipeline | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit the multinomial classifier on transition rows.

        Args:
            train_df: Training dataframe containing required M2.5 feature
                columns and target column `belief_t1`.

        Raises:
            ValueError: If required feature columns are missing.
        """
        missing_cols = [col for col in self.feature_cols if col not in train_df.columns]
        if missing_cols:
            raise ValueError(
                "Training dataframe is missing required M2.5 feature columns: "
                f"{missing_cols}"
            )

        X = train_df[self.feature_cols]
        y = train_df["belief_t1"]

        categorical_cols = ["belief_t"]
        numeric_cols = [col for col in self.feature_cols if col not in categorical_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_cols,
                ),
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                ),
            ]
        )

        self.model = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    _make_logreg(
                        max_iter=self.config["max_iter"],
                        random_state=self.config["random_state"],
                    ),
                ),
            ]
        )
        self.model.fit(X, y)

    def fit_full_trajectories(self, train_payload: dict[str, Any]) -> None:
        """Fit by maximizing full trajectory likelihood.

        For this memoryless mean-field-plus-persistence model, the full trajectory
        likelihood factorizes over one-step transitions. Thus the full-trajectory
        objective reduces to the same multinomial transition fit applied to all
        transitions implied by the training trajectories.

        Args:
            train_payload: Training payload with key `train_transitions_df`.

        Raises:
            KeyError: If `train_transitions_df` is missing.
            TypeError: If `train_transitions_df` is not a DataFrame.
            ValueError: If `train_transitions_df` is empty.
        """
        train_df = get_train_transitions_from_payload(train_payload)
        self.fit(train_df)

    def predict_next(self, df: pd.DataFrame) -> pd.Series:
        """Predict the most likely next belief label.

        Args:
            df: Input dataframe containing required M2.5 feature columns.

        Returns:
            Predicted labels aligned with `df.index`.

        Raises:
            RuntimeError: If the model has not been fit.
            ValueError: If required feature columns are missing.
        """
        if self.model is None:
            raise RuntimeError("Model must be fit before prediction.")

        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                "Prediction dataframe is missing required M2.5 feature columns: "
                f"{missing_cols}"
            )

        return pd.Series(self.model.predict(df[self.feature_cols]), index=df.index)

    def predict_proba_next(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities for next-step belief labels.

        Args:
            df: Input dataframe containing required M2.5 feature columns.

        Returns:
            DataFrame of class probabilities aligned with `df.index`.

        Raises:
            RuntimeError: If the model has not been fit.
            ValueError: If required feature columns are missing.
        """
        if self.model is None:
            raise RuntimeError("Model must be fit before prediction.")

        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                "Prediction dataframe is missing required M2.5 feature columns: "
                f"{missing_cols}"
            )

        probs = self.model.predict_proba(df[self.feature_cols])
        classes = self.model.named_steps["classifier"].classes_
        return pd.DataFrame(probs, columns=classes, index=df.index)
