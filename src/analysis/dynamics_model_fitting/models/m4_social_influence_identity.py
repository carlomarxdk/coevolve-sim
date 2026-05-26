from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import BaseDynamicsModel, get_train_transitions_from_payload, _make_logreg


class M4SocialInfluenceIdentityModel(BaseDynamicsModel):
    """Social influence model with model/role identity.

    This model predicts next-step belief labels using:
    1) the agent's current belief,
    2) neighbor belief composition at time t, and
    3) the agent's model identity and assigned role.

    Conceptually:
        P(X_{t+1} | X_t, neighbor composition, agent_model, agent_role)
    """

    DEFAULT_SOCIAL_FEATURES = [
        "belief_t",
        "neighbor_frac_neg1_t",
        "neighbor_frac_0_t",
        "neighbor_frac_1_t",
    ]

    DEFAULT_IDENTITY_FEATURES = [
        "agent_model",
        "agent_role",
    ]

    def __init__(
        self,
        social_feature_cols: list[str] | None = None,
        identity_cols: list[str] | None = None,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        """Initialize the M4 social influence with identity model.

        Args:
            social_feature_cols: Core M3-style social influence features.
                If None, uses `belief_t` and neighbor belief fractions.
            identity_cols: Identity columns to include as categorical features.
                If None, uses `agent_model` and `agent_role`.
            max_iter: Maximum iterations for the logistic regression solver.
            random_state: Random seed for reproducible model fitting.
        """
        super().__init__(
            social_feature_cols=social_feature_cols,
            identity_cols=identity_cols,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.social_feature_cols = social_feature_cols or list(
            self.DEFAULT_SOCIAL_FEATURES
        )
        self.identity_cols = identity_cols or list(self.DEFAULT_IDENTITY_FEATURES)
        self.feature_cols = self.social_feature_cols + self.identity_cols
        self.model: Pipeline | None = None

    def _validate_feature_cols(self, df: pd.DataFrame) -> None:
        """Check that all required M4 feature columns are present."""
        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                "Dataframe is missing required M4 feature columns: "
                f"{missing_cols}"
            )

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit the multinomial logit model on transition rows.

        Args:
            train_df: Training dataframe containing feature columns and
                target column `belief_t1`.

        Raises:
            ValueError: If any required feature columns are missing.
        """
        self._validate_feature_cols(train_df)

        X = train_df[self.feature_cols]
        y = train_df["belief_t1"]

        categorical_cols = [
            "belief_t",
            *self.identity_cols,
        ]
        numeric_cols = [
            col for col in self.feature_cols if col not in categorical_cols
        ]

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

        As with M3, this model factorizes over one-step conditionals given the
        current trajectory state, neighbor composition, and identity features.
        Therefore, the full-trajectory objective reduces to fitting on all
        training transitions implied by the observed trajectories.

        Args:
            train_payload: Training payload that must include
                `train_transitions_df`.
        """
        train_df = get_train_transitions_from_payload(train_payload)
        self.fit(train_df)

    def _check_ready_for_prediction(self, df: pd.DataFrame) -> None:
        """Validate model state and prediction columns."""
        if self.model is None:
            raise RuntimeError("Model must be fit before prediction.")

        self._validate_feature_cols(df)

    def predict_next(self, df: pd.DataFrame) -> pd.Series:
        """Predict the most likely next belief label for each row.

        Args:
            df: Input dataframe containing the required feature columns.

        Returns:
            A Series of predicted next-step belief labels aligned to `df.index`.
        """
        self._check_ready_for_prediction(df)

        assert self.model is not None
        preds = self.model.predict(df[self.feature_cols])
        return pd.Series(preds, index=df.index)

    def predict_proba_next(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities for next-step belief labels.

        Args:
            df: Input dataframe containing the required feature columns.

        Returns:
            DataFrame of class probabilities with columns equal to the
            classifier's learned classes and index aligned to `df.index`.
        """
        self._check_ready_for_prediction(df)

        assert self.model is not None
        probs = self.model.predict_proba(df[self.feature_cols])
        classes = self.model.named_steps["classifier"].classes_
        return pd.DataFrame(probs, columns=classes, index=df.index)

    def get_params(self) -> dict[str, Any]:
        """Return model configuration and resolved feature columns."""
        params = super().get_params()
        params["feature_cols"] = self.feature_cols
        params["social_feature_cols"] = self.social_feature_cols
        params["identity_cols"] = self.identity_cols
        return params