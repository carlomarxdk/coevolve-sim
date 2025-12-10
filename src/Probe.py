from __future__ import annotations

import logging
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


log = logging.getLogger("Probe")


class BinaryLinearProbe(BaseEstimator, ClassifierMixin):
    """Pre-trained binary linear probe for belief scoring.

    This probe uses pre-trained coefficients and intercept to classify inputs
    without requiring additional training. It follows the scikit-learn estimator
    interface for compatibility.

    Adapted from: https://github.com/carlomarxdk/trilemma-of-truth

    Attributes:
        coef: Linear coefficients for the probe.
        intercept: Intercept term for the probe.
        coef_: Alias for coef (scikit-learn convention).
        intercept_: Alias for intercept (scikit-learn convention).
        classes_: Binary classes [0, 1].
        is_fitted_: Always True since probe is pre-trained.
    """

    def __init__(self, coef: np.ndarray, intercept: np.ndarray):
        """Initialize probe with pre-trained parameters.

        Args:
            coef: Linear coefficients array.
            intercept: Intercept values array.
        """
        self.coef = np.asarray(coef)
        self.intercept = np.asarray(intercept).ravel()
        self.coef_ = self.coef
        self.intercept_ = self.intercept
        self.classes_ = [0, 1]
        self.is_fitted_ = True

    def fit(self, X: np.ndarray = None, y: np.ndarray = None) -> "BinaryLinearProbe":
        """No-op fit method (probe is pre-trained).

        Args:
            X: Ignored.
            y: Ignored.

        Returns:
            self
        """
        log.warning("This probe is pre-trained and does not require fitting.")
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function scores for samples.

        Args:
            X: Input array of shape (N, d) where N is number of samples
               and d is feature dimension.

        Returns:
            Array of shape (N,) containing decision scores.
        """
        check_is_fitted(self)
        X = np.asarray(X)
        return (X @ self.coef_.T).ravel() + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using sigmoid of decision function.

        Args:
            X: Input array of shape (N, d).

        Returns:
            Array of shape (N,) containing probabilities in [0, 1].
        """
        return expit(self.decision_function(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels.

        Args:
            X: Input array of shape (N, d).

        Returns:
            Array of shape (N,) containing binary predictions (0 or 1).
        """
        return self.predict_proba(X).round()


class Probe:
    """Neural probe for scoring agent beliefs from model activations.

    Manages loading and using pre-trained probes to extract belief scores
    from language model hidden states. Probes are loaded from disk and
    optionally calibrated.

    Attributes:
        cfg: Probe configuration dictionary.
        model_cfg: Model-specific configuration.
        io: IOManager for file operations.
        _name: Name of the probe (e.g., 'sawmil').
        _basemodel: Name of the base model.
        _layer: Model layer to extract features from.
        cls: BinaryLinearProbe instance.
        scaler: StandardScaler for feature normalization.
        calibrator: Optional calibrator for probability calibration.
    """

    def __init__(self, cfg: Dict, io, model_cfg: Dict = None):
        """Initialize probe configuration.

        Args:
            cfg: Probe configuration containing probe name.
            io: IOManager instance for file operations.
            model_cfg: Model-specific configuration containing probe settings.
        """
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.io = io

        self._name = cfg.get("name")  # probe name
        self._basemodel = self.model_cfg.get("model").get("name")
        self._layer = (
            self.model_cfg.get("model").get("probe").get(self._name).get("layer")
        )

    def load_probe(self) -> None:
        """Load probe, scaler, and optional calibrator from disk.

        Loads the pre-trained linear probe coefficients, feature scaler,
        and optional conformal calibrator from the resources/probes directory.
        """
        path = Path(f"./resources/probes/{self._name}/{self._basemodel}/")

        self.cls = BinaryLinearProbe(
            coef=np.load(path / f"coef_{self._layer}c.npy", allow_pickle=True),
            intercept=np.load(path / f"bias_{self._layer}c.npy", allow_pickle=True),
        )
        self.scaler = joblib.load(path / f"scaler_{self._layer}.joblib")
        self.calibrator = joblib.load(path / f"calibrator_{self._layer}.joblib")

    def score(self, activation: np.ndarray, *args, **kwargs) -> Tuple[float, float]:
        """Generate belief score from model activations.

        Extracts features from the specified layer, scales them, applies the
        linear probe, and optionally calibrates the output.

        Args:
            activation: Array of model activations from all layers.
                       Expected shape: (num_layers, batch_size, hidden_dim).
            *args: Additional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Tuple of (calibrated_label, raw_score) where:
                - calibrated_label: Calibrated prediction (-1, 0, or 1)
                - raw_score: Raw decision function score
        """
        X = activation[self._layer]
        _, n, d = X.shape
        # Use last 10 tokens
        X = X.reshape(n, d)[-10:]

        X = self.scaler.transform(X)
        scores = np.max(self.cls.decision_function(X))
        yhat = self.calibrator.predict([scores])
        return yhat.ravel()[0], scores.ravel()[0]


class ZeroShotProbe:
    """Zero-shot probe for belief scoring using token probabilities.

    This probe extracts logit probabilities for specific answer tokens
    (1-6) and maps them to belief classes:
    - Token '1' → True (class 1)
    - Token '2' → False (class 0)
    - Tokens '3', '4', '5', '6' → Uncertain (class -1)

    Attributes:
        cfg: Probe configuration dictionary.
        model_cfg: Model-specific configuration.
        io: IOManager for file operations.
        _name: Name of the probe (e.g., 'zeroshot').
        tokenizer: Tokenizer for encoding answer tokens.
        token_ids: Dictionary mapping answer options to token IDs.
    """

    def __init__(self, cfg: Dict, io, model_cfg: Dict = None):
        """Initialize zero-shot probe configuration.

        Args:
            cfg: Probe configuration containing probe name.
            io: IOManager instance for file operations.
            model_cfg: Model-specific configuration.
        """
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.io = io
        self._name = cfg.get("name")  # probe name
        self.tokenizer = None
        self.token_ids = {}

    def load_probe(self, tokenizer=None) -> None:
        """Load tokenizer and prepare token IDs for logit extraction.

        Args:
            tokenizer: Optional tokenizer instance. If None, will be set later.
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer
            self._prepare_token_ids()

    def augment_token_list(self, tokens: List[str]) -> Dict[str, List[str]]:
        result = {}
        for t in tokens:
            _r = [t, f" {t}"]
            result[t] = _r
        return result

    def set_tokenizer(self, tokenizer) -> None:
        """Set the tokenizer and prepare token IDs.

        Args:
            tokenizer: Tokenizer instance for encoding tokens.
        """
        self.tokenizer = tokenizer
        self._prepare_token_ids()

    def _prepare_token_ids(self) -> None:
        """Prepare token IDs for answer options 1-6.

        Creates augmented token lists (with and without leading space)
        and filters out shared tokens to ensure unique mappings.
        """
        if self.tokenizer is None:
            log.warning("Tokenizer not set, cannot prepare token IDs")
            return

        # Answer options 1-6
        answer_tokens = ["1", "2", "3", "4", "5", "6"]

        # Augment token list with variants (with and without leading space)
        augmented_tokens = {}
        for token in answer_tokens:
            variants = [token, f" {token}"]
            augmented_tokens[token] = variants

        # Get token IDs and filter shared tokens
        self.token_ids = self._get_unique_token_ids(augmented_tokens)
        log.debug(f"Prepared token IDs for zero-shot probe: {self.token_ids}")

    def _get_unique_token_ids(
        self, token_dict: Dict[str, List[str]]
    ) -> Dict[str, List[int]]:
        """Get unique token IDs for each answer option.

        Filters out tokens that appear in multiple answer options to avoid
        ambiguous mappings.

        Args:
            token_dict: Dictionary mapping answer options to token variants.

        Returns:
            Dictionary mapping answer options to unique token IDs.
        """
        # Collect all token IDs and count occurrences
        token_counts = Counter()
        for variants in token_dict.values():
            unique_tokens = set()
            for variant in variants:
                encoded = self.tokenizer.encode(variant)
                unique_tokens.update(encoded)
            token_counts.update(unique_tokens)

        # Identify shared tokens (appearing in multiple options)
        shared_tokens = {token for token, count in token_counts.items() if count > 1}

        # Filter out shared tokens from each option
        output = {}
        for option, variants in token_dict.items():
            unique_ids = []
            for variant in variants:
                encoded = self.tokenizer.encode(variant, add_special_tokens=False)
                unique_ids.extend([tok for tok in encoded if tok not in shared_tokens])
            # Remove duplicates
            output[option] = list(set(unique_ids))

        return output

    def score(self, logits: np.ndarray, *args, **kwargs) -> Tuple[float, float]:
        """Generate belief score from output logits.

        Extracts probabilities for answer tokens 1-6 and maps them to
        belief classes (True=1, False=0, Uncertain=-1).

        Args:
            logits: Logits from the model's final layer.
                   Expected shape: (batch_size, vocab_size) or (vocab_size,).
            *args: Additional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Tuple of (calibrated_label, p_true) where:
                - calibrated_label: Predicted class (-1, 0, or 1)
                - p_true: Probability of True class
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set. Call set_tokenizer() first.")

        # Ensure logits is 2D
        if isinstance(logits, np.ndarray):
            if logits.ndim == 1:
                logits = logits.reshape(1, -1)
        else:
            # Handle tensor input
            import torch

            if isinstance(logits, torch.Tensor):
                logits = logits.cpu().numpy()
                if logits.ndim == 1:
                    logits = logits.reshape(1, -1)
        # Extract probabilities for each answer option
        probs = self._collect_probabilities(logits)
        log.debug(f"Probabilities: {probs}")
        # Map to belief classes
        # Token '1' → True (class 1)
        # Token '2' → False (class 0)
        # Tokens '3', '4', '5', '6' → Uncertain (class -1)
        p_true = probs.get("1", 0.0)
        p_false = probs.get("2", 0.0)
        p_uncertain = sum(probs.get(str(key), 0.0) for key in [3, 4, 5, 6, "else"])

        # Determine predicted class based on highest probability
        max_prob = max(p_true, p_false, p_uncertain)
        if p_true == max_prob:
            predicted_class = 1.0
            # confidence = p_true
        elif p_false == max_prob:
            predicted_class = 0.0
            # confidence = p_false
        else:
            predicted_class = -1.0
            # confidence = p_uncertain

        log.debug(f"Probabilities: {probs}")
        # Raw score is the confidence of the prediction
        # raw_score = confidence

        return predicted_class, p_true  # raw_score

    def _collect_probabilities(self, logits: np.ndarray) -> Dict[str, float]:
        """Collect probabilities for each answer option from logits.

        Aggregates logits for each option by summation and normalizes them using
        softmax to create a probability distribution over the answer space.
        Unaccounted probability mass is assigned to an "else" class, then all
        probabilities are computed using softmax over the aggregated logits.

        Note: This sums logits directly (not using logsumexp), which creates
        a different probability distribution than summing the original token
        probabilities. This is intentional and follows the pattern suggested
        in the original TODO comment.

        Args:
            logits: Logits tensor of shape (batch_size, vocab_size).

        Returns:
            Dictionary mapping answer options ('1'-'6') to their probabilities.
            When answer tokens dominate, probabilities sum to ~1.0.
            When other tokens dominate ("else" category), probabilities are low.
        """
        # Sum logits for each answer option
        probs = softmax(logits[0], axis=-1)  # Convert logits to probabilities

        option_probs = []
        option_names = []

        for option, token_ids in self.token_ids.items():
            probs_sum = 0.0
            for token_id in token_ids:
                probs_sum += probs[token_id]
            option_probs.append(probs_sum)
            option_names.append(option)

        # Calculate "else" logit for unaccounted probability mass
        else_probs = 1 - sum(option_probs)
        option_probs.append(else_probs)

        # print('Option probabilities  :', option_probs)

        # option_probs_array = np.array(option_probs)

        # # Map back to dictionary (includes all 6 options, excludes "else")
        # option_probs = {}
        # for i, option in enumerate(option_names):
        #     option_probs_array[option] = float(probs[i])

        output = {}
        for i, option in enumerate(option_names + ["else"]):
            output[option] = float(option_probs[i])

        return output
