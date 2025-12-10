import numpy as np
from sklearn.base import BaseEstimator


# Nonconformity scores
def reciprocal_nonconformity(y, f, c=0):
    """
    Compute the reciprocal/inverse-based nonconformity scores.

    Parameters:
    -----------
    y : array-like of shape (n_samples,)
        The candidate or true labels (e.g., 0 or 1).
    f : array-like of shape (n_samples,)
        The SVM decision function outputs (distance to the hyperplane).

    Returns:
    --------
    scores : numpy array of shape (n_samples,)
        The nonconformity scores.
    """
    scores = []
    for yi, fi in zip(y, f):
        # Determine the predicted label: 1 if fi > 0, otherwise 0.
        pred = int(fi > c)
        # pred = 1 if fi > 0 else 0
        if int(yi) == int(pred):
            # Lower nonconformity (more conforming) when prediction matches.
            score = 1 / (1 + abs(fi))
        else:
            # Higher nonconformity when prediction does not match.
            score = 1 + abs(fi)
        scores.append(score)
    return np.array(scores)


def symmetric_nonconformity_with_threshold(y, f, threshold=0.0):
    """
    Compute a symmetric nonconformity score for binary classification
    with an adjustable threshold.

    Parameters:
    -----------
    y : array-like, shape (n_samples,)
        Candidate or true labels (0 or 1).
    f : array-like, shape (n_samples,)
        SVM decision function outputs.
    threshold : float, default=0.0
        The threshold to shift the decision function.

    Returns:
    --------
    scores : numpy array, shape (n_samples,)
        The nonconformity scores.

    The score is defined as:
        score = exp( - ( (2*y - 1) * (f - threshold) ) )
    """
    y_sym = 2 * np.asarray(y) - 1  # Convert 0 -> -1, 1 -> +1
    f = np.asarray(f)
    return np.exp(-y_sym * (f - threshold))


def symmetric_nonconformity(y, f):
    """
    Compute a symmetric nonconformity score for binary classification
    with an adjustable threshold.

    Parameters:
    -----------
    y : array-like, shape (n_samples,)
        Candidate or true labels (0 or 1).
    f : array-like, shape (n_samples,)
        SVM decision function outputs.
    Returns:
    --------
    scores : numpy array, shape (n_samples,)
        The nonconformity scores.

    The score is defined as:
        score = exp( - ( (2*y - 1) * (f - threshold) ) )
    """
    y_sym = 2 * np.asarray(y) - 1  # Convert 0 -> -1, 1 -> +1
    f = np.asarray(f)
    return np.exp(-y_sym * f)


def inverse_probability_nc(y, probs):
    """
    Compute the nonconformity score based on predicted probabilities for a classifier.

    Parameters:
    -----------
    y : array-like of shape (n_samples,)
        The candidate or true labels (e.g., 0, 1, 2, ...).
    probs : array-like of shape (n_samples, n_classes)
        The predicted probabilities for each class.

    Returns:
    --------
    scores : numpy array of shape (n_samples,)
        The nonconformity scores, defined as 1 minus the predicted probability
        for the candidate label.
    """
    y = np.asarray(y)
    probs = np.asarray(probs)

    # For each sample i, get the probability for the candidate label y[i]
    candidate_probs = probs[np.arange(len(y)), y]

    # Nonconformity score is defined as 1 - probability for the candidate label.
    scores = 1 - candidate_probs
    return scores


def create_symmetric_nc_with_threshold(threshold):
    def nc(a, b):
        return symmetric_nonconformity_with_threshold(a, b, threshold=threshold)

    return nc


def margin_nonconformity(y, scores):
    """
    For each sample, if the candidate label matches the predicted label
    (i.e. yhat > 0 indicates label 1, else 0), then the nonconformity score is low.
    Otherwise, it is high.
    """
    preds = scores > 0
    output = []
    for i in range(len(y)):
        if y[i] == preds[i]:
            output.append(1 / (1 + abs(scores[i])))
        else:
            output.append(1 + abs(scores[i]))
    return np.array(output)


def cumulative_softmax_nc(y, probs):
    """
    Compute the nonconformity score based on the cumulative softmax score.

    For each sample, the nonconformity score for the candidate label y is defined as
    the sum of predicted probabilities for all classes that have a higher probability than
    the candidate label. Lower scores indicate that the candidate label is strongly supported.

    Parameters:
    -----------
    y : array-like of shape (n_samples,)
        Candidate (or true) labels (e.g., 0, 1, 2, ...).
    probs : array-like of shape (n_samples, n_classes)
        Predicted probability distributions for each sample (softmax outputs).

    Returns:
    --------
    scores : numpy array of shape (n_samples,)
        The nonconformity scores.
    """
    y = np.asarray(y)
    probs = np.asarray(probs)
    scores = np.zeros_like(y, dtype=float)
    for i in range(y.shape[0]):
        candidate = int(y[i])
        p_candidate = np.array(probs[i, candidate])
        # Sum the probabilities for all classes with a probability strictly greater than the candidate's.
        scores[i] = np.sum(probs[i, :][probs[i, :] > p_candidate])
    return scores


def probability_margin_nc(y, probs):
    """
    Probability margin nonconformity functions computes the difference :math:`d_p` between the predicted probability
    of the actual class and the largest probability corresponding to some other class. To put the values on scale
    from 0 to 1, the nonconformity function returns :math:`(1 - d_p) / 2`.

    The score is defined as:
         (1.0 - (py - pz)) / 2
    where:
         - py is the probability of the candidate label y,
         - pz is the maximum probability among the other classes.

    Parameters:
    -----------
    y : int
        The candidate (or true) label (e.g., 0, 1, 2, ...).
    probs : array-like of shape (n_classes,)
        The predicted probabilities (softmax outputs) for each class.

    Returns:
    --------
    float
        The nonconformity score.
    """
    y = np.asarray(y)
    probs = np.atleast_1d(probs)
    scores = np.zeros_like(y, dtype=float)
    for i in range(y.shape[0]):
        candidate = int(y[i])
        py = probs[i, candidate]
        # Remove the candidate's probability and find the maximum among the other classes.
        other_probs = np.delete(probs[i], candidate)
        pz = np.max(other_probs)
        scores[i] = (1.0 - (py - pz)) / 2
    return scores


class InductiveConformalPredictor(BaseEstimator):
    def __init__(
        self,
        nonconformity_func=margin_nonconformity,
        alpha: float = 0.1,
        tie_breaking=True,
        **nc_kwargs,
    ):
        """
        Args:
            nonconformity_func: a function that takes true labels and scores,
                            and returns nonconformity scores.
            alpha: significance level (e.g., 0.05 for 95% confidence)
            tie_breaking: whether to break ties (if more than one candidate is in the conformal set)
            nc_kwargs: additional keyword arguments for the nonconformity function.
        """
        self.alpha = alpha
        self.nc_kwargs = nc_kwargs
        self.nc_func = nonconformity_func
        self.calibration_scores = None
        self.tie_breaking = tie_breaking
        super().__init__()

    def fit(self, y, scores):
        """
        Compute calibration scores.
        y: array-like true labels of the calibration set.
        scores: array-like decision function outputs for the calibration set.
        """
        self._is_fitted = True
        self.calibration_scores = self.nc_func(y, scores, **self.nc_kwargs)

    def _predict_set(self, scores):
        """
        Given new decision function outputs scores, for each new sample
        determine the conformal set.

        scores: array-like decision function outputs for new samples.

        Returns a list of lists; each inner list is the set of candidate labels (from {0,1})
        that are not rejected at level alpha.
        """
        assert self.calibration_scores is not None, "Fit the model first."
        conformal_sets = []
        p_vals = []
        # Ensure that scores is list-like object
        scores = np.atleast_1d(scores)
        for i in range(len(scores)):
            candidate_set = []
            p_sets = []
            # Try each candidate label (0 and 1)
            for candidate in [0, 1]:
                # Compute the candidate's nonconformity score for this sample.
                candidate_score = self.nc_func(
                    np.array([candidate]), np.array([scores[i]]), **self.nc_kwargs
                )[0]
                # Compute the p-value: the proportion of calibration scores
                # that are greater than or equal to the candidate's score.
                p_val = (np.sum(self.calibration_scores >= candidate_score) + 1) / (
                    len(self.calibration_scores) + 1
                )
                # If the candidate is not too “nonconforming,” include it.
                if p_val > self.alpha:
                    candidate_set.append(candidate)
                p_sets.append(p_val)
            conformal_sets.append(candidate_set)
            p_vals.append(p_sets)
        return conformal_sets, p_vals

    def evaluate(self, scores):
        """
        Given new decision function outputs scores, for each new sample
        determine the conformal set and return the prediction.
        Args:
            scores: array-like decision function outputs for new samples.
        Returns:
            predictions: array-like of predicted labels.
            conformal_sets: list of lists; each inner list is the set of candidate labels (from {0,1})
        that are not rejected at level alpha.
            p_values: array-like of p-values for each candidate label.
        """
        conformal_sets, p_vals = self._predict_set(scores)
        preds = []
        for i in range(len(conformal_sets)):
            if len(conformal_sets[i]) == 0:
                preds.append(-1)
            elif len(conformal_sets[i]) > 1:
                if self.tie_breaking:
                    if p_vals[i][0] > p_vals[i][1]:
                        preds.append(0)
                    else:
                        preds.append(1)
                else:
                    preds.append(-1)
            else:
                preds.append(conformal_sets[i][0])
        return {
            "predictions": np.array(preds),
            "conformal_sets": conformal_sets,
            "pvalues": np.array(p_vals),
        }

    def acceptance_rate(self, scores):
        """
        Compute the acceptance rate for new samples.
        Args:
            scores: array-like decision function outputs for new samples.
        """
        assert self.calibration_scores is not None, "Fit the model first."
        eval = self.evaluate(scores)
        abstained = eval["predictions"] == -1
        return 1 - np.mean(abstained)

    def coverage(self, scores, y):
        """
        Compute the coverage for new samples.
        """
        assert self.calibration_scores is not None, "Fit the model first."
        eval = self.evaluate(scores)
        res = [_y in _set for _y, _set in zip(y, eval["conformal_sets"])]
        return np.mean(res)

    def mask(self, scores):
        """
        Compute the mask for new samples.
        """
        assert self.calibration_scores is not None, "Fit the model first."
        eval = self.evaluate(scores)
        return eval["predictions"] != -1

    def predict_set(self, scores):
        return self.evaluate(scores)["conformal_sets"]

    def predict(self, scores):
        return self.evaluate(scores)["predictions"]

    def get_params(self, deep=False):
        """
        Get the parameters of the model.
        """
        params = {
            "alpha": self.alpha,
            "tie_breaking": self.tie_breaking,
            "nc_kwargs": self.nc_kwargs,
            "nonconformity_func": self.nc_func.__name__,
        }
        return params
