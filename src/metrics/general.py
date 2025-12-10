from collections import Counter
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import mutual_info_score


def label_distribution(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the distribution of labels at a specific ROUND across multiple runs.

    Args:
        labels: Array of shape (num_agents) containing belief labels

    Returns:
        Tuple of (avg_fractions, std_fractions) where each is array of shape (3,)
        representing fractions for labels [0, 1, -1]
    """
    labels_counts_per_run: List[List[int]] = []
    for run in range(labels.shape[0]):
        label_counts = [np.sum(labels[run] == val) for val in [0, 1, -1]]
        labels_counts_per_run.append(label_counts)

    # Convert to numpy array for easier calculation
    # Shape: (num_runs, 3) for [0,1,-1]
    labels_counts_array = np.array(labels_counts_per_run)

    # Calculate total agents per run
    num_agents = labels.shape[1]

    # Calculate fractions for each run (safe division)
    if num_agents > 0:
        fractions_per_run = labels_counts_array / num_agents  # Shape: (num_runs, 3)
    else:
        fractions_per_run = np.zeros_like(labels_counts_array, dtype=float)

    # Calculate average fractions and standard deviations
    avg_fractions = fractions_per_run.mean(axis=0)
    if fractions_per_run.shape[0] > 1:
        std_fractions = fractions_per_run.std(axis=0, ddof=1)  # Sample std deviation
    else:
        std_fractions = np.zeros(fractions_per_run.shape[1])
    return avg_fractions, std_fractions


def diffusion_speed(labels: np.ndarray, categories: bool = False) -> np.ndarray:
    """
    Compute the speed of belief diffusion as the rate of change in beliefs (FOR ONE RUN).

    Args:
        labels: could be scores as well | Array of shape ( num_rounds, num_agents)
        categories: whether the labels are categorical or continuous

    Returns:
        Array of shape (num_rounds-1,) with diffusion speed per round
    """
    if labels.shape[0] < 2:
        return np.array([])

    # Compute absolute change in beliefs per round, averaged over agents
    if categories:
        # For categorical labels, consider change only if label differs
        changes = (np.diff(labels, axis=0) != 0).astype(float)  # (T-1, A)
    else:
        changes = np.abs(np.diff(labels, axis=0))  # (T-1, A)
    diffusion_speed = changes.mean(axis=1)  # (T-1)

    return diffusion_speed


def compute_diffusion_speed(labels: np.ndarray, categories: bool = False) -> np.ndarray:
    """
    Compute the speed of belief diffusion as the rate of change in beliefs (ACROSS MULTIPLE RUNS).

    Args:
        labels: Array of shape (num_runs, num_rounds, num_agents)
        categories: whether the labels are categorical or continuous
    Returns:
        Array of shape (num_runs, num_rounds-1) with diffusion speed per round
    """
    if labels.shape[1] < 2:
        return np.array([])

    # Compute absolute change in beliefs per round, averaged over agents
    if categories:
        # For categorical labels, consider change only if label differs
        changes = (np.diff(labels, axis=1) != 0).astype(float)  # (R, T-1, A)
    else:
        changes = np.abs(np.diff(labels, axis=1))  # (R, T-1, A)
    diffusion_speed = changes.mean(axis=2)  # (R, T-1)

    return diffusion_speed


def consensus_fraction(labels: np.ndarray) -> float:
    """
    Compute the consensus fraction as the proportion of agents in the majority belief (FOR ONE ROUND).

    Args:
        labels: Array of shape (num_agents,) containing belief labels

    Returns:
        float: Fraction of agents with the majority belief
    """
    N = len(labels)
    if N > 0:
        belief_counts = Counter(labels)
        consensus_frac = float(max(belief_counts.values()) / N)
    else:
        consensus_frac = 0.0

    return consensus_frac


def belief_change_metrics(
    prev: Optional[np.ndarray], curr: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute flip rate, L1 change, and L2 change between two rounds of beliefs.
    Args:
        prev: array of previous-round beliefs (or None)
        curr: array of current beliefs

    Returns: (flip_rate, change_l1, change_l2)
    """
    N = len(curr)
    if prev is None or len(prev) != N:
        return 0.0, 0.0, 0.0

    prev = prev.astype(float)
    curr = curr.astype(float)

    changes = curr != prev
    flip_rate = float(np.sum(changes) / N) if N > 0 else 0.0

    diff = curr - prev
    change_l1 = float(np.linalg.norm(diff, ord=1))
    change_l2 = float(np.linalg.norm(diff, ord=2))

    return flip_rate, change_l1, change_l2


def binary_magnetization(beliefs: np.ndarray) -> float:
    """Compute magnetization for binary beliefs (0 or 1).

    Magnetization = |mean(2*b_i - 1)|.

    Args:
        beliefs: array of beliefs (0 or 1) of shape (num_agents,)

    Returns:
        Magnetization value.
    """
    if len(beliefs) == 0:
        return 0.0

    b = beliefs.astype(int)
    spins = 2 * b - 1  # map {0,1} -> {-1,+1}
    return float(np.abs(np.mean(spins)))


def binary_polarization(beliefs: np.ndarray) -> float:
    """Compute polarization for binary beliefs (0 or 1).

    Polarization index: P = 4 p (1 - p) where p is the fraction of 1s.

    Args:
        beliefs: array of beliefs (0 or 1) of shape (num_agents,)

    Returns:
        Polarization value in [0, 1]:
            0 -> full consensus (all 0 or all 1)
            1 -> perfect 50-50 split
    """
    if len(beliefs) == 0:
        return 0.0

    b = beliefs.astype(int)
    p = b.mean()  # fraction of 1s

    return float(4.0 * p * (1.0 - p))


def binary_entropy(beliefs: np.ndarray) -> float:
    """
    Compute entropy of binary beliefs (0/1).
    Returns entropy in nats (log base e).
    Args:
        beliefs: array of beliefs (0 or 1) of shape (num_agents,)
    Returns:
        float: entropy value
    """
    if len(beliefs) == 0:
        return 0.0

    b = beliefs.astype(int)
    p = b.mean()  # fraction of 1s

    if p < 1e-12 or p > 1 - 1e-12:
        return 0.0  # full consensus

    return float(-p * np.log(p) - (1 - p) * np.log(1 - p))


def binary_entropy_normalized(beliefs: np.ndarray) -> float:
    """
    Compute normalized entropy of binary beliefs (0/1).
    Normalized entropy is in [0, 1], where 0 means full consensus and 1 means maximum uncertainty.
    Args:
        beliefs: array of beliefs (0 or 1) of shape (num_agents,)
    Returns:
        float: normalized entropy value in [0, 1]
    """
    H = binary_entropy(beliefs)
    return float(H / np.log(2.0))  # normalize to [0,1]


def entropy_multiclass_normalized(beliefs: np.ndarray) -> float:
    """
    Compute normalized entropy for multi-class categorical beliefs.
    Args:
        beliefs: array of beliefs (categorical labels) of shape (num_agents,)
    Returns:
        float: normalized entropy value in [0, 1]
    """
    if len(beliefs) == 0:
        return 0.0

    values, counts = np.unique(beliefs, return_counts=True)
    K = len(values)

    # If only one unique value, entropy is 0
    if K == 1:
        return 0.0

    p = counts / counts.sum()
    H = -np.sum(p * np.log(p + 1e-12))

    return float(H / np.log(K))


def granger_influence_score(history: Sequence[np.ndarray]) -> np.ndarray:
    """
    Compute Granger influence score based on history of beliefs.
    Args:
        history: Sequence of belief arrays over time (each np.ndarray of shape (num_agents,))
    Returns:
        np.ndarray: Granger influence score per agent
    """
    X = np.array(history)  # Shape: (T, N)
    T, N = X.shape
    if T < 2:
        return np.zeros(N)

    # Past and future slices
    X_past = X[:-1]  # shape (T-1, N)
    X_future = X[1:]  # shape (T-1, N)

    influence = np.zeros(N)

    for i in range(N):
        xi = X_past[:, i]  # agent i past

        for j in range(N):
            if i == j:
                continue

            y_j_future = X_future[:, j]
            y_j_past = X_past[:, j]

            # ------- Baseline model: y_j_future ~ y_j_past --------
            X_base = y_j_past.reshape(-1, 1)  # (T-1, 1)
            coef_base, *_ = np.linalg.lstsq(X_base, y_j_future, rcond=None)
            pred_base = X_base @ coef_base
            rss_base = np.sum((y_j_future - pred_base) ** 2)

            # ------- Expanded model: y_j_future ~ y_j_past + x_i_past --------
            X_exp = np.column_stack([y_j_past, xi])  # (T-1, 2)
            coef_exp, *_ = np.linalg.lstsq(X_exp, y_j_future, rcond=None)
            pred_exp = X_exp @ coef_exp
            rss_exp = np.sum((y_j_future - pred_exp) ** 2)

            # Influence = error reduction
            influence[i] += max(0.0, rss_base - rss_exp)

    return influence


def simple_influence_scores(history: Sequence[np.ndarray]) -> np.ndarray:
    """
    Influence_i = sum_j corr( b_i(t-1), b_j(t) ), j != i (based on scores)
    Args:
        history: Sequence of belief arrays over time (each np.ndarray of shape (num_agents,))
    Returns:
        np.ndarray: array of influence scores per agent
    """
    if not history or len(history) < 2:
        return np.array([])

    # Matrix shape (T, N)
    X = np.vstack(history)
    T, N = X.shape

    # Past (T-1, N) and future (T-1, N)
    X_past = X[:-1]
    X_fut = X[1:]

    # Compute columnwise means and stds
    xm = X_past.mean(axis=0)
    xs = X_past.std(axis=0)
    ym = X_fut.mean(axis=0)
    ys = X_fut.std(axis=0)

    # Avoid division by zero
    xs[xs < 1e-12] = 1e-12
    ys[ys < 1e-12] = 1e-12

    # Normalize columns
    Xp = (X_past - xm) / xs
    Yp = (X_fut - ym) / ys

    # Cross-correlation matrix R[i,j] = corr(i_past, j_future)
    # This is matrix multiplication normalized by T-1
    R = (Xp.T @ Yp) / (T - 1)

    # Influence_i = sum over j != i of R[i,j]
    np.fill_diagonal(R, 0.0)

    return R.sum(axis=1)


def _binary_mi(x: np.ndarray, y: np.ndarray) -> float:
    """
    Mutual information between two binary sequences x,y in {0,1}.
    """
    x = x.astype(int)
    y = y.astype(int)

    T = len(x)
    if T == 0:
        return 0.0

    # Joint counts
    p00 = np.sum((x == 0) & (y == 0)) / T
    p01 = np.sum((x == 0) & (y == 1)) / T
    p10 = np.sum((x == 1) & (y == 0)) / T
    p11 = np.sum((x == 1) & (y == 1)) / T

    p0x = p00 + p01  # P(X=0)
    p1x = p10 + p11  # P(X=1)
    p0y = p00 + p10  # P(Y=0)
    p1y = p01 + p11  # P(Y=1)

    mi = 0.0
    for pxy, px, py in [
        (p00, p0x, p0y),
        (p01, p0x, p1y),
        (p10, p1x, p0y),
        (p11, p1x, p1y),
    ]:
        if pxy > 0:
            mi += pxy * np.log(pxy / (px * py + 1e-12) + 1e-12)

    # Clip to zero to avoid negative values from floating point errors
    return float(max(0.0, mi))


def mutual_information_matrix(belief_history: List[np.ndarray]) -> np.ndarray:
    """
    Compute NxN mutual information matrix for *binary* beliefs.
    Args:
        belief_history: list of arrays (T arrays of length N)
    Returns:
        np.ndarray: NxN mutual information matrix
    """
    if not belief_history:
        return np.array([])

    X = np.vstack(belief_history).astype(int)  # shape T x N
    T, N = X.shape

    mi = np.zeros((N, N), float)

    for i in range(N):
        for j in range(N):
            if i == j:
                mi[i, j] = np.nan
            else:
                mi[i, j] = _binary_mi(X[:, i], X[:, j])

    return mi


def transfer_entropy_matrix(
    belief_history: List[np.ndarray], A: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute NxN transfer entropy matrix for *binary* beliefs.

    Args:
        belief_history: list of arrays (T arrays of length N)
        A: Optional adjacency matrix to mask non-neighbors

    Returns:
        NxN transfer entropy matrix
    """
    X = np.vstack(belief_history).astype(int)  # T x N
    T, N = X.shape
    te = np.zeros((N, N), dtype=float)
    if A is not None:
        A = np.asarray(A, dtype=float)
        te = te * A

    for i in range(N):
        xi = X[:-1, i]  # past of i
        for j in range(N):
            if i == j:
                te[i, j] = np.nan
                continue
            y_t = X[:-1, j]  # past of j
            y_tp1 = X[1:, j]  # future of j

            # MI( [i_t, j_t] ; j_{t+1} )
            joint = xi * 2 + y_t  # binary pair encoded as {0,1,2,3}
            mi_joint = _binary_mi(joint, y_tp1)

            # MI( j_t ; j_{t+1} )
            mi_base = _binary_mi(y_t, y_tp1)

            te[i, j] = max(mi_joint - mi_base, 0.0)

    return te


def information_leadership(te_matrix: np.ndarray) -> np.ndarray:
    """
    Compute leadership score for each agent:
    sum of outgoing TE influence.
    Args:
        te_matrix: NxN transfer entropy matrix
    Returns:
        np.ndarray: array of leadership scores per agent
    """
    te = np.array(te_matrix, dtype=float)
    np.fill_diagonal(te, 0.0)

    # Leadership = sum over outgoing edges
    leadership = te.sum(axis=1)
    return leadership


def influence_matrix(te_matrix: np.ndarray) -> np.ndarray:
    """
    Build influence network where edges(i->j) = TE[i,j].
    Removes NaN on diagonal and returns a clean adjacency matrix.
    Args:
        te_matrix: NxN transfer entropy matrix
    Returns:
        np.ndarray: NxN influence adjacency matrix
    """
    te = np.array(te_matrix, dtype=float)

    # Replace diagonal NaNs with 0
    np.fill_diagonal(te, 0.0)

    # Negative values shouldn't exist (TE is >= 0),
    # but clamp just in case of numerical noise:
    te[te < 0] = 0.0

    return te


### Continious TE


def graph_transfer_entropy(
    beliefs_history: np.ndarray, adjacency_matrix: np.ndarray, bins: int = 5
) -> np.ndarray:
    """
    Transfer entropy using sklearn's mutual information implementation.
    """
    beliefs_history = np.asarray(beliefs_history, dtype=float)

    if adjacency_matrix.shape[0] != beliefs_history.shape[1]:
        beliefs_history = beliefs_history.T

    T, N = beliefs_history.shape
    A = np.asarray(adjacency_matrix)
    TE = np.zeros((N, N))

    # Discretize all data at once
    min_val, max_val = np.min(beliefs_history), np.max(beliefs_history)
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    beliefs_discrete = np.clip(np.digitize(beliefs_history, bin_edges) - 1, 0, bins - 1)

    for i in range(N):
        for j in range(N):
            if i == j or A[i, j] <= 0:
                continue

            # Time series
            x_tm1 = beliefs_discrete[:-1, i]
            y_tm1 = beliefs_discrete[:-1, j]
            y_t = beliefs_discrete[1:, j]

            # FIX: Create combined state for (Y_{t-1}, X_{t-1})
            # Method 1: Create unique integer for each combination
            y_x_combined = y_tm1 * bins + x_tm1  # This creates unique ID for each pair

            # Alternative Method 2: Use string concatenation (slower but clearer)
            # le = LabelEncoder()
            # y_x_combined = le.fit_transform(
            #     [f"{y}_{x}" for y, x in zip(y_tm1, x_tm1)]
            # )

            # TE = I(Y_t; X_{t-1} | Y_{t-1})
            # Using chain rule: I(Y_t; Y_{t-1}, X_{t-1}) - I(Y_t; Y_{t-1})
            mi_joint = mutual_info_score(y_t, y_x_combined)
            mi_y_only = mutual_info_score(y_t, y_tm1)

            TE[i, j] = max(mi_joint - mi_y_only, 0.0)

    total_te = np.sum(TE)
    mean_te = np.mean(TE[A > 0]) if np.any(A > 0) else 0.0

    return TE, total_te, mean_te


def collective_accuracy_scores(beliefs: np.ndarray, correct_label: int) -> np.ndarray:
    """
    Compute collective accuracy as the fraction of agents holding the correct belief.
    Takes into account the belief scores.

    Args:
        beliefs: array of beliefs (categorical labels) of shape (num_agents,)
        correct_label: the correct belief label (0 or 1)
    """
    assert correct_label in [0, 1], "correct_label must be 0 or 1"

    return 1 - np.abs(beliefs - correct_label)


def collective_accuracy_label(beliefs: np.ndarray, correct_label: int) -> np.ndarray:
    """
    Compute collective accuracy as the fraction of agents holding the correct belief.
    Takes into account the belief labels.

    Args:
        beliefs: array of beliefs (categorical labels) of shape (num_agents,)
        correct_label: the correct belief label (0 or 1)
    """
    assert correct_label in [0, 1], "correct_label must be 0 or 1"
    return (beliefs == correct_label).astype(float)
