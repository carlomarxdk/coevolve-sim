"""Diversity metrics and maximin selection algorithms."""

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA


def maximin_select(
    X: np.ndarray, K: int, init_selected: list[int] | None = None
) -> list[int]:
    """Select K points from X using the maximin criterion.

    The maximin criterion iteratively selects points that maximize the minimum
    distance to already-selected points, promoting diversity in feature space.

    Args:
        X: Feature matrix of shape (N, D).
        K: Number of points to select.
        init_selected: Pre-selected point indices to include.

    Returns:
        List of selected point indices.

    Example:
        >>> X = np.random.rand(100, 5)
        >>> selected = maximin_select(X, K=10)
        >>> len(selected)
        10
    """
    if init_selected is not None and len(init_selected) > 0:
        selected = list(init_selected)
        remaining = K - len(selected)
        if remaining <= 0:
            return selected[:K]
    else:
        selected = [int(np.argmax(np.linalg.norm(X, axis=1)))]
        remaining = K - 1

    for _ in range(remaining):
        distances = cdist(X, X[selected], metric="euclidean")
        min_dist = distances.min(axis=1)
        min_dist[selected] = -np.inf
        next_idx = np.argmax(min_dist)
        selected.append(int(next_idx))

    return selected


def maximin_select_balanced(
    X: np.ndarray,
    K: int,
    labels: np.ndarray,
    init_selected: list[int] | None = None,
) -> list[int]:
    """Select K points with balanced binary labels via alternating maximin.

    Maintains equal representation of binary classes (K/2 each) while maximizing
    minimum pairwise distances. Alternates selection between classes to ensure
    balance throughout the process.

    Args:
        X: Feature matrix of shape (N, D).
        K: Even number of points to select.
        labels: Binary labels of shape (N,).
        init_selected: Optional pre-selected indices.

    Returns:
        Indices of selected points.

    Raises:
        ValueError: If K is odd, per-class quota exceeded, or not enough samples.

    Example:
        >>> X = np.random.rand(100, 5)
        >>> labels = np.random.randint(0, 2, 100)
        >>> selected = maximin_select_balanced(X, K=20, labels=labels)
        >>> sum(labels[selected] == 0)
        10
    """
    if K % 2:
        raise ValueError(f"K must be even, got {K}")
    quota = K // 2

    selected = list(init_selected or [])
    c0 = sum(labels[i] == 0 for i in selected)
    c1 = len(selected) - c0
    if c0 > quota or c1 > quota:
        raise ValueError("Pre-selected indices exceed per-class quota")

    if not selected:
        first = int(np.argmax(np.linalg.norm(X, axis=1)))
        selected.append(first)
        c0 += labels[first] == 0
        c1 += labels[first] == 1

    if np.sum(labels == 0) < quota or np.sum(labels == 1) < quota:
        raise ValueError("Insufficient samples to fill balanced quotas")

    while c0 < quota or c1 < quota:
        target = 0 if c0 < quota else 1
        dist = cdist(X, X[selected], metric="euclidean").min(axis=1)
        dist[selected] = -np.inf
        dist[labels != target] = -np.inf
        nxt = int(np.argmax(dist))
        selected.append(nxt)
        c0 += labels[nxt] == 0
        c1 += labels[nxt] == 1

    return selected


def diversity_report(df: pl.DataFrame, idx: list[int], feature_cols: list[str]) -> dict:
    """Compute diversity diagnostics on selected rows.

    Evaluates the spread and coverage of selected points in feature space using
    pairwise distances, per-feature statistics, and PCA decomposition.

    Args:
        df: DataFrame containing features.
        idx: Row indices to analyze.
        feature_cols: Column names for features to include.

    Returns:
        Dictionary with diagnostic metrics:
            - min_pairwise_dist: Minimum Euclidean distance between points.
            - mean_pairwise_dist: Mean pairwise distance.
            - median_pairwise_dist: Median pairwise distance.
            - feature_variance: Variance per feature.
            - feature_range: Range (max - min) per feature.
            - pca_explained_var: PCA explained variance per component.
            - pca_explained_ratio: PCA explained variance ratios.

    Note:
        Higher min_pairwise_dist and larger feature ranges indicate better
        diversity. PCA ratios closer to uniform suggest isotropic spread.

    Example:
        >>> report = diversity_report(df, [0, 5, 10], ["f1", "f2", "f3"])
        >>> print(f"Min distance: {report['min_pairwise_dist']:.3f}")
    """
    X = (
        df.filter(pl.col("idx").is_in(idx))
        .select(feature_cols)
        .to_numpy()
        .astype(float)
    )
    dists = pdist(X, metric="euclidean")
    pca = PCA().fit(X)
    return {
        "min_pairwise_dist": float(dists.min()),
        "mean_pairwise_dist": float(dists.mean()),
        "median_pairwise_dist": float(np.median(dists)),
        "feature_variance": dict(
            zip(feature_cols, X.var(axis=0).tolist(), strict=True)
        ),
        "feature_range": dict(
            zip(feature_cols, (X.max(axis=0) - X.min(axis=0)).tolist(), strict=True)
        ),
        "pca_explained_var": pca.explained_variance_.tolist(),
        "pca_explained_ratio": pca.explained_variance_ratio_.tolist(),
    }
