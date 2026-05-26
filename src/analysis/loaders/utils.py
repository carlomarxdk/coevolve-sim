"""Methods that have nothing to do with the actual loading of data, but are used by the loaders.
Gaussian Transfer Entropy for CoevolveSim
==========================================

Computes directed information transfer between LLM agents using
the Gaussian (linear) Transfer Entropy estimator, which is equivalent
to linear Granger causality (Barnett, Barrett & Seth, 2009).

    TE(i → j) = 0.5 * ln( σ²_restricted / σ²_unrestricted )

where:
    Restricted model:   Y_t ~ Y_{t-1}
    Unrestricted model: Y_t ~ Y_{t-1} + X_{t-1}

Convention:
    TE_matrix[i, j] = transfer entropy FROM agent i TO agent j
    Row = source (influencer), Column = target (influenced)
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Literal
from collections import defaultdict

NormMode = Literal["none", "out", "in", "both", "max", "log"]



def build_adjacency_matrix(edges: list[tuple[int, int]], n_agents: int) -> np.ndarray:
    """Build adjacency matrix from edge list.

    Args:
        edges: List of undirected edges as (i, j) tuples.
        n_agents: Number of agents in the network.

    Returns:
        Symmetric adjacency matrix of shape (n_agents, n_agents).
    """
    adj= np.zeros((n_agents, n_agents), dtype=int)
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1
    return adj

def normalize_te_matrix(
    te_matrix: np.ndarray,
    adjacency: np.ndarray,
    mode: NormMode = "none",
) -> np.ndarray:
    """Normalize a TE matrix by degree or global maximum.

    All normalization modes preserve NaN on the diagonal and 0.0 on
    non-edges so that downstream nansum / nanmean behave correctly.

    Args:
        te_matrix: Raw TE matrix of shape (n, n). Diagonal = NaN,
            non-edges = 0.0 (as produced by compute_te_matrix).
        adjacency: Binary adjacency matrix of shape (n, n).
        mode: Normalization strategy:
            - "none"  : return te_matrix unchanged.
            - "out"   : divide row i by out-degree[i].
                        nansum over axis=1 then gives per-neighbor mean TE.
            - "in"    : divide col j by in-degree[j].
                        nansum over axis=0 then gives per-neighbor mean TE.
            - "both"  : divide entry (i,j) by sqrt(out_degree[i] * in_degree[j]).
                        Symmetric degree correction; best for role-pair matrices.
            - "max"   : divide all entries by the global nanmax.
                        Scales to [0, 1]; useful for cross-run comparisons.

    Returns:
        Normalized TE matrix of the same shape. Non-edges remain 0.0,
        diagonal remains NaN.
    """
    if mode == "none":
        return te_matrix.copy()

    te_norm = te_matrix.copy()

    # Identify structural zeros (non-edges) and diagonal to restore later
    non_edges = adjacency == 0
    n = te_matrix.shape[0]

    if mode == "out":
        out_degree = adjacency.sum(axis=1).clip(min=1).astype(float)  # (n,)
        te_norm = te_norm / out_degree[:, np.newaxis]

    elif mode == "in":
        in_degree = adjacency.sum(axis=0).clip(min=1).astype(float)  # (n,)
        te_norm = te_norm / in_degree[np.newaxis, :]

    elif mode == "both":
        out_degree = adjacency.sum(axis=1).clip(min=1).astype(float)
        in_degree = adjacency.sum(axis=0).clip(min=1).astype(float)
        denom = np.sqrt(np.outer(out_degree, in_degree))  # (n, n)
        te_norm = te_norm / denom

    elif mode == "max":
        max_val = float(np.nanmax(te_norm))
        if max_val > 0:
            te_norm = te_norm / max_val

    else:
        raise ValueError(
            f"Unknown normalization mode '{mode}'. "
            "Choose from: 'none', 'out', 'in', 'both', 'max'."
        )

    # Restore structure: non-edges → 0.0, diagonal → NaN
    te_norm[non_edges] = 0.0
    np.fill_diagonal(te_norm, np.nan)

    return te_norm



def gaussian_te_f_test(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.05
) -> tuple[float, float, float, bool]:
    """Compute Gaussian TE with an F-test for significance (Granger causality test).

    The F-statistic tests whether X_{t-1} significantly improves the prediction
    of Y_t beyond what Y_{t-1} alone provides.

    Args:
        x: Source agent belief time series, shape (T,).
        y: Target agent belief time series, shape (T,).
        alpha: Significance level for the F-test.

    Returns:
        Tuple of (te_value, f_stat, p_value, significant) where:
            - te_value: Transfer entropy in nats.
            - f_stat: F-statistic.
            - p_value: p-value from F-distribution.
            - significant: Whether p_value < alpha.

    Notes:
        With T=11 (10 transitions), df_resid = 10 - 3 = 7. This gives LOW
        statistical power. A non-significant result does not imply absence of
        influence.
    """
    T = len(y)
    assert len(x) == T

    y_t = y[1:]
    y_lag = y[:-1]
    x_lag = x[:-1]
    n = len(y_t)

    # Edge case: constant target
    if np.var(y_lag) == 0:
        return (0.0, 0.0, 1.0, False)

    # Restricted model
    X_r = np.column_stack([np.ones(n), y_lag])
    beta_r = np.linalg.lstsq(X_r, y_t, rcond=None)[0]
    resid_r = y_t - X_r @ beta_r
    rss_r = np.sum(resid_r**2)

    # Unrestricted model
    X_u = np.column_stack([np.ones(n), y_lag, x_lag])
    beta_u = np.linalg.lstsq(X_u, y_t, rcond=None)[0]
    resid_u = y_t - X_u @ beta_u
    rss_u = np.sum(resid_u**2)

    # TE
    var_r = np.mean(resid_r**2)
    var_u = np.mean(resid_u**2)

    if var_r <= 0 or var_u <= 0:
        return (0.0, 0.0, 1.0, False)

    te = max(0.5 * np.log(var_r / var_u), 0.0)

    # F-test
    k = 1  # number of additional predictors
    df_resid = n - 3  # unrestricted model: intercept + y_lag + x_lag

    if df_resid <= 0 or rss_u <= 0:
        return (te, 0.0, 1.0, False)

    f_stat = ((rss_r - rss_u) / k) / (rss_u / df_resid)
    f_stat = max(f_stat, 0.0)  # numerical safety

    p_value = float(1.0 - stats.f.cdf(f_stat, k, df_resid))

    return (te, f_stat, p_value, p_value < alpha)


def gaussian_te_pairwise(
    x: np.ndarray, 
    y: np.ndarray, 
    bias_corrected: bool = False
) -> float:
    """Compute Gaussian Transfer Entropy from time series x (source) to y (target).

    Args:
        x: Belief score time series of the SOURCE agent, shape (T,).
        y: Belief score time series of the TARGET agent, shape (T,).
        bias_corrected: If True, use Akaike-corrected variance (RSS / (n - k))
            instead of population variance (RSS / n). Partially corrects
            small-sample bias.

    Returns:
        Transfer entropy in nats, clipped to min 0.
    """
    T = len(y)
    assert len(x) == T, "x and y must have the same length"
    assert T >= 4, "Need at least 4 time points (3 transitions) for TE estimation"

    # Lagged variables: n = T-1 observations
    y_t = y[1:]  # target at time t
    y_lag = y[:-1]  # target at time t-1
    x_lag = x[:-1]  # source at time t-1
    n = len(y_t)

    # --- Edge case: constant target series ---
    if np.var(y_lag) == 0:
        return 0.0

    # --- Restricted model: Y_t ~ intercept + Y_{t-1} ---
    X_r = np.column_stack([np.ones(n), y_lag])
    beta_r, _, _, _ = np.linalg.lstsq(X_r, y_t, rcond=None)
    resid_r = y_t - X_r @ beta_r

    if bias_corrected:
        # k_r = 2 parameters (intercept + y_lag)
        var_restricted = np.sum(resid_r**2) / (n - 2)
    else:
        var_restricted = np.mean(resid_r**2)

    # --- Unrestricted model: Y_t ~ intercept + Y_{t-1} + X_{t-1} ---
    X_u = np.column_stack([np.ones(n), y_lag, x_lag])
    beta_u, _, _, _ = np.linalg.lstsq(X_u, y_t, rcond=None)
    resid_u = y_t - X_u @ beta_u

    if bias_corrected:
        # k_u = 3 parameters (intercept + y_lag + x_lag)
        var_unrestricted = np.sum(resid_u**2) / (n - 3)
    else:
        var_unrestricted = np.mean(resid_u**2)

    # --- Degenerate cases ---
    if var_restricted <= 0 or var_unrestricted <= 0:
        return 0.0

    # Clip the variance ratio to avoid numerical blow-up when the
    # unrestricted model fits near-perfectly (e.g., exact lagged copy).
    ratio = var_restricted / var_unrestricted
    ratio = min(ratio, 1e10)  # cap at a very large but finite value

    te = 0.5 * np.log(ratio)
    return max(te, 0.0)


def compute_te_matrix(
    agents_data: dict[str, dict],
    network_edges: list[list[int]],
    n_agents: int,
    bias_corrected: bool = False,
    norm: NormMode = "none",
    start_t: int = 0,
) -> np.ndarray:
    """Compute the full NxN Gaussian TE matrix.

    Args:
        agents_data: Keyed by agent_id (str "0".."47"), each value has:
            'belief_scores': dict keyed by round (str "0".."10"), float values,
            'role': str,
            'model': str.
        network_edges: Undirected edges as list of [int, int].
        n_agents: Number of agents (e.g. 48).
        bias_corrected: Use small-sample corrected variance estimator.
        norm: Normalization mode for TE values. Options are:
            - "none": No normalization.
            - "out": Normalize by out-degree.
            - "in": Normalize by in-degree.
            - "both": Normalize by both in- and out-degree.
            - "max": Normalize by the maximum TE value.

    Returns:
        TE_matrix with shape (n_agents, n_agents) where TE_matrix[i, j] = TE
        from agent i to agent j. Diagonal = NaN. Non-adjacent pairs = 0.0
        (or NaN if not restricted).
    """
    # Build score matrix: (n_agents, T)
    assert start_t >=0, "start_t must be non-negative"
    score_matrix = agents_data["belief_con"][:, start_t:]  # shape (n_agents, T)

    # Build adjacency set
    adj: set[tuple[int, int]] = set()
    for edge in network_edges:
        u, v = int(edge[0]), int(edge[1])
        adj.add((u, v))
        adj.add((v, u))

    adjacency = build_adjacency_matrix(
        [(u, v) for u, v in adj if u < v], n_agents
    )

    # Compute TE matrix
    te_matrix = np.full((n_agents, n_agents), np.nan)

    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue  # diagonal stays NaN

            elif (i, j) not in adj:
                te_matrix[i, j] = 0.0  # non-adjacent pairs set to 0
                continue

            te_matrix[i, j] = gaussian_te_pairwise(
                score_matrix[i], score_matrix[j], bias_corrected=bias_corrected
            )

    # Apply normalization
    if norm != "none":
        te_matrix = normalize_te_matrix(te_matrix, adjacency, mode=norm)

    return te_matrix


def compute_te_matrix_by_group(
    agents_data: dict,
    network_edges: list[list[int]],
    group_to_idx: dict[str, list[int]],
    n_agents: int,
    bias_corrected: bool = False,
    norm: NormMode = "none",
    start_t: int = 0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute group-aggregated TE matrix. Just a wrapper around compute_te_matrix_by_role."""
    return compute_te_matrix_by_role(agents_data, network_edges, group_to_idx, n_agents, bias_corrected, norm, start_t)

def compute_te_matrix_by_role(
    agents_data: dict,
    network_edges: list[list[int]],
    role_to_idx: dict[str, list[int]],
    n_agents: int,
    bias_corrected: bool = False,
    norm: NormMode = "none",
    start_t: int = 0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute role-aggregated TE matrix.

    Args:
        agents_data: Agent data dictionary.
        network_edges: Undirected edges as list of [int, int].
        role_to_idx: Mapping from role name to list of agent indices.
        n_agents: Number of agents.
        bias_corrected: Use small-sample corrected variance estimator.

    Returns:
        Tuple of (TE_role_matrix, role_to_ridx) where:
            - TE_role_matrix: shape (n_roles, n_roles), mean TE from role_i
              agents to role_j agents.
            - role_to_ridx: dict mapping role name to role matrix index.
    """
    te_matrix = compute_te_matrix(
        agents_data, network_edges, n_agents, bias_corrected=bias_corrected, norm=norm, start_t=start_t
    )

    roles = sorted(role_to_idx.keys())
    role_to_ridx = {role: ridx for ridx, role in enumerate(roles)}
    n_roles = len(roles)

    te_role_matrix = np.full((n_roles, n_roles), np.nan)

    for ri, role_i in enumerate(roles):
        for rj, role_j in enumerate(roles):
            agents_i = role_to_idx[role_i]
            agents_j = role_to_idx[role_j]

            values = []
            for ai in agents_i:
                for aj in agents_j:
                    if ai == aj:
                        continue  # only skip self-pairs, not same-role pairs
                    val = te_matrix[ai, aj]
                    if not np.isnan(val):
                        values.append(val)

            if values:
                te_role_matrix[ri, rj] = np.mean(values)
            # else stays NaN — no adjacent pairs between these roles in this run

    return te_role_matrix, role_to_ridx



def alignment(b_i: np.ndarray, b_j: np.ndarray) -> float:
    """
    b_i, b_j: belief score time series, shape (T,)
    Returns Pearson correlation between the pull i exerts 
    and the update j makes.
    """
    pull   = b_i[:-1] - b_j[:-1]   # b_i(t) - b_j(t)
    update = b_j[1:]  - b_j[:-1]   # b_j(t+1) - b_j(t)
    if np.std(pull) == 0 or np.std(update) == 0:
        return 0.0
    return np.corrcoef(pull, update, )[0, 1]