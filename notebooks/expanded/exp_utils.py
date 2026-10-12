

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.loaders import (  # noqa: E402
    load_adjacency_matrix,
    load_agents_data,
    load_run_data,
)

# Final CSVs behind every manuscript figure and appendix table, one file each, prefixed by notebook.
TABLES_DIR = REPO_ROOT / "data" / "analysis" / "aggregated_expanded" / "tables"


def save_table(df, name: str, index: bool = False) -> Path:
    """Write `df` to TABLES_DIR / f"{name}.csv" and return the path."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    print("Saved:", path.relative_to(REPO_ROOT))
    return path

GRAPH_FAMILY_RE = re.compile(r"^(?P<family>.+?)(?:-\d{2})?$")

GRAPH_BIN_EDGES = [0, 8.1, 13.55,  float("inf")]
GRAPH_BIN_LABELS = ["low", "medium", "high"]

# Stored probe label (1 = true, 0 = false, -1 = neither) -> column index in
# belief_ful, whose class order is [true, false, neither].
STANCE_LABEL_TO_IDX = {1: 0, 0: 1, -1: 2}


def graph_bin(value: float) -> str:
    """Bin a graph by average shortest path length: low <value <= medium <value <= high."""
    for hi, lab in zip(GRAPH_BIN_EDGES[1:], GRAPH_BIN_LABELS):
        if value < hi:
            return lab
    return GRAPH_BIN_LABELS[-1]



def graph_family(graph_type: str) -> str:
    """Strip the two-digit realization suffix: ``erdos-renyi-03`` -> ``erdos-renyi``."""
    return GRAPH_FAMILY_RE.match(graph_type).group("family")


def compute_agent_stats(belief_con: np.ndarray, y_true: int) -> dict:
    """
    Compute additional statistics from per-round 3-class probability distributions.
    
    Parameters
    ----------
    belief_con : np.ndarray
        Array of shape (T+1, 3) containing [P(true), P(false), P(neither)] for each round.
    y_true : int
        The true label for the statement (0=True, 1=False, 2=Neither).
    
    Returns
    -------
    dict with keys:
        plasticity_tv     : total-variation plasticity (primary outcome, in [0, 1])
        plasticity_true   : per-class plasticity on P(true)
        plasticity_false  : per-class plasticity on P(false)
        plasticity_neither: per-class plasticity on P(neither)
        plasticity_tv_1   : total-variation plasticity excluding 0th round (primary outcome, in [0, 1])
        net_belief_change : net distance between final and initial distribution
        net_correct_shift : net change in probability of the ground_true label
        monotonicity      : net_change/ (T * plasticity_tv), in [0, 1]
                            close to 1 = monotone movement, close to 0 = oscillation
        T                 : number of transitions used
        total_movement    : total movement across rounds in the probability of the correct label
        progress_ratio    : (forward - backward) / total_movement, in [-1, 1]
                            close to 1 = mostly forward, close to -1 = mostly backward
        agent_accuracy    : probability of the true label at the final round
        agent_error       : probability of the false label at the final round
        progress_forward  : total forward movement in probability of the correct label
        progress_backward : total backward movement in probability of the correct label
        progress_efficiency: net_correct_shift / max_possible_gain, in [0, 1]
                            close to 1 = efficient progress, close to 0 = inefficient progress
                            
    """
    if y_true == -1:
        y_true = 2
    
    P = np.array(belief_con, dtype=float)  # (T+1, 3)
    
    T = P.shape[0] - 1
    diffs = np.abs(np.diff(P, axis=0))         # (T, 3)
    
    plasticity_per_class = diffs.mean(axis=0)  # (3,)
    plasticity_tv = 0.5 * diffs.sum(axis=1).mean()
    plasticity_tv_1 = 0.5 * diffs[1:].sum(axis=1).mean()  # Exclude 0th round
    
    
    net_belief_change = 0.5 * np.abs(P[-1] - P[0]).sum()
    denom = T * plasticity_tv
    monotonicity = float(net_belief_change / denom) if denom > 1e-12 else np.nan
    
    
    net_uncertain =  0.5 * np.abs(P[-1] - P[0]).sum()  # Changes in "neither" probability
    denom_uncertainty = plasticity_per_class[2].mean() * T
    monotonicity_uncertainty = float(net_uncertain / denom_uncertainty)
    
    # Movement in respect to correct label    
    P_correct = P[:, y_true]
    p0, pT = P_correct[0], P_correct[-1]
    net_correct_shift = pT - p0

    eps = 1e-6
    if p0 > 1 - eps or p0 < eps:
       p0 = np.clip(p0, eps, 1 - eps)  # Avoid division by zero or near-zero
    if net_correct_shift >= 0:
        progress_efficiency = net_correct_shift / (1 - p0)   # fraction of upside captured
    else:
        progress_efficiency = net_correct_shift / p0          # fraction of downside realized (negative)
        
    
    diffs_correct = np.diff(P_correct)
    forward = diffs_correct[diffs_correct > 0].sum()    # total movement toward truth
    backward = -diffs_correct[diffs_correct < 0].sum()  # total movement away from truth
    total_movement = forward + backward
    progress_ratio = (forward - backward) / total_movement if total_movement > 0 else 0
    ## Accuracy-like scores
    y_prob = P[T, y_true]  # Probability of the true label across rounds    


    return {
        "plasticity_tv":     float(plasticity_tv),
        "plasticity_true":   float(plasticity_per_class[0]),
        "plasticity_tv_1":   float(plasticity_tv_1),
        "plasticity_false":  float(plasticity_per_class[1]),
        "plasticity_neither":float(plasticity_per_class[2]),
        "net_belief_change": float(net_belief_change),
        "net_correct_shift": float(net_correct_shift),
        "monotonicity":       monotonicity,
        "total_movement":     float(total_movement),
        "progress_ratio":     float(progress_ratio),
        "agent_accuracy":     float(y_prob),
        "agent_error":        float(1 - y_prob),
        "progress_forward": float(forward),
        "progress_backward": float(backward),
        "progress_efficiency": float(progress_efficiency),
        "net_uncertain": float(net_uncertain),
        "monotonicity_uncertainty": float(monotonicity_uncertainty),
        "T":                  T,
    }


def compute_agent_influence(belief_ful: np.ndarray, A: np.ndarray) -> dict:
    """
    Per-agent influence metrics from belief trajectories.

    Notation
    --------
    Δb_i(t) = b_i(t+1) - b_i(t)            belief change vector, t = 0 .. T-1
    tv_i(t) = 0.5 * ||Δb_i(t)||_1          total-variation size of that change, in [0, 1]
    ||Δb_i(t)||                            Euclidean (L2) norm of that change

    The magnitude and contrast families use the TV size; the directional family
    uses L2 norms, because cosine similarity is defined against the L2 metric.

    All metrics pair a *lag* round t with the following *lead* round t+1, so t
    ranges over 0 .. T-2 and the lead index over 1 .. T-1. The first transition
    is therefore never used as a lead, and the last is never used as a lag. This
    also applies to `magnitude_out` / `magnitude_in`, which have no lag term of
    their own; they are restricted to the same lead window so that all metrics
    are averages over the same rounds and can be compared directly.

    Computes three complementary families of influence:

    Magnitude-based:
        magnitude_out[i] = mean over (t, j in N(i)) of tv_j(t+1)
            "How much do my neighbors move?"
        magnitude_in[i]  = mean over t of tv_i(t+1)
            "How much do I move?" (plasticity-like)
        magnitude_weighted[i] = weighted mean over (t, j in N(i)) of tv_j(t+1),
            weighted by the agent's own movement tv_i(t)
            "Do my neighbors move a lot in the rounds right after I moved?"

        magnitude_weighted follows the same lagged-product form as the outgoing
        influence proxy of CoevolveSim Eq. 7, with the lead sum ending at T-1
        (the last round with a recorded next-round belief).

    Directional (activity-weighted cosine similarity between consecutive belief changes):
        directional_out[i] = weighted mean over (t, j in N(i)) of cos(Δb_i(t), Δb_j(t+1)),
            weighted by ||Δb_i(t)|| * ||Δb_j(t+1)||
            "Do my neighbors move in the same direction as I did?"
        directional_in[i]  = weighted mean over (t, j in N(i)) of cos(Δb_j(t), Δb_i(t+1)),
            weighted by ||Δb_j(t)|| * ||Δb_i(t+1)||
            "Do I move in the same direction as my neighbors did?"

        Weighting by movement magnitude downweights low-activity (near-zero Δb) pairs
        whose cosine is noise-dominated, giving a more reliable estimate in settings
        where many agents barely move.

    Pull (movement toward a belief *position*, in TV units per round):
        pull_out[i] = mean over (s, j in N(i)) of TV(b_j(s), b_i(s)) - TV(b_j(s+1), b_i(s))
            "Do my neighbors move toward where I stand?"
        pull_in[i]  = mean over (s, j in N(i)) of TV(b_i(s), b_j(s)) - TV(b_i(s+1), b_j(s))
            "Do I move toward where my neighbors stand?"

        Positive = attraction, negative = repulsion, in [-1, 1]. Unlike directional_*,
        the target is a position rather than a change, so an agent that never moves
        can still pull its neighbors and pull_out stays defined for it.

        Exception to the lead window: pull uses every transition s = 0 .. T-1. It
        has no lag term, and the first transition (the response to the initial
        beliefs) is where most movement happens in low-plasticity settings.

    Contrast-based (neighbors vs. non-neighbors, within the same run and round):
        contrast_out[i] =
            weighted mean over t of [mean_{j in N(i)} tv_j(t+1) - mean_{j not in N(i), j != i} tv_j(t+1)],
            weighted by the agent's own movement tv_i(t)
        contrast_out_unweighted[i] = the same difference, averaged over t with equal weights

        The non-neighbour term is a within-round control: if a round is one where
        the whole network churns, both terms rise and the difference cancels it.
        magnitude_out credits agent i for network-wide dynamics; contrast_out
        credits it only for movement localised to its neighbourhood. The gap
        between contrast_out and contrast_out_unweighted is the part of that
        excess neighbourhood movement that is timed to i moving, rather than a
        static property of whoever i's neighbours happen to be.

        Both are in TV units per round. Positive means neighbours move more than
        non-neighbours; negative means they are more inert. Neither is directional
        (a neighbour moving away from i scores like one moving toward it) and
        neither is causal - degree, clustering and homophily all confound it. The
        non-neighbour baseline is estimated from N - |N(i)| - 1 agents, so it
        becomes noisy at high degree and is undefined on a complete graph.

    Parameters
    ----------
    belief_ful : np.ndarray, shape (N, T+1, 3)
        Per-agent belief trajectories. Last dim is [P(true), P(false), P(neither)].
    A : np.ndarray, shape (N, N)
        Adjacency matrix. Only the support is used: j is a neighbour of i iff
        A[i, j] > 0, so edge weights are ignored. Self-loops are dropped.

    Returns
    -------
    dict with keys (all arrays of shape (N,)):
        magnitude_out           magnitude of neighbors' updates
        magnitude_in            magnitude of the agent's own updates
        magnitude_weighted      own-movement-weighted magnitude of neighbors' updates
        directional_out         activity-weighted directional alignment, neighbors with self
        directional_in          activity-weighted directional alignment, self with neighbors
        contrast_out            neighbors - non-neighbors, own-movement-weighted
        contrast_out_unweighted neighbors - non-neighbors, unweighted
        pull_out                neighbors' movement toward the agent's belief
        pull_in                 the agent's movement toward its neighbors' beliefs
        n_pairs                number of valid (j, t) pairs per agent (diagnostic, int)
        effective_n_out         effective sample size for directional_out
                                = (sum of weights)^2 / sum of squared weights (Kish formula)
        effective_n_in          effective sample size for directional_in

    Undefined entries are np.nan rather than 0.0, since 0.0 is a meaningful value
    for every one of these metrics. An entry is nan when:
      - the agent is isolated (no neighbours): every metric except magnitude_in,
        which is a self-only quantity and stays defined
      - the agent never moves, so the weights sum to zero: magnitude_weighted,
        contrast_out, and (jointly with the neighbours' movement) directional_*
      - the graph is complete for that agent, leaving no non-neighbours:
        contrast_out, contrast_out_unweighted
    n_pairs and effective_n_* stay numeric (0) in those cases; they are counts.
    """
    N, Tp1, _ = belief_ful.shape
    T = Tp1 - 1

    if T < 2:
        raise ValueError(f"Need at least 2 transitions for influence; got T={T}.")

    delta = np.diff(belief_ful, axis=1)              # (N, T, 3)
    norms = np.linalg.norm(delta, axis=2)            # (N, T)
    tv_mag = 0.5 * np.abs(delta).sum(axis=2)         # (N, T) per-round TV distance
    eps = 1e-12

    out_mag = np.full(N, np.nan)
    in_mag = np.full(N, np.nan)
    out_mag_weighted = np.full(N, np.nan)
    out_dir = np.full(N, np.nan)
    in_dir = np.full(N, np.nan)
    out_contrast = np.full(N, np.nan)
    out_contrast_unw = np.full(N, np.nan)
    out_pull = np.full(N, np.nan)
    in_pull = np.full(N, np.nan)
    n_pairs = np.zeros(N, dtype=int)
    eff_n_out = np.zeros(N)
    eff_n_in = np.zeros(N)

    # Total movement per lead round, over *all* agents (for the non-neighbor mean).
    tv_lead_total = tv_mag[:, 1:].sum(axis=0)        # (T-1,)

    for i in range(N):
        # Self-only, so it stays defined even for an isolated agent.
        in_mag[i] = tv_mag[i, 1:].mean()

        neighbors = np.where(A[i] > 0)[0]
        neighbors = neighbors[neighbors != i]        # self-loops are not neighbours
        k = len(neighbors)
        if k == 0:
            continue

        n_pairs[i] = k * (T - 1)

        # --- Lag / lead slices -------------------------------------------------
        di_lag  = delta[i, :-1]                      # (T-1, 3)
        di_lead = delta[i, 1:]                       # (T-1, 3)
        ni_lag  = norms[i, :-1]                      # (T-1,)
        ni_lead = norms[i, 1:]                       # (T-1,)

        dj_lag  = delta[neighbors, :-1]              # (k, T-1, 3)
        dj_lead = delta[neighbors, 1:]               # (k, T-1, 3)
        nj_lag  = norms[neighbors, :-1]              # (k, T-1)
        nj_lead = norms[neighbors, 1:]               # (k, T-1)

        tv_i_lag  = tv_mag[i, :-1]                   # (T-1,)
        tv_i_lead = tv_mag[i, 1:]                    # (T-1,)
        tv_j_lead = tv_mag[neighbors, 1:]            # (k, T-1)

        # --- Magnitude family --------------------------------------------------
        out_mag[i] = tv_j_lead.mean()
        in_mag[i] = tv_i_lead.mean()

        # Weighted mean of tv_j(t+1) with weight tv_i(t). The weight does not
        # depend on j, so the weight total is k * sum_t tv_i(t).
        own_movement = tv_i_lag.sum()
        if own_movement > eps:
            out_mag_weighted[i] = (tv_i_lag[None] * tv_j_lead).sum() / (own_movement * k)

        # --- Directional family: activity-weighted cosine ----------------------
        # A weighted mean of cosines with weight ||a||*||b|| collapses to
        # sum(dot) / sum(weight), because cos * weight = dot.

        # out: cos(Δb_i(t), Δb_j(t+1)), weighted by ||Δb_i(t)|| * ||Δb_j(t+1)||
        dots_out = (di_lag[None] * dj_lead).sum(axis=2)         # (k, T-1)
        weights_out = ni_lag[None] * nj_lead                    # (k, T-1)
        total_weight_out = weights_out.sum()
        if total_weight_out > eps:
            out_dir[i] = dots_out.sum() / total_weight_out
            # Kish effective sample size
            sumsq_out = (weights_out ** 2).sum()
            if sumsq_out > eps:
                eff_n_out[i] = (total_weight_out ** 2) / sumsq_out

        # in: cos(Δb_j(t), Δb_i(t+1)), weighted by ||Δb_j(t)|| * ||Δb_i(t+1)||
        dots_in = (dj_lag * di_lead[None]).sum(axis=2)          # (k, T-1)
        weights_in = nj_lag * ni_lead[None]                     # (k, T-1)
        total_weight_in = weights_in.sum()
        if total_weight_in > eps:
            in_dir[i] = dots_in.sum() / total_weight_in
            sumsq_in = (weights_in ** 2).sum()
            if sumsq_in > eps:
                eff_n_in[i] = (total_weight_in ** 2) / sumsq_in

        # --- Contrast family: neighbors vs. non-neighbors, same run and round ---
        n_non = N - k - 1                                       # exclude self and neighbors
        if n_non > 0:
            nb_mean = tv_j_lead.mean(axis=0)                    # (T-1,)
            non_mean = (tv_lead_total - tv_j_lead.sum(axis=0) - tv_i_lead) / n_non
            diff = nb_mean - non_mean                           # (T-1,)
            out_contrast_unw[i] = diff.mean()
            if own_movement > eps:
                out_contrast[i] = float((tv_i_lag * diff).sum() / own_movement)

        # --- Pull family: distance to a fixed target, before vs. after the update ---
        b_i_prev, b_i_next = belief_ful[i, :-1], belief_ful[i, 1:]              # (T, 3)
        b_j_prev, b_j_next = belief_ful[neighbors, :-1], belief_ful[neighbors, 1:]  # (k, T, 3)
        # out: target is i at s, mover is j
        out_pull[i] = (0.5 * np.abs(b_j_prev - b_i_prev[None]).sum(axis=2)
                       - 0.5 * np.abs(b_j_next - b_i_prev[None]).sum(axis=2)).mean()
        # in: target is j at s, mover is i
        in_pull[i] = (0.5 * np.abs(b_i_prev[None] - b_j_prev).sum(axis=2)
                      - 0.5 * np.abs(b_i_next[None] - b_j_prev).sum(axis=2)).mean()

        out_dir = np.nan_to_num(out_dir)
        in_dir = np.nan_to_num(in_dir)

    return {
        "magnitude_out":           out_mag,
        "magnitude_in":            in_mag,
        "magnitude_weighted":      out_mag_weighted,
        "directional_out":         out_dir,
        "directional_in":          in_dir,
        "contrast_out":            out_contrast,
        "contrast_out_unweighted": out_contrast_unw,
        "pull_out":                out_pull,
        "pull_in":                 in_pull,
        "n_pairs":                n_pairs,
        "effective_n_out":         eff_n_out,
        "effective_n_in":          eff_n_in,
    }


def directional_projection(belief_ful: np.ndarray, A: np.ndarray,
                           residualize: bool = False, eps: float = 1e-12) -> dict[str, np.ndarray]:
    """
    Outgoing directional influence as a projection, with a same-run non-neighbour placebo.

    With Δb_i(t) = b_i(t+1) - b_i(t), per agent i:

        proj_out[i]      = sum_t mean_{j in N(i)} <Δb_i(t), Δb_j(t+1)> / sum_t ||Δb_i(t)||
        proj_non[i]      = same, averaged over non-neighbours (excluding i)
        proj_contrast[i] = proj_out[i] - proj_non[i]

    `proj_out` is a weighted mean, over rounds, of how far i's neighbours move along
    i's direction in the next round, per unit of i's own move (weights ||Δb_i(t)||).
    It is in belief units, not a cosine. Non-neighbours cannot see i at lag 1, so
    `proj_non` is the common drift and `proj_contrast` is the neighbour-specific part.

    Undefined (NaN) when i never moves, or has no neighbours / non-neighbours. An agent
    that moves but whose neighbours never respond gets 0.

    `residualize=True` subtracts a pooled lag-1 autocorrelation rho from every lead
    move (lead - rho * lag). With rho < 0 this adds back j's same-round move, which
    re-introduces contemporaneous co-movement, so it is off by default.
    """
    P = np.asarray(belief_ful, dtype=float)                  # (N, T+1, 3)
    N, Tp1, _ = P.shape
    if Tp1 - 1 < 2:
        raise ValueError("Need at least 2 transitions.")
    D = np.diff(P, axis=1)                                   # (N, T, 3)
    lag, lead = D[:, :-1], D[:, 1:]                          # (N, T-1, 3)

    rho = np.nan
    if residualize:
        rho = float((lag * lead).sum() / max((lag ** 2).sum(), eps))
        lead = lead - rho * lag

    NB = (np.asarray(A) > 0).astype(float)
    np.fill_diagonal(NB, 0.0)
    NN = 1.0 - NB - np.eye(N)                                # non-neighbours, excluding self
    k, n_non = NB.sum(1), NN.sum(1)

    dots = np.einsum("itc,jtc->ij", lag, lead)               # sum_t <Δb_i(t), Δb_j(t+1)>
    w = np.linalg.norm(lag, axis=2).sum(1)                   # sum_t ||Δb_i(t)||

    with np.errstate(invalid="ignore", divide="ignore"):
        nb_term = (NB * dots).sum(1) / k
        non_term = (NN * dots).sum(1) / n_non
        moves = w > eps
        proj_out = np.where(moves & (k > 0), nb_term / w, np.nan)
        proj_non = np.where(moves & (n_non > 0), non_term / w, np.nan)
        proj_contrast = np.where(moves & (k > 0) & (n_non > 0), (nb_term - non_term) / w, np.nan)
    return {"proj_out": proj_out, "proj_non": proj_non, "proj_contrast": proj_contrast,
            "own_weight": w, "rho": np.full(N, rho)}


def leadlag_asymmetry(belief_ful: np.ndarray, A: np.ndarray, eps: float = 1e-12) -> dict[str, np.ndarray]:
    """
    Lead-lag asymmetry (Nagy et al. 2010; edge-reversal logic of Anagnostopoulos et al. 2008).

        C[i, j] = sum_{t=0}^{T-2} <Δb_i(t), Δb_j(t+1)>          (i moves, then j)
        W[i, j] = sum_{t=0}^{T-2} ||Δb_i(t)|| * ||Δb_j(t+1)||
        pair_net[i, j] = C[i, j] - C[j, i] on edges, 0 elsewhere  (antisymmetric)

        net_dot[i] = sum_{j in N(i)} pair_net[i, j] / (k_i * (T-1))                   L2^2 per round
        net_cos[i] = sum_{j in N(i)} pair_net[i, j] / sum_{j in N(i)} (W[i, j] + W[j, i])   in [-1, 1]

    Positive = i's moves tend to precede aligned moves by its neighbours more than the
    reverse. Zero-sum across each edge, so it ranks who moves first relative to whom;
    common drift cancels. `net_cos` is primary: `net_dot` scales with plasticity^2.
    NaN (never 0) when i is isolated, never moves, or (net_cos) the weights sum to 0.
    """
    P = np.asarray(belief_ful, dtype=float)
    if P.ndim != 3 or P.shape[2] != 3:
        raise ValueError(f"belief_ful must be (N, T+1, 3); got {P.shape}")
    N, Tp1, _ = P.shape
    T = Tp1 - 1
    if T < 2:
        raise ValueError(f"Need at least 2 transitions; got T={T}.")

    NB = (np.asarray(A) > 0).astype(float)
    np.fill_diagonal(NB, 0.0)
    if NB.shape != (N, N) or not (NB == NB.T).all():
        raise ValueError("A must be (N, N) with symmetric support.")
    k = NB.sum(1)

    D = np.diff(P, axis=1)                          # (N, T, 3)
    nrm = np.linalg.norm(D, axis=2)                 # (N, T)
    C = np.einsum("itc,jtc->ij", D[:, :-1], D[:, 1:])
    W = nrm[:, :-1] @ nrm[:, 1:].T

    pair_net = (C - C.T) * NB
    pair_w = (W + W.T) * NB
    num, den = pair_net.sum(1), pair_w.sum(1)
    own_move = nrm.sum(1)

    ok = (k > 0) & (own_move > eps)
    with np.errstate(invalid="ignore", divide="ignore"):
        net_dot = np.where(ok, num / (k * (T - 1)), np.nan)
        net_cos = np.where(ok & (den > eps), num / den, np.nan)
    return {"net_dot": net_dot, "net_cos": net_cos, "pair_net": pair_net,
            "n_neighbors": k.astype(int), "own_move": own_move}


def rewire_degree_preserving(A: np.ndarray, rng: np.random.Generator,
                             n_swaps_per_edge: int = 10,
                             max_attempts_per_swap: int = 20) -> np.ndarray:
    """
    Degree-preserving null graph via double-edge swaps (configuration-model rewiring).

    Repeatedly picks two edges (a, b) and (c, d) and replaces them with (a, d) and
    (c, b), rejecting proposals that would create a self-loop or a multi-edge. Every
    node keeps its exact degree, so any gap between observed and null influence
    cannot be attributed to degree.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Binary symmetric adjacency matrix.
    rng : np.random.Generator
    n_swaps_per_edge : int
        Target number of successful swaps = n_swaps_per_edge * |E|.
    max_attempts_per_swap : int
        Give up after this many rejected proposals per targeted swap.

    Returns
    -------
    np.ndarray, shape (N, N) : rewired binary symmetric adjacency matrix.
    """
    iu, ju = np.nonzero(np.triu(A, 1))
    edges = np.stack([iu, ju], axis=1)
    m = len(edges)
    if m < 2:
        return A.copy()

    B = A.copy()
    target = n_swaps_per_edge * m
    max_attempts = target * max_attempts_per_swap
    done = attempts = 0

    while done < target and attempts < max_attempts:
        attempts += 1
        e1, e2 = rng.integers(0, m, size=2)
        if e1 == e2:
            continue
        a, b = edges[e1]
        c, d = edges[e2]
        if rng.random() < 0.5:              # randomise which endpoints get paired
            c, d = d, c
        if len({a, b, c, d}) < 4:           # would create a self-loop
            continue
        if B[a, d] or B[c, b]:              # would create a multi-edge
            continue
        B[a, b] = B[b, a] = 0
        B[c, d] = B[d, c] = 0
        B[a, d] = B[d, a] = 1
        B[c, b] = B[b, c] = 1
        edges[e1] = (a, d)
        edges[e2] = (c, b)
        done += 1

    return B


def build_null_graphs(A: np.ndarray, seed: int, n_null: int = 20) -> list[np.ndarray]:
    """
    Draw `n_null` degree-preserving rewirings of one graph.

    Split out from the per-agent computation because the null graphs depend only on
    `A` and the seed, never on the beliefs. Since `graph_id` is `f"{graph_type}-{seed}"`,
    every run sharing a graph also shares the seed, so building these once per
    `graph_id` and reusing them across that graph's runs is equivalent to
    redrawing them per run (the RNG stream is identical) at a fraction of the cost.
    """
    rng = np.random.default_rng(seed)
    return [rewire_degree_preserving(A, rng) for _ in range(n_null)]


def null_directional_out(belief_ful: np.ndarray,
                         null_graphs: list[np.ndarray]) -> np.ndarray:
    """
    Expected `directional_out` under degree-preserving rewiring.

    Recomputes the directional influence on randomised neighbourhoods (the same
    agents and the same belief trajectories, but a shuffled wiring) and averages
    over the replicates. Because the rewiring preserves every node's degree, the gap
    between the observed value and this baseline cannot be explained by degree; it is
    attributable to *which* agents an agent is connected to.

    Aggregated with `nanmean`: `directional_out` is undefined when the activity weight
    ||db_i(t)|| * ||db_j(t+1)|| sums to zero, and because that weight depends on which
    neighbours were drawn, a replicate can be undefined while its siblings are fine.
    A plain mean would let one such replicate void the whole baseline.

    Parameters
    ----------
    belief_ful : np.ndarray, shape (N, T+1, 3)
    null_graphs : list of (N, N) adjacency matrices from `build_null_graphs`.

    Returns
    -------
    np.ndarray, shape (N,) : per-agent mean of `directional_out` across replicates.
        NaN for an agent undefined in every replicate.
    """
    return null_influence(belief_ful, null_graphs, keys=("directional_out",))["directional_out"]


def null_influence(belief_ful: np.ndarray,
                   null_graphs: list[np.ndarray],
                   keys: tuple[str, ...] = ("directional_out", "pull_out")) -> dict:
    """
    Expected value of several `compute_agent_influence` metrics under degree-preserving
    rewiring, from one pass over the null graphs.

    Same construction and `nanmean` aggregation as `null_directional_out`; this just
    avoids recomputing the influence once per metric.

    Returns
    -------
    dict : key -> np.ndarray of shape (N,), the per-agent mean across replicates.
    """
    draws = {key: np.full((len(null_graphs), belief_ful.shape[0]), np.nan) for key in keys}
    for r, A_null in enumerate(null_graphs):
        influence = compute_agent_influence(belief_ful, A_null)
        for key in keys:
            draws[key][r] = influence[key]

    with warnings.catch_warnings():
        # An agent undefined in every replicate yields an all-NaN slice.
        warnings.simplefilter("ignore", RuntimeWarning)
        return {key: np.nanmean(d, axis=0) for key, d in draws.items()}


def compute_run_stats(belief_ful: np.ndarray, ground_truth_idx) -> dict:
    """
    Compute consensus metrics across agents within a single run.
    
    Parameters
    ----------
    belief_ful : np.ndarray, shape (N, T+1, 3)
        Per-agent belief trajectories. Last dim is [P(true), P(false), P(neither)].
    ground_truth_idx : int
        Index of the ground truth label (0=true, 1=false, 2=neither).
    Returns
    -------
    dict with keys:
        modal_consensus_T       : fraction of agents agreeing with modal label at round T, [1/3, 1]
        modal_consensus_0       : same, at round 0
        modal_consensus_mean    : mean of modal_consensus across all rounds
        polarization_T          : mean pairwise TV distance between agent distributions at T, [0, 1]
                                  high = polarized, low = consensus
        polarization_0          : same, at round 0
        polarization_mean       : mean across rounds
        entropy_T               : Shannon entropy of population label distribution at T (3-class), [0, log(3)]
                                  0 = perfect consensus, log(3) ≈ 1.10 = uniform
        entropy_0               : same, at round 0
        entropy_change          : entropy_T - entropy_0 (positive = more diverse, negative = converged)
        n_distinct_labels_T     : number of distinct modal labels at T (1, 2, or 3)
        majority_label_T        : modal label at round T (0=true, 1=false, 2=neither)
        majority_share_T        : same as modal_consensus_T (alias)
    """
    P = np.asarray(belief_ful, dtype=float)  # (N, T+1, 3)
    N, Tp1, K = P.shape
    T = Tp1 - 1
    
    # Discrete labels per agent per round
    labels = P.argmax(axis=2)  # (N, T+1)
    
    modal_consensus = np.zeros(Tp1)
        
    for t in range(Tp1):
        counts = np.bincount(labels[:, t], minlength=K)
        modal_consensus[t] = counts.max() / N    
    # --- Polarization: mean pairwise TV distance between agents ---
    def mean_pairwise_tv(P_t):
        """P_t: (N, K). Returns scalar."""
        # TV(p, q) = 0.5 * ||p - q||_1
        diffs = np.abs(P_t[:, None, :] - P_t[None, :, :]).sum(axis=2) * 0.5  # (N, N)
        # Exclude diagonal (self-distance = 0)
        n = P_t.shape[0]
        return diffs.sum() / (n * (n - 1))
    
    polarization_T = mean_pairwise_tv(P[:, T])
    polarization_0 = mean_pairwise_tv(P[:, 0])
    polarization_per_round = np.array([mean_pairwise_tv(P[:, t]) for t in range(Tp1)])
    
    accuracy_0 = P[:, 0, ground_truth_idx].mean()
    accuracy_T = P[:, T, ground_truth_idx].mean()
    diff_accuracy = accuracy_T - accuracy_0
    
    # --- Population-level entropy of discrete labels ---
    def label_entropy(labels_t):
        counts = np.bincount(labels_t, minlength=K)
        p = counts / counts.sum()
        # Avoid log(0)
        p_nonzero = p[p > 0]
        return -np.sum(p_nonzero * np.log(p_nonzero))
    
    entropy_T = label_entropy(labels[:, T])
    entropy_0 = label_entropy(labels[:, 0])
    
    # --- Modal label and distinct labels at T ---
    counts_T = np.bincount(labels[:, T], minlength=K)
    majority_label_T = int(counts_T.argmax())
    n_distinct_T = int((counts_T > 0).sum())
    
    return {
        "modal_consensus_T":     float(modal_consensus[T]),
        "modal_consensus_0":     float(modal_consensus[0]),
        "modal_consensus_mean":  float(modal_consensus.mean()),
        "polarization_T":        float(polarization_T),
        "polarization_0":        float(polarization_0),
        "polarization_1":        float(polarization_per_round[1]),
        "polarization_mean":     float(polarization_per_round.mean()),
        "entropy_T":             float(entropy_T),
        "entropy_0":             float(entropy_0),
        "entropy_change":        float(entropy_T - entropy_0),
        "n_distinct_labels_T":   n_distinct_T,
        "majority_label_T":      majority_label_T,
        "majority_share_T":      float(modal_consensus[T]),
        "accuracy_0":            float(accuracy_0),
        "accuracy_T":            float(accuracy_T),
        "diff_accuracy":         float(diff_accuracy),
        "T":                     T,
    }