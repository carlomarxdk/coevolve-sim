"""Network graph selection using structural diversity criteria."""

import numpy as np
import polars as pl
from diversity import maximin_select, maximin_select_stratified
from sklearn.preprocessing import StandardScaler

from src.core.metrics.network import (
    connected_components,
    global_clustering_coefficient,
    mean_shortest_path,
    shortest_path_lengths,
)
from src.core.network import Network

#: Structural features used by the maximin criterion (order matters).
FEATURE_COLS = ["mean_sp", "gcc", "mean_deg", "std_deg", "node0_deg"]

#: Generator-specific parameters that a realization may vary.
PARAM_COLS = {"ER": ["p"], "WS": ["k", "beta"]}


def graph2vec(G: Network) -> tuple[np.ndarray, int]:
    """Convert Network to feature vector representation.

    Extracts structural properties for diversity-based selection:
    1. Mean shortest path length
    2. Global clustering coefficient
    3. Mean degree
    4. Standard deviation of degrees
    5. Degree of node 0 (reserved for special agent placement)

    Args:
        G: The input network.

    Returns:
        Tuple of (feature_vector, n_components) where:
            - feature_vector: Shape (5,) structural metrics.
            - n_components: Number of connected components.

    Example:
        >>> cfg = {
        ...     "seed": 42,
        ...     "network": {"generator": "ER", "params": {"n": 50, "p": 0.1}},
        ... }
        >>> G = Network(cfg)
        >>> vec, n_comp = graph2vec(G)
        >>> vec.shape
        (5,)
    """
    A = G.adjacency_matrix()
    mean_sp = mean_shortest_path(A)
    gcc = global_clustering_coefficient(A)
    degrees = np.sum(A, axis=1)
    mean_deg = np.mean(degrees)
    std_deg = np.std(degrees)
    node0_deg = degrees[0]

    return np.array([mean_sp, gcc, mean_deg, std_deg, node0_deg]), len(
        connected_components(A)
    )


def generate_graph_pool(
    cfg_template: dict, seeds: np.ndarray, N: int
) -> tuple[np.ndarray, list[int]]:
    """Generate pool of candidate graphs with structural features.

    Args:
        cfg_template: Network configuration template (seed will be overwritten).
        seeds: Array of random seeds for graph generation.
        N: Number of nodes per graph.

    Returns:
        Tuple of (feature_matrix, component_counts) where:
            - feature_matrix: Shape (len(seeds), 5) structural features.
            - component_counts: Number of connected components per graph.

    Example:
        >>> cfg = {"network": {"generator": "ER", "params": {"n": 48, "p": 0.3}}}
        >>> rng = np.random.Generator(np.random.PCG64(42))
        >>> seeds = rng.integers(0, 2**32, size=100)
        >>> X, components = generate_graph_pool(cfg, seeds, N=48)
        >>> X.shape
        (100, 5)
    """
    X = []
    component_counts = []

    for seed in seeds:
        cfg = cfg_template.copy()
        cfg["seed"] = int(seed)
        G = Network(cfg, remap_seed=False)
        vec, n_components = graph2vec(G)
        X.append(vec)
        component_counts.append(n_components)

    return np.array(X), component_counts


def select_diverse_graphs(
    X: np.ndarray,
    component_counts: list[int],
    K: int,
    require_connected: bool = True,
) -> tuple[list[int], np.ndarray]:
    """Select K structurally diverse graphs using maximin criterion.

    Args:
        X: Feature matrix of shape (N, 5) with structural metrics.
        component_counts: Number of connected components per graph.
        K: Number of graphs to select.
        require_connected: If True, only select fully connected graphs.

    Returns:
        Tuple of (selected_indices, scaled_features) where:
            - selected_indices: Indices of selected graphs.
            - scaled_features: Z-score normalized feature matrix.

    Raises:
        ValueError: If require_connected=True but insufficient connected graphs.

    Example:
        >>> X = np.random.rand(100, 5)
        >>> components = [1] * 100  # All connected
        >>> selected, Xz = select_diverse_graphs(X, components, K=8)
        >>> len(selected)
        8
    """
    if require_connected:
        connected_mask = np.array(component_counts) == 1
        if connected_mask.sum() < K:
            raise ValueError(
                f"Only {connected_mask.sum()} connected graphs available, need {K}"
            )
        X_filtered = X[connected_mask]
    else:
        X_filtered = X
        connected_mask = None

    # Standardize features
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_filtered)

    # Maximin selection
    selected_local = maximin_select(Xz, K=K)

    # Map back to original indices if filtering was applied
    if require_connected and connected_mask is not None:
        original_indices = np.where(connected_mask)[0]
        selected_indices = [int(original_indices[i]) for i in selected_local]
    else:
        selected_indices = selected_local

    return selected_indices, Xz


def build_graph_pool(
    generator: str,
    n: int,
    seeds: np.ndarray,
    param_sets: list[dict],
) -> pl.DataFrame:
    """Build a candidate pool over the product of seeds and generator parameters.

    Unlike :func:`generate_graph_pool`, which holds the generator parameters
    fixed, this sweeps a list of parameter settings so that the pool covers a
    range of structural regimes (e.g. several edge probabilities for ER).

    Args:
        generator: "ER" or "WS".
        n: Number of nodes per graph.
        seeds: Candidate graph seeds.
        param_sets: Generator parameter settings, e.g. ``[{"p": 0.2}, ...]``
            for ER or ``[{"k": 8, "beta": 0.1}, ...]`` for WS.

    Returns:
        One row per (parameter setting, seed) with the generator parameters, the
        five structural features in :data:`FEATURE_COLS`, ``n_components``, and
        ``diameter`` (infinite when the graph is disconnected). The last two are
        eligibility filters, not maximin features.

    Example:
        >>> seeds = np.array([1, 2, 3])
        >>> pool = build_graph_pool("ER", 48, seeds, [{"p": 0.2}, {"p": 0.3}])
        >>> pool.height
        6
    """
    param_cols = PARAM_COLS[generator]
    rows = []

    for params in param_sets:
        for seed in seeds:
            cfg = {
                "seed": int(seed),
                "network": {"generator": generator, "params": {"n": n, **params}},
            }
            G = Network(cfg, remap_seed=False)
            vec, n_components = graph2vec(G)
            rows.append(
                {
                    "seed": int(seed),
                    **{c: params[c] for c in param_cols},
                    **dict(zip(FEATURE_COLS, vec.tolist(), strict=True)),
                    "n_components": n_components,
                    "diameter": graph_diameter(G.adjacency_matrix()),
                }
            )

    return pl.DataFrame(rows)


def graph_diameter(A: np.ndarray) -> float:
    """Longest shortest path in the graph.

    Args:
        A: Adjacency matrix of shape (N, N).

    Returns:
        The diameter, or ``inf`` if the graph is disconnected.

    Note:
        Influence spreads one hop per debate round, so a graph whose diameter
        exceeds ``experiment.max_rounds`` is not fully traversable within a run.

    Example:
        >>> graph_diameter(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        2.0
    """
    spl = shortest_path_lengths(A)
    return float(spl.max())  # inf if any pair is unreachable


def extend_graph_selection(
    base: pl.DataFrame,
    pool: pl.DataFrame,
    n_add: int,
    strata_col: str | list[str] | None = None,
    quotas: dict | None = None,
    require_connected: bool = True,
    max_diameter: float | None = None,
) -> pl.DataFrame:
    """Add realizations to an existing selection using the maximin criterion.

    The already-selected realizations in ``base`` are pinned as the initial set
    of Algorithm 1, so the additions maximize their distance to what has already
    been simulated. Features are standardized on the *union* of ``base`` and
    ``pool``, which is what makes distances comparable once the generator
    parameters (and hence the feature scales) vary.

    Args:
        base: Already-selected realizations; must contain :data:`FEATURE_COLS`.
        pool: Candidate realizations from :func:`build_graph_pool`.
        n_add: Number of realizations to add.
        strata_col: Column(s) of ``pool`` defining the strata for the quota
            (e.g. "p", or ``["k", "beta_band"]`` for a two-factor design). If
            None, plain maximin is used.
        quotas: Map stratum value -> minimum number of picks, or a list of such
            maps aligned with ``strata_col``. With several columns the quotas
            apply *marginally*, one per factor, so a design needs a quota per
            level rather than per cell of the grid. Required when ``strata_col``
            is set. Picks beyond the quota total are placed by unrestricted
            maximin.
        require_connected: Drop candidates with more than one component.
        max_diameter: Drop candidates whose diameter exceeds this. Influence
            spreads one hop per debate round, so a graph with a diameter above
            ``experiment.max_rounds`` is never fully traversed within a run.

    Returns:
        The selected pool rows, in selection order, with two extra columns:
        ``selection_order`` and ``d_min`` (distance to the already-selected set
        in standardized feature space at the moment of selection).

    Raises:
        ValueError: If ``strata_col`` is given without ``quotas``, or if the
            filters leave fewer eligible candidates than ``n_add``.

    Example:
        >>> new = extend_graph_selection(
        ...     base, pool, n_add=6, strata_col="p", quotas={0.2: 1, 0.25: 1}
        ... )  # doctest: +SKIP
    """
    if strata_col is not None and quotas is None:
        raise ValueError("quotas must be provided when strata_col is set")

    cand = pool.filter(pl.col("n_components") == 1) if require_connected else pool
    if max_diameter is not None:
        cand = cand.filter(pl.col("diameter") <= max_diameter)
    if cand.height < n_add:
        raise ValueError(
            f"Only {cand.height} eligible candidates for {n_add} additions; "
            "relax require_connected / max_diameter or enlarge the pool"
        )

    # Drop graph seeds already in the design. The seed alone -- not the seed
    # together with its parameters -- is what identifies a realization in the
    # run configs and downstream analysis, so reusing one would collide two
    # structurally different graphs under a single graph id.
    cand = cand.join(base.select("seed").unique(), on="seed", how="anti")

    X = np.vstack(
        [base.select(FEATURE_COLS).to_numpy(), cand.select(FEATURE_COLS).to_numpy()]
    ).astype(float)
    Xz = StandardScaler().fit_transform(X)

    n_base = base.height
    anchors = list(range(n_base))

    if strata_col is None:
        selected = maximin_select(Xz, K=n_base + n_add, init_selected=anchors)[n_base:]
    else:
        # Anchors carry a sentinel stratum so they never consume quota.
        cols = [strata_col] if isinstance(strata_col, str) else list(strata_col)
        factors = []
        for col in cols:
            labels = np.empty(n_base + cand.height, dtype=object)
            labels[:n_base] = None
            labels[n_base:] = cand[col].to_list()
            factors.append(labels)

        selected = maximin_select_stratified(
            Xz,
            n_select=n_add,
            strata=factors if len(factors) > 1 else factors[0],
            quotas=quotas,
            init_selected=anchors,
            allow_overflow=True,
        )

    # Distance to the growing selection, recorded at the time each point was added.
    d_min = []
    for step, idx in enumerate(selected):
        prev = anchors + selected[:step]
        d_min.append(float(np.linalg.norm(Xz[prev] - Xz[idx], axis=1).min()))

    return (
        cand[[i - n_base for i in selected]]
        .with_columns(
            selection_order=pl.Series(range(1, len(selected) + 1)),
            d_min=pl.Series(d_min),
        )
        .drop("n_components")
    )
