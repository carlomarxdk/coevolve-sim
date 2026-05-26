"""Network graph selection using structural diversity criteria."""

import numpy as np
from diversity import maximin_select
from sklearn.preprocessing import StandardScaler

from src.core.metrics.network import (
    connected_components,
    global_clustering_coefficient,
    mean_shortest_path,
)
from src.core.network import Network


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
