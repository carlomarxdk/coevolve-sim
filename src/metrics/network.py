from collections import deque
from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as scipy_connected_components
from scipy.sparse.csgraph import shortest_path


def to_binary_adj(A: np.ndarray) -> np.ndarray:
    """
    Ensure a simple, undirected, binary adjacency matrix with zero diagonal.
    Args:
        A: Adjacency matrix (np.ndarray) of shape (N, N)
    Returns:
        Binary adjacency matrix (np.ndarray) of shape (N, N)
    """
    A_bin = (A > 0).astype(int)
    np.fill_diagonal(A_bin, 0)

    return A_bin


def network_density(A: np.ndarray) -> float:
    """
    Compute the density of the graph represented by adjacency matrix A.
    Density = 2 * E / (N * (N - 1)) for undirected graphs
    Args:
        A: Adjacency matrix (np.ndarray) of shape (N, N)
    Returns:
        float: density of the graph
    """
    N = A.shape[0]
    if N < 2:
        return 0.0
    E = np.sum(A) / 2.0  # undirected edges
    return float(2.0 * E / (N * (N - 1)))


def shortest_path_lengths(
    A: np.ndarray, directed: bool = False, unweighted: bool = True
) -> np.ndarray:
    """
    Compute all-pairs shortest path lengths.
    Args:
        A: Adjacency matrix (np.ndarray) of shape (N, N)
        directed: Whether the graph is directed
        unweighted: Whether to treat the graph as unweighted
    Returns:
        Distance matrix with np.inf for disconnected pairs
    """
    A_sparse = csr_matrix(A)
    return shortest_path(A_sparse, directed=directed, unweighted=unweighted)


def connected_components(A: np.ndarray) -> List[List[int]]:
    """
    Find connected components in an undirected graph using scipy.
    Args:
        A: Binary adjacency matrix (np.ndarray) of shape (N, N)
    Returns:
        List of components, each component is a list of node indices
    """
    A_sparse = csr_matrix(A)
    n_components, labels = scipy_connected_components(A_sparse, directed=False)

    comps = []
    for c in range(n_components):
        comps.append(np.where(labels == c)[0].tolist())

    return comps


def graph_radius_diameter(dist_matrix: np.ndarray) -> Tuple[int, int]:
    """
    Compute radius and diameter from the all-pairs shortest path matrix.
    Compute the radius and diameter of the graph from the distance matrix.
    Radius: minimum eccentricity (max distance from a node to all others)
    Diameter: maximum eccentricity

    :param dist_matrix: all-pairs shortest path lengths (np.ndarray)
    :return: (radius, diameter)
    """
    ecc = np.nanmax(np.where(np.isfinite(dist_matrix), dist_matrix, np.nan), axis=1)
    ecc = ecc[ecc > 0]
    if len(ecc) == 0:
        return (0, 0)
    return int(np.min(ecc)), int(np.max(ecc))


def k_core_decomposition(A: np.ndarray) -> np.ndarray:
    """
    Compute k-core number for each node.
    The k-core of a graph is the maximal subgraph where each node has at least k neighbors.
    Args:
        A: adjacency matrix (numpy array)
    Returns:
        np.ndarray: array of k-core numbers for each node
    """
    N = A.shape[0]
    adj = [np.where(A[i] > 0)[0].tolist() for i in range(N)]
    degree = np.array([len(nei) for nei in adj])
    k_core = np.zeros(N, dtype=int)

    # Queue of nodes to remove at current k-level
    from collections import deque

    k = 0
    remaining = set(range(N))

    while remaining:
        # Initialize queue with nodes of degree <= k
        queue = deque([n for n in remaining if degree[n] <= k])

        if not queue:
            k += 1
            continue

        while queue:
            node = queue.popleft()
            if node not in remaining:
                continue

            k_core[node] = k
            remaining.remove(node)

            # Decrease degree of neighbors
            for neigh in adj[node]:
                if neigh in remaining:
                    degree[neigh] -= 1
                    if degree[neigh] <= k:
                        queue.append(neigh)

    return k_core


def node_triangles(A: np.ndarray) -> np.ndarray:
    """
    Count how many triangles EACH node participates in (undirected graph).
    Args:
        A: adjacency matrix (numpy array)
    Returns:
        np.ndarray: array of triangle counts per node
    """
    A_bin = (A > 0).astype(int)  # ensure binary
    N = A_bin.shape[0]
    tri = np.zeros(N, dtype=int)

    for i in range(N):
        neighbors = np.where(A_bin[i])[0]
        d = len(neighbors)
        if d < 2:
            continue

        # Count edges among neighbors
        subgraph = A_bin[np.ix_(neighbors, neighbors)]
        # Each edge is 1 triangle containing i
        tri[i] = int(np.sum(subgraph) // 2)

    return tri


def total_triangles(A: np.ndarray) -> int:
    """
    Count total number of triangles in an undirected simple graph.
    Args:
        A: adjacency matrix (numpy array)
    Returns:
        int: number of triangles in the graph
    """
    # Ensure binary adjacency
    A_bin = (A > 0).astype(int)

    # Ensure no self-loops
    np.fill_diagonal(A_bin, 0)

    # Compute trace(A^3) / 6
    A3 = np.linalg.matrix_power(A_bin, 3)
    tri = int(np.trace(A3) // 6)
    return tri


def node_wedges(A: np.ndarray) -> np.ndarray:
    """
    Number of wedges (open or closed triples) centered at each node.
    For node i with degree k_i: wedges_i = C(k_i, 2) = k_i (k_i - 1) / 2
    Args:
        A: adjacency matrix (numpy array)
    Returns:
        np.ndarray: array of wedge counts per node
    """
    A_bin = to_binary_adj(A)
    degrees = A_bin.sum(axis=1)
    wedges = (degrees * (degrees - 1) // 2).astype(int)
    return wedges


def total_wedges(A: np.ndarray) -> int:
    """
    Total number of wedges in the graph.
    Args:
        A: adjacency matrix (numpy array)
    Returns:
        int: total number of wedges in the graph
    """
    return int(node_wedges(A).sum())


def eigenvector_centrality(
    A: np.ndarray, max_iter: int = 100, tol: float = 1e-6
) -> np.ndarray:
    """
    Compute eigenvector centrality using power iteration.
    Eigenvector centrality measures the influence of a node in a network.
    Args:
        A: adjacency matrix (numpy array)
        max_iter: maximum number of iterations
        tol: tolerance for convergence
    Returns:
        np.ndarray: array of eigenvector centrality scores for each node
    """
    N = A.shape[0]

    # No edges = all zeros
    if np.sum(A) == 0:
        return np.zeros(N, float)

    # Initial vector
    x = np.ones(N, dtype=float) / N

    for _ in range(max_iter):
        x_new = A.dot(x)

        # Compute norm for normalization
        norm = np.linalg.norm(x_new)

        if norm < tol:
            # Graph is too sparse or degenerately disconnected
            return np.zeros(N, float)

        x_new = x_new / norm

        # Check convergence (relative)
        if np.linalg.norm(x_new - x) < tol * np.linalg.norm(x):
            return x_new

        x = x_new

    # Return last iterate if not fully converged
    return x


def closeness_centrality(A: np.ndarray) -> np.ndarray:
    """
    Compute closeness centrality for an unweighted graph from adjacency matrix A.

    Uses the standard NetworkX definition:
        C(i) = (N - 1) / sum_j d(i,j)
    where d(i,j) is shortest path distance.

    If a node cannot reach all others (disconnected graph),
    it uses reachable nodes only (again matching NetworkX behavior).

    Args:
        A: Adjacency matrix (numpy array, shape NxN)
    Returns:
        np.ndarray: closeness centrality scores per node
    """
    N = A.shape[0]
    closeness = np.zeros(N, dtype=float)

    # Precompute adjacency lists for fast BFS
    neighbors = [np.where(A[i] > 0)[0] for i in range(N)]

    for i in range(N):
        # BFS to compute shortest paths from node i
        dist = np.full(N, np.inf)
        dist[i] = 0
        queue = deque([i])

        while queue:
            v = queue.popleft()
            for w in neighbors[v]:
                if dist[w] == np.inf:
                    dist[w] = dist[v] + 1
                    queue.append(w)

        # Mask unreachable nodes
        reachable = dist < np.inf
        reachable_count = np.sum(reachable)

        if reachable_count > 1:  # More than just itself
            total_dist = np.sum(dist[reachable])

            # NetworkX normalization: (reachable_count - 1) / sum distances
            closeness[i] = (reachable_count - 1) / total_dist
        else:
            closeness[i] = 0.0

    return closeness


def bridging_coefficient(A: np.ndarray) -> np.ndarray:
    """
    Compute bridging coefficient for each node.
    Args:
        A: adjacency matrix (numpy array, NxN)
    Returns:
        np.ndarray: bridging coefficient (length N)
    """
    N = A.shape[0]
    degrees = A.sum(axis=1)

    # Prevent division by zero
    degrees_safe = np.where(degrees == 0, 1e-12, degrees)

    bc = np.zeros(N, dtype=float)

    for i in range(N):
        neighbors = np.where(A[i] > 0)[0]

        if len(neighbors) == 0:
            bc[i] = 0.0
            continue

        inv_deg_i = 1.0 / degrees_safe[i]
        inv_deg_neighbors = 1.0 / degrees_safe[neighbors]

        bc[i] = inv_deg_i / inv_deg_neighbors.sum()

    return bc


def betweenness_centrality(A: np.ndarray) -> np.ndarray:
    """
    Compute betweenness centrality for all nodes.
    Betweenness centrality measures the number of times a node acts as a bridge
    along the shortest path between two other nodes.
    Args:
        A: adjacency matrix (numpy array)
    Returns:
        np.ndarray: array of betweenness centrality scores for each node
    """

    N = A.shape[0]
    bc = np.zeros(N, dtype=float)

    adj = [np.where(A[i] > 0)[0] for i in range(N)]

    for s in range(N):
        stack: List[int] = []
        pred: List[List[int]] = [[] for _ in range(N)]

        # Shortest paths count
        sigma = np.zeros(N)
        sigma[s] = 1

        # Distance
        dist = np.full(N, -1, dtype=int)
        dist[s] = 0

        # BFS queue
        q = deque([s])

        # --- BFS (computes sigma and dist) ---
        while q:
            v = q.popleft()
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:  # w found first time
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:  # shortest path
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # --- Dependency accumulation ---
        delta = np.zeros(N)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta_v = (sigma[v] / sigma[w]) * (1 + delta[w])
                delta[v] += delta_v
            if w != s:
                bc[w] += delta[w]

    # Normalize for undirected graph
    bc /= 2.0

    return bc


def bridging_centrality(A: np.ndarray) -> np.ndarray:
    """
    Compute bridging centrality (pure NumPy, no networkx).

    BridgingCentrality = BetweennessCentrality * BridgingCoefficient
    """
    bc = bridging_coefficient(A)
    bet = betweenness_centrality(A)
    return bet * bc


def local_clustering_coefficients(A: np.ndarray) -> np.ndarray:
    """
    Local clustering coefficient for each node.
    For node i: C_i = 2 T_i / (k_i (k_i - 1)) if k_i >= 2, else 0.
    Args:
        A: adjacency matrix (numpy array)
    Returns:
        np.ndarray: array of local clustering coefficients per node
    """
    A_bin = to_binary_adj(A)
    degrees = A_bin.sum(axis=1)
    node_tri = node_triangles(A)

    C = np.zeros_like(degrees, dtype=float)
    mask = degrees >= 2

    denom = degrees[mask] * (degrees[mask] - 1)
    C[mask] = (2.0 * node_tri[mask]) / denom

    return C


def global_clustering_coefficient(A: np.ndarray) -> float:
    """
    Global clustering coefficient (transitivity).
    Defined as: 3 * (#triangles) / (#wedges), with 0 if no wedges.
    Args:
        A: adjacency matrix (numpy array)
    Returns:
        float: global clustering coefficient
    """
    total_tri = total_triangles(A)
    total_wedge_count = total_wedges(A)

    if total_wedge_count == 0:
        return 0.0

    closed_triples = 3 * total_tri
    return closed_triples / total_wedge_count


def cross_belief_fraction_discrete(beliefs: np.ndarray, A: np.ndarray) -> float:
    """
    Fraction of edges connecting nodes with different (categorical) beliefs.
    Args:
        beliefs: array of beliefs (categorical labels) of shape (num_agents,)
        adjacency_matrix: adjacency matrix of the network (numpy array)
    Returns:
        float: fraction of cross-belief edges

    """
    b = beliefs  # categorical labels: e.g., 0,1,2

    iu = np.triu_indices(len(b), k=1)

    edges = A[iu] > 0
    if edges.sum() == 0:
        return 0.0

    cross = (b[iu[0]] != b[iu[1]]) & edges

    return cross.sum() / edges.sum()


def cross_belief_fraction_continuous(beliefs: np.ndarray, A: np.ndarray) -> float:
    """
    Average normalized absolute belief difference across edges.
    Returns 0 if no edges or all beliefs identical.
    Args:
        beliefs: array of beliefs (continuous values) of shape (num_agents,)
        A: adjacency matrix of the network (numpy array)
    Returns:
        float: average normalized belief difference across edges
    """
    b = beliefs.astype(float)
    N = len(b)
    iu = np.triu_indices(N, k=1)

    edges = A[iu] > 0
    m = edges.sum()
    if m == 0:
        return 0.0

    b_range = b.max() - b.min()
    if b_range < 1e-12:
        return 0.0

    diff = np.abs(b[iu[0]] - b[iu[1]])[edges]
    return float((diff / b_range).mean())


def assortativity_continuous(beliefs: np.ndarray, A: np.ndarray) -> float:
    """
    Assortativity for continuous beliefs = Pearson correlation
    of beliefs at the ends of edges.
    Args:
        beliefs: array of beliefs (continuous values) of shape (num_agents,)
        A: adjacency matrix of the network (numpy array)
    Returns:
        float: assortativity coefficient (Pearson correlation)
    """
    b = beliefs.astype(float)
    N = len(b)
    iu = np.triu_indices(N, k=1)

    edges = A[iu] > 0
    if edges.sum() == 0:
        return 0.0

    x = b[iu[0]][edges]
    y = b[iu[1]][edges]

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0

    r = np.corrcoef(x, y)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(r)


def assortativity_categorical(labels: np.ndarray, A: np.ndarray) -> float:
    """
    Newman's assortativity coefficient for categorical attributes.

    Args:
        labels: int array of shape (N,)
        A: undirected adjacency matrix
    Returns:
        float: assortativity coefficient (Newman's assortativity)
    """
    labels = labels.astype(int)
    N = len(labels)
    K = labels.max() + 1

    iu = np.triu_indices(N, k=1)
    edges_mask = A[iu] > 0
    if edges_mask.sum() == 0:
        return 0.0

    u = iu[0][edges_mask]
    v = iu[1][edges_mask]
    lu = labels[u]
    lv = labels[v]

    m = len(u)  # number of undirected edges

    # e_ij: fraction of "directed" edges from type i to type j
    e = np.zeros((K, K), dtype=float)
    for a, b in zip(lu, lv):
        e[a, b] += 1.0
        e[b, a] += 1.0  # count both directions

    e /= 2.0 * m

    a = e.sum(axis=1)
    b = e.sum(axis=0)  # same for undirected

    trace_e = np.trace(e)
    ab_sum = (a * b).sum()

    if 1.0 - ab_sum == 0:
        return 0.0

    r = (trace_e - ab_sum) / (1.0 - ab_sum)
    return float(r)


def modularity_categorical(A: np.ndarray, labels: np.ndarray) -> float:
    """
    Modularity of a given partition defined by categorical labels.
    Args:
        A: undirected adjacency matrix
        labels: int array of shape (N,)
    """
    A = A.astype(float)
    labels = labels.astype(int)
    N = len(labels)

    m = A.sum() / 2.0
    if m == 0:
        return 0.0

    k = A.sum(axis=1)  # degrees

    K = labels.max() + 1
    S = np.zeros((N, K), dtype=float)
    S[np.arange(N), labels] = 1.0  # one-hot group membership

    B = A - np.outer(k, k) / (2.0 * m)

    Q = (S.T @ B @ S).trace() / (2.0 * m)
    return float(Q)


def local_agreement_metrics(
    A: np.ndarray, beliefs: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Compute local agreement per node and its mean.

    Args:
        A: adjacency matrix (NxN)
        beliefs: array of node beliefs (length N)

    Returns:
        local_agreement:
        local_agreement_mean: float
        neighbor_mean_beliefs: array of length N
    """
    N = len(beliefs)
    A = A.astype(int)
    degrees = A.sum(axis=1)

    local_agreement = np.zeros(N, dtype=float)
    neighbor_mean_beliefs = np.zeros(N, dtype=float)

    for i in range(N):
        deg = degrees[i]
        if deg == 0:
            local_agreement[i] = 1.0  # convention
            neighbor_mean_beliefs[i] = 0.0
            continue

        neighbors = np.where(A[i] > 0)[0]
        same = np.sum(beliefs[neighbors] == beliefs[i])
        local_agreement[i] = same / deg
        neighbor_mean_beliefs[i] = float(np.mean(beliefs[neighbors]))

    return (local_agreement, float(np.mean(local_agreement)), neighbor_mean_beliefs)


def edge_disagreement(A: np.ndarray, beliefs: np.ndarray) -> float:
    """
    Fraction of edges where endpoint beliefs differ.
    Args:
        A: adjacency matrix (NxN), undirected
        beliefs: length-N belief array

    Returns:
        float in [0,1]
    """
    iu = np.triu_indices_from(A, k=1)

    edges = A[iu] > 0
    total_edges = np.sum(edges)
    if total_edges == 0:
        return 0.0

    bi = beliefs[iu[0]]
    bj = beliefs[iu[1]]

    disagree = (bi != bj) & edges
    return float(np.sum(disagree) / total_edges)


#     Args:
#         A: Binary adjacency matrix (np.ndarray) of shape (N, N)

#     Returns:
#         Distance matrix with np.inf for disconnected pairs
#     """
#     N = A.shape[0]
#     adj_lists = [np.where(A[i])[0] for i in range(N)]

#     dist = np.full((N, N), np.inf)
#     np.fill_diagonal(dist, 0)

#     for source in range(N):
#         dist_source = dist[source]
#         visited = np.zeros(N, dtype=bool)
#         visited[source] = True

#         queue = deque([source])

#         while queue:
#             current = queue.popleft()
#             current_dist = dist_source[current]

#             for neighbor in adj_lists[current]:
#                 if not visited[neighbor]:
#                     visited[neighbor] = True
#                     dist_source[neighbor] = current_dist + 1
#                     queue.append(neighbor)

#     return dist


# def graph_radius_diameter(dist_matrix: np.ndarray) -> Tuple[int, int]:
#     """
#     Compute radius and diameter from the all-pairs shortest path matrix.
#     Compute the radius and diameter of the graph from the distance matrix.
#     Radius: minimum eccentricity (max distance from a node to all others)
#     Diameter: maximum eccentricity

#     :param dist_matrix: all-pairs shortest path lengths (np.ndarray)
#     :return: (radius, diameter)
#     """
#     # Eccentricity: max finite distance in each row
#     ecc = np.nanmax(np.where(np.isfinite(dist_matrix), dist_matrix, np.nan), axis=1)

#     # Remove zero (self-distances)
#     ecc = ecc[ecc > 0]

#     if len(ecc) == 0:
#         return (0, 0)

#     return int(np.min(ecc)), int(np.max(ecc))


# def connected_components(A: np.ndarray) -> List[List[int]]:
#     """
#     Find connected components in an undirected graph using BFS.
#     Args:
#         A: Binary adjacency matrix (np.ndarray) of shape (N, N)
#     Returns:
#         List of components, each component is a list of node indices
#     """
#     N = A.shape[0]
#     adj = [np.where(A[i] > 0)[0] for i in range(N)]
#     visited = np.zeros(N, dtype=bool)
#     comps = []

#     for i in range(N):
#         if not visited[i]:
#             comp = []
#             q = deque([i])
#             visited[i] = True

#             while q:
#                 u = q.popleft()
#                 comp.append(u)
#                 for v in adj[u]:
#                     if not visited[v]:
#                         visited[v] = True
#                         q.append(v)

#             comps.append(comp)

#     return comps


#### Influence Scores
def compute_node_influence(
    beliefs_history: np.ndarray, adjacency_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute how much each node influences others
    beliefs_history: T x N array (T timesteps, N nodes)
    adjacency_matrix: N x N adjacency matrix
    """

    if adjacency_matrix.shape[0] != beliefs_history.shape[1]:
        beliefs_history = beliefs_history.T
    T, N = beliefs_history.shape
    node_influence = np.zeros(N)

    for t in range(1, T):
        for i in range(N):
            # Find who i influences (i's outgoing edges)
            influenced_nodes = np.where(adjacency_matrix[i] > 0)[0]

            for j in influenced_nodes:
                # How much did j change?
                belief_change_j = beliefs_history[t, j] - beliefs_history[t - 1, j]

                # How close was j's new belief to i's previous belief?
                pull_toward_i = beliefs_history[t - 1, i] - beliefs_history[t - 1, j]

                # If j moved toward i's belief, credit i
                if np.sign(belief_change_j) == np.sign(pull_toward_i):
                    node_influence[i] += np.abs(belief_change_j)

    # Normalize by time and number of neighbors
    for i in range(N):
        n_neighbors = np.sum(adjacency_matrix[i] > 0)
        if n_neighbors > 0:
            node_influence[i] /= (T - 1) * n_neighbors

    return node_influence


def belief_convergence_influence(
    beliefs_history: np.ndarray, adjacency_matrix: np.ndarray, lag: int = 1
) -> np.ndarray:
    """
    Identify nodes whose belief changes precede network-wide changes
    """
    if adjacency_matrix.shape[0] != beliefs_history.shape[1]:
        beliefs_history = beliefs_history.T
    T, N = beliefs_history.shape
    authority_scores = np.zeros(N)

    for i in range(N):
        for t in range(lag, T - lag):
            # Node i's change at time t
            i_change = beliefs_history[t, i] - beliefs_history[t - lag, i]

            if np.abs(i_change) > 0.01:  # Significant change
                # Check how many neighbors follow this trend
                neighbors = np.where(adjacency_matrix[i] > 0)[0]
                follow_score = 0

                for j in neighbors:
                    # Neighbor j's subsequent change
                    j_future_change = (
                        beliefs_history[t + lag, j] - beliefs_history[t, j]
                    )

                    # Did j move in same direction as i?
                    if np.sign(j_future_change) == np.sign(i_change):
                        correlation = np.abs(j_future_change / i_change)
                        follow_score += min(correlation, 1.0)  # Cap at 1

                if len(neighbors) > 0:
                    authority_scores[i] += follow_score / len(neighbors)

    # Normalize by number of time steps
    authority_scores /= T - 2 * lag

    return authority_scores
