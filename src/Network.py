from __future__ import annotations

import logging
import random
from typing import Dict, Iterator, List, Set, Tuple

import numpy as np

log = logging.getLogger("Network")


class Network:
    """Social network topology for agent interactions.

    Manages network structure including node connections, edge operations,
    and various network generation algorithms. Currently supports Erdős-Rényi
    random graphs.

    Attributes:
        generator: Type of network generator used (e.g., 'ER' for Erdős-Rényi).
        n: Number of nodes in the network.
        adj: Adjacency list mapping node IDs to lists of neighbor IDs.
        _edges: Set of edges as (u, v) tuples.
        nodes: List of all node IDs.
        p: Edge probability for ER generator.
        rng: Random number generator for reproducibility.
    """

    @classmethod
    def from_edge_list(
        cls, edge_list: List[Tuple[int, int]], n: int = None, seed: int = 42
    ) -> "Network":
        """Create a network from a list of edges.

        Args:
            edge_list: List of edge tuples (u, v).
            n: Number of nodes. If None, inferred from max node ID + 1.
            seed: Random seed for reproducibility.

        Returns:
            Network instance with the specified edges.

        Raises:
            ValueError: If edge list contains invalid node IDs.
        """
        # Infer number of nodes if not provided
        if n is None:
            if not edge_list:
                n = 0
            else:
                n = max(max(u, v) for u, v in edge_list) + 1

        # Validate edge list
        for u, v in edge_list:
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"Edge ({u}, {v}) contains invalid node ID for n={n}")

        # Create minimal configuration (not currently used)
        # cfg = {"network": {"generator": "edge_list", "params": {"n": n}}, "seed": seed}

        # Create instance with custom generator
        network = cls.__new__(cls)
        network.generator = "edge_list"
        network.n = n
        network.adj = {u: [] for u in range(n)}
        network._edges = set()
        network.nodes = list(range(n))
        network.p = None
        network.rng = random.Random(seed)

        # Add all edges
        for u, v in edge_list:
            network.add_edge_undirected(u, v)

        log.info(
            f"Built network from edge list with {n} nodes and {len(edge_list)} edges"
        )
        return network

    def __init__(self, cfg: Dict) -> None:
        """Initialize network from configuration.

        Args:
            cfg: Configuration dictionary containing:
                - network.generator: Network type (e.g., 'ER')
                - network.params.n: Number of nodes
                - network.params.p: Edge probability (for ER)
                - seed: Random seed for reproducibility

        Raises:
            ValueError: If required parameters are missing.
            NotImplementedError: If generator type is not supported.
        """
        self.generator: str = cfg.get("network").get("generator")
        self.n: int = cfg.get("network").get("params").get("n")
        self.adj: Dict[int, List[int]] = {u: [] for u in range(self.n)}
        self._edges: Set[Tuple[int, int]] = set()
        self.nodes: List[int] = list(range(self.n))

        # model-specific parameters
        self.p: float = cfg.get("network").get("params").get("p", None)

        # set up rng
        self.rng = random.Random(cfg.get("seed", 42))

        # initialize network
        if self.generator == "ER":  # Erdős–Rényi
            if self.p is None:
                raise ValueError("ER generator requires parameter p")
            self.build_er()
        else:
            raise NotImplementedError(f'Unknown network model: "{self.generator}"')

    def build_er(self) -> None:
        """Build an Erdős-Rényi G(n, p) random graph.

        Each possible edge between distinct nodes is included independently
        with probability p.

        Raises:
            ValueError: If p is not in the range [0, 1].
        """
        if not (0.0 <= self.p <= 1.0):
            raise ValueError("p must be in [0,1]")

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.rng.random() < self.p:
                    self.add_edge_undirected(i, j)
        log.info(f"Built ER network with {self.n} nodes and p={self.p}")

    def add_edge_undirected(self, u: int, v: int) -> None:
        """Add an undirected edge between two nodes.

        The edge is added in both directions in the adjacency list.
        Self-loops and duplicate edges are ignored.

        Args:
            u: First node ID.
            v: Second node ID.
        """
        # check if self-edge or edge exists
        if u == v:
            return
        if (u, v) in self._edges or (v, u) in self._edges:
            return

        # add edge
        self._edges.add((u, v))
        self.adj[u].append(v)
        self.adj[v].append(u)

    def neighbors(self, i: int) -> List[int]:
        """Get all neighbors of a node.

        Args:
            i: Node ID.

        Returns:
            List of neighbor node IDs.
        """
        return self.adj.get(i, [])

    def degree(self, i: int) -> int:
        """Get the degree of a node (number of neighbors).

        Args:
            i: Node ID.

        Returns:
            Number of neighbors of the node.
        """
        return len(self.adj.get(i, []))

    def edges(self) -> Iterator[Tuple[int, int]]:
        """Get an iterator over all edges in the network.

        Returns:
            Iterator over edge tuples (u, v).
        """
        return iter(self._edges)

    def has_edge(self, u: int, v: int) -> bool:
        """Check if an edge exists between two nodes.

        Args:
            u: First node ID.
            v: Second node ID.

        Returns:
            True if edge exists, False otherwise.
        """
        return (u, v) in self._edges or (v, u) in self._edges

    def add_edge(self, u: int, v: int) -> None:
        """Add an edge between two nodes.

        Alias for add_edge_undirected.

        Args:
            u: First node ID.
            v: Second node ID.
        """
        self.add_edge_undirected(u, v)

    def remove_edge(self, u: int, v: int) -> None:
        """Remove an edge between two nodes.

        Removes the edge from both the edge set and adjacency lists.

        Args:
            u: First node ID.
            v: Second node ID.
        """
        if (u, v) in self._edges:
            self._edges.remove((u, v))
            if v in self.adj[u]:
                self.adj[u].remove(v)
            if u in self.adj[v]:
                self.adj[v].remove(u)

        elif (v, u) in self._edges:
            self._edges.remove((v, u))
            if v in self.adj[u]:
                self.adj[u].remove(v)
            if u in self.adj[v]:
                self.adj[v].remove(u)

    def adjacency_matrix(self) -> List[List[int]]:
        """
        Return adjacency matrix.

        :return: adjacency matrix
        """

        N = self.n
        A = np.zeros((N, N), dtype=float)
        for u, nbrs in self.adj.items():
            for v in nbrs:
                A[u, v] = 1.0
        # enforce symmetry for undirected graphs
        A = np.maximum(A, A.T)
        return A
