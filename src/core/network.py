"""Network topology management for agent-based simulations.

This module provides classes and utilities for creating and managing social network
structures used in agent-based debate simulations. Supports multiple network generation
algorithms including Erdős-Rényi and Watts-Strogatz models.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from omegaconf import DictConfig


log = logging.getLogger("Network")


def seed_remap(seed: int | None, seed_map: dict) -> int | None:
    """Remap a random seed into the predefined set for the given generator.

    See the `misc_graph_selection.ipynb` notebook for details.

    Args:
        seed: Original seed value.
        seed_map: Map from seeds to network seeds

    Returns:
        Remapped seed within [0, 2**32 - 1], or None if seed is None.
    """
    if seed is None:
        return None

    try:
        output = seed_map[seed]
        log.info(f"Remapped seed {seed} to {output} using remapping lookup.")
    except KeyError:
        log.warning(f"Seed {seed} not found in remapping lookup; using original seed.")
        output = seed
    return output


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
        cls, edge_list: list[tuple[int, int]], n: int | None = None, seed: int = 42
    ) -> Network:
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
            n = 0 if not edge_list else max(max(u, v) for u, v in edge_list) + 1

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
        network.k = None
        network.beta = None
        network.rng = random.Random(seed)

        # Add all edges
        for u, v in edge_list:
            network.add_edge_undirected(u, v)

        log.info(
            f"Built network from edge list with {n} nodes and {len(edge_list)} edges"
        )
        return network

    def __init__(self, cfg: dict | DictConfig, remap_seed: bool = True) -> None:
        """Initialize network from configuration.

        Args:
            cfg: Configuration dictionary containing:
                - network.generator: Network type (e.g., 'ER')
                - network.params.n: Number of nodes
                - network.params.p: Edge probability (for ER)
                - network.params.k: K nearest neighbors (for WS)
                - network.params.beta: Beta, re-wiring probability (for WS)
                - seed: Random seed for reproducibility
            remap_seed: Whether to remap the seed for reproducibility.

        Raises:
            ValueError: If required parameters are missing.
            NotImplementedError: If generator type is not supported.
        """
        network_cfg = cfg.get("network")
        if network_cfg is None:
            raise ValueError("Configuration must contain 'network' key")

        self.generator: str | None = network_cfg.get("generator")
        if self.generator is None:
            raise ValueError("Network configuration must contain 'generator' key")

        if remap_seed:
            self.seed: int | None = seed_remap(
                cfg.get("seed"), cfg["network"].get("seed_map", {})
            )
        else:
            self.seed: int | None = cfg.get("seed")

        params_cfg = network_cfg.get("params")
        if params_cfg is None:
            raise ValueError("Network configuration must contain 'params' key")

        n_val = params_cfg.get("n")
        if n_val is None:
            raise ValueError("Network params must contain 'n' (number of nodes)")
        self.n: int = n_val

        self.adj: dict[int, list[int]] = {u: [] for u in range(self.n)}
        self._edges: set[tuple[int, int]] = set()
        self.nodes: list[int] = list(range(self.n))

        # Model-Specific Parameters
        ## ER Network
        self.p: float | None = params_cfg.get("p")
        ## WS Network
        self.k: int | None = params_cfg.get("k")
        self.beta: float | None = params_cfg.get("beta")

        # set up rng with remapped seed
        self.rng = random.Random(self.seed)

        # initialize network
        if self.generator == "ER":  # Erdős-Rényi
            if self.p is None:
                raise ValueError("ER generator requires parameter p")
            self.build_er()
        elif self.generator == "WS":  # Watts-Strogatz
            if self.k is None or self.beta is None:
                raise ValueError("WS generator requires parameters k and beta")
            self.build_ws()
        else:
            raise NotImplementedError(f'Unknown network model: "{self.generator}"')

    def build_er(self) -> None:
        """Build an Erdős-Rényi G(n, p) random graph.

        Each possible edge between distinct nodes is included independently
        with probability p.

        Raises:
            ValueError: If p is not in the range [0, 1] or is None.
        """
        if self.p is None:
            raise ValueError("p must be set for ER network")
        if not (0.0 <= self.p <= 1.0):
            raise ValueError("p must be in [0,1]")

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.rng.random() < self.p:
                    self.add_edge_undirected(i, j)
        log.info(f"Built ER network with {self.n} nodes and p={self.p}")

    def build_ws(self) -> None:
        """Build a Watts-Strogatz small-world network.

        Raises:
            ValueError: If k is not even or beta is not in [0, 1] or either is None.
        """
        if self.k is None:
            raise ValueError("k must be set for WS network")
        if self.beta is None:
            raise ValueError("beta must be set for WS network")

        if self.k % 2 != 0:
            raise ValueError("k must be even for Watts-Strogatz")
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError("beta must be in [0,1]")

        n = self.n
        k = self.k
        beta = self.beta

        # build ring lattice
        half_k = k // 2
        for i in range(n):
            for d in range(1, half_k + 1):
                j = (i + d) % n
                self.add_edge_undirected(i, j)

        # rewiring step
        edges = list(self._edges)  # snapshot to avoid mutation issues
        for u, v in edges:
            if self.rng.random() < beta:
                self.remove_edge(u, v)

                # choose new target
                while True:
                    w = self.rng.randrange(n)
                    if w != u and not self.has_edge(u, w):
                        break

                self.add_edge_undirected(u, w)

        log.info(f"Built WS network with n={n}, k={k}, beta={beta}")

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

    def neighbors(self, i: int) -> list[int]:
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

    def degrees(self) -> Sequence[int]:
        """Get the degrees of all nodes in the network.

        Returns:
            List of degrees for each node.
        """
        return [len(self.adj[u]) for u in range(self.n)]

    def edges(self) -> Iterator[tuple[int, int]]:
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

    def adjacency_matrix(self) -> np.ndarray:
        """Return adjacency matrix.

        Returns:
            Adjacency matrix as a numpy array.
        """
        N = self.n
        A = np.zeros((N, N), dtype=float)
        for u, nbrs in self.adj.items():
            for v in nbrs:
                A[u, v] = 1.0
        # enforce symmetry for undirected graphs
        A = np.maximum(A, A.T)
        return A
