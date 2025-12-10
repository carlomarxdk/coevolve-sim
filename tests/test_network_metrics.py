"""
Tests for network metrics functions in metrics modules
"""

import numpy as np
import pytest

from src.metrics.network import (
    betweenness_centrality,
    eigenvector_centrality,
    graph_radius_diameter,
    k_core_decomposition,
    node_triangles,
    shortest_path_lengths,
    total_triangles,
)


class TestNetworkMetrics:
    """Test network metrics computation functions"""

    def test_count_triangles_complete_graph(self):
        """Test triangle counting on a complete graph"""
        # K4 (complete graph with 4 nodes) has 4 triangles
        A = np.ones((4, 4)) - np.eye(4)
        triangles = total_triangles(A)
        assert triangles == 4

    def test_count_node_triangles_complete_graph(self):
        """Test per-node triangle counting on complete graph"""
        # K4 - each node participates in 3 triangles
        A = np.ones((4, 4)) - np.eye(4)
        tri_counts = node_triangles(A)
        assert len(tri_counts) == 4
        assert all(t == 3 for t in tri_counts)

    def test_count_node_triangles_simple(self):
        """Test per-node triangle counting on simple triangle"""
        # Single triangle: 0-1-2-0
        A = np.zeros((3, 3))
        A[0, 1] = A[1, 0] = 1
        A[1, 2] = A[2, 1] = 1
        A[2, 0] = A[0, 2] = 1

        tri_counts = node_triangles(A)
        assert len(tri_counts) == 3
        assert all(t == 1 for t in tri_counts)

    def test_count_node_triangles_line_graph(self):
        """Test per-node triangle counting on line graph (no triangles)"""
        # Line graph: 0-1-2 (no triangles)
        A = np.zeros((3, 3))
        A[0, 1] = A[1, 0] = 1
        A[1, 2] = A[2, 1] = 1

        tri_counts = node_triangles(A)
        assert len(tri_counts) == 3
        assert all(t == 0 for t in tri_counts)

    def test_betweenness_centrality_line_graph(self):
        """Test betweenness centrality on line graph"""
        # Line graph: 0-1-2
        # Node 1 is on the path between 0 and 2
        A = np.zeros((3, 3))
        A[0, 1] = A[1, 0] = 1
        A[1, 2] = A[2, 1] = 1

        bc = betweenness_centrality(A)
        assert len(bc) == 3
        assert bc[0] == 0.0  # endpoints have zero betweenness
        assert bc[2] == 0.0
        assert bc[1] > 0.0  # middle node has positive betweenness

    def test_betweenness_centrality_complete_graph(self):
        """Test betweenness centrality on complete graph"""
        # In a complete graph, all nodes have equal betweenness (should be 0)
        A = np.ones((4, 4)) - np.eye(4)
        bc = betweenness_centrality(A)
        assert len(bc) == 4
        # All nodes should have roughly equal betweenness (all ~0 since all paths are direct)
        assert np.allclose(bc, 0.0)

    def test_eigenvector_centrality_complete_graph(self):
        """Test eigenvector centrality on complete graph"""
        # In a complete graph, all nodes have equal centrality
        A = np.ones((4, 4)) - np.eye(4)
        ec = eigenvector_centrality(A)
        assert len(ec) == 4
        # All nodes should have equal centrality
        assert np.allclose(ec, ec[0])

    def test_eigenvector_centrality_star_graph(self):
        """Test eigenvector centrality on star graph"""
        # Star graph: node 0 connected to all others
        A = np.zeros((5, 5))
        for i in range(1, 5):
            A[0, i] = A[i, 0] = 1

        ec = eigenvector_centrality(A)
        assert len(ec) == 5
        # In a star graph, all nodes have equal eigenvector centrality
        # because the graph is bipartite
        assert np.allclose(ec, ec[0])

    def test_eigenvector_centrality_empty_graph(self):
        """Test eigenvector centrality on graph with no edges"""
        A = np.zeros((4, 4))
        ec = eigenvector_centrality(A)
        assert len(ec) == 4
        assert np.allclose(ec, 0.0)

    def test_k_core_complete_graph(self):
        """Test k-core decomposition on complete graph"""
        # K4 - all nodes have degree 3, so all are in 3-core
        A = np.ones((4, 4)) - np.eye(4)
        k_cores = k_core_decomposition(A)
        assert len(k_cores) == 4
        assert all(k == 3 for k in k_cores)

    def test_k_core_line_graph(self):
        """Test k-core decomposition on line graph"""
        # Line graph: 0-1-2
        # Endpoints have degree 1, middle has degree 2
        A = np.zeros((3, 3))
        A[0, 1] = A[1, 0] = 1
        A[1, 2] = A[2, 1] = 1

        k_cores = k_core_decomposition(A)
        assert len(k_cores) == 3
        # Endpoints should be in 0-core or 1-core
        assert k_cores[0] <= 1
        assert k_cores[2] <= 1
        # Middle node could be in 1-core
        assert k_cores[1] <= 2

    def test_k_core_disconnected_nodes(self):
        """Test k-core decomposition with isolated nodes"""
        # One isolated node
        A = np.zeros((4, 4))
        A[0, 1] = A[1, 0] = 1
        A[1, 2] = A[2, 1] = 1
        # Node 3 is isolated

        k_cores = k_core_decomposition(A)
        assert len(k_cores) == 4
        assert k_cores[3] == 0  # isolated node is in 0-core

    def test_shortest_path_lengths_complete_graph(self):
        """Test shortest path computation on complete graph"""
        # In K4, all pairwise distances should be 1
        A = np.ones((4, 4)) - np.eye(4)
        dist = shortest_path_lengths(A)

        for i in range(4):
            for j in range(4):
                if i == j:
                    assert dist[i][j] == 0
                else:
                    assert dist[i][j] == 1

    def test_shortest_path_lengths_line_graph(self):
        """Test shortest path computation on line graph"""
        # Line graph: 0-1-2
        A = np.zeros((3, 3))
        A[0, 1] = A[1, 0] = 1
        A[1, 2] = A[2, 1] = 1

        dist = shortest_path_lengths(A)
        assert dist[0][0] == 0
        assert dist[0][1] == 1
        assert dist[0][2] == 2
        assert dist[1][0] == 1
        assert dist[1][1] == 0
        assert dist[1][2] == 1
        assert dist[2][0] == 2
        assert dist[2][1] == 1
        assert dist[2][2] == 0

    def test_graph_radius_diameter_complete_graph(self):
        """Test radius and diameter on complete graph"""
        # K4 has radius=1 and diameter=1
        A = np.ones((4, 4)) - np.eye(4)
        dist = shortest_path_lengths(A)
        radius, diameter = graph_radius_diameter(dist)

        assert radius == 1
        assert diameter == 1

    def test_graph_radius_diameter_line_graph(self):
        """Test radius and diameter on line graph"""
        # Line graph 0-1-2 has radius=1 (from node 1) and diameter=2 (from 0 to 2)
        A = np.zeros((3, 3))
        A[0, 1] = A[1, 0] = 1
        A[1, 2] = A[2, 1] = 1

        dist = shortest_path_lengths(A)
        radius, diameter = graph_radius_diameter(dist)

        assert radius == 1  # min eccentricity
        assert diameter == 2  # max eccentricity

    def test_graph_radius_diameter_disconnected(self):
        """Test radius and diameter on disconnected graph"""
        # Two disconnected nodes
        A = np.zeros((2, 2))
        dist = shortest_path_lengths(A)
        radius, diameter = graph_radius_diameter(dist)

        # With no edges, radius and diameter should be 0
        assert radius == 0
        assert diameter == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
