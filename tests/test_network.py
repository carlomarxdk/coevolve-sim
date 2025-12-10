"""
Tests for Network class
"""

import pytest

from src.Network import Network


class TestNetwork:
    """Test the Network class"""

    @pytest.fixture
    def basic_er_config(self):
        """Create a basic ER network configuration"""
        return {
            "seed": 42,
            "network": {"generator": "ER", "params": {"n": 5, "p": 0.5}},
        }

    def test_network_init_er(self, basic_er_config):
        """Test network initialization with ER generator"""
        network = Network(basic_er_config)

        assert network.generator == "ER"
        assert network.n == 5
        assert network.p == 0.5
        assert len(network.nodes) == 5
        assert all(i in network.nodes for i in range(5))

    def test_network_invalid_generator(self):
        """Test that invalid generator raises NotImplementedError"""
        cfg = {"seed": 42, "network": {"generator": "INVALID", "params": {"n": 5}}}
        with pytest.raises(NotImplementedError):
            Network(cfg)

    def test_er_missing_p_parameter(self):
        """Test that ER generator requires p parameter"""
        cfg = {"seed": 42, "network": {"generator": "ER", "params": {"n": 5}}}
        with pytest.raises(ValueError, match="ER generator requires parameter p"):
            Network(cfg)

    def test_er_invalid_p_value(self):
        """Test that p must be in [0,1]"""
        cfg = {"seed": 42, "network": {"generator": "ER", "params": {"n": 5, "p": 1.5}}}
        with pytest.raises(ValueError, match="p must be in"):
            Network(cfg)

    def test_add_edge_undirected(self, basic_er_config):
        """Test adding undirected edges"""
        network = Network(basic_er_config)

        # Clear existing edges
        network._edges.clear()
        network.adj = {u: [] for u in range(network.n)}

        network.add_edge_undirected(0, 1)

        assert network.has_edge(0, 1)
        assert network.has_edge(1, 0)
        assert 1 in network.neighbors(0)
        assert 0 in network.neighbors(1)

    def test_add_edge_prevents_self_loops(self, basic_er_config):
        """Test that self-edges are not added"""
        network = Network(basic_er_config)

        initial_edges = len(network._edges)
        network.add_edge_undirected(0, 0)

        assert len(network._edges) == initial_edges
        assert not network.has_edge(0, 0)

    def test_add_edge_prevents_duplicates(self, basic_er_config):
        """Test that duplicate edges are not added"""
        network = Network(basic_er_config)

        # Clear existing edges
        network._edges.clear()
        network.adj = {u: [] for u in range(network.n)}

        network.add_edge_undirected(0, 1)
        initial_edges = len(network._edges)

        network.add_edge_undirected(0, 1)
        assert len(network._edges) == initial_edges

        network.add_edge_undirected(1, 0)
        assert len(network._edges) == initial_edges

    def test_neighbors(self, basic_er_config):
        """Test neighbors method"""
        network = Network(basic_er_config)

        # Clear and add specific edges
        network._edges.clear()
        network.adj = {u: [] for u in range(network.n)}

        network.add_edge_undirected(0, 1)
        network.add_edge_undirected(0, 2)

        neighbors_0 = network.neighbors(0)
        assert set(neighbors_0) == {1, 2}

        neighbors_1 = network.neighbors(1)
        assert neighbors_1 == [0]

    def test_degree(self, basic_er_config):
        """Test degree calculation"""
        network = Network(basic_er_config)

        # Clear and add specific edges
        network._edges.clear()
        network.adj = {u: [] for u in range(network.n)}

        network.add_edge_undirected(0, 1)
        network.add_edge_undirected(0, 2)
        network.add_edge_undirected(0, 3)

        assert network.degree(0) == 3
        assert network.degree(1) == 1
        assert network.degree(4) == 0

    def test_remove_edge(self, basic_er_config):
        """Test edge removal"""
        network = Network(basic_er_config)

        # Clear and add specific edges
        network._edges.clear()
        network.adj = {u: [] for u in range(network.n)}

        network.add_edge_undirected(0, 1)
        assert network.has_edge(0, 1)

        network.remove_edge(0, 1)
        assert not network.has_edge(0, 1)
        assert 1 not in network.neighbors(0)
        assert 0 not in network.neighbors(1)

    def test_remove_edge_both_directions(self, basic_er_config):
        """Test that edge removal works in both directions"""
        network = Network(basic_er_config)

        # Clear and add specific edges
        network._edges.clear()
        network.adj = {u: [] for u in range(network.n)}

        network.add_edge_undirected(0, 1)

        # Remove using reverse order
        network.remove_edge(1, 0)
        assert not network.has_edge(0, 1)

    def test_edges_iterator(self, basic_er_config):
        """Test edges iterator"""
        network = Network(basic_er_config)

        # Clear and add specific edges
        network._edges.clear()
        network.adj = {u: [] for u in range(network.n)}

        network.add_edge_undirected(0, 1)
        network.add_edge_undirected(1, 2)

        edges = list(network.edges())
        assert len(edges) == 2
        assert (0, 1) in edges or (1, 0) in edges
        assert (1, 2) in edges or (2, 1) in edges

    def test_er_deterministic_with_seed(self):
        """Test that ER generation is deterministic with same seed"""
        cfg1 = {
            "seed": 42,
            "network": {"generator": "ER", "params": {"n": 10, "p": 0.3}},
        }
        cfg2 = {
            "seed": 42,
            "network": {"generator": "ER", "params": {"n": 10, "p": 0.3}},
        }

        network1 = Network(cfg1)
        network2 = Network(cfg2)

        assert network1._edges == network2._edges

    def test_er_different_with_different_seed(self):
        """Test that ER generation differs with different seeds"""
        cfg1 = {
            "seed": 42,
            "network": {"generator": "ER", "params": {"n": 10, "p": 0.3}},
        }
        cfg2 = {
            "seed": 123,
            "network": {"generator": "ER", "params": {"n": 10, "p": 0.3}},
        }

        network1 = Network(cfg1)
        network2 = Network(cfg2)

        # With different seeds, networks should likely be different
        # (there's a tiny chance they could be the same, but very unlikely)
        assert network1._edges != network2._edges


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
