"""
Tests for Network class
"""

import pytest

from src.core.network import Network


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


class TestSeedRemap:
    """Test the seed_remap function"""

    @pytest.fixture
    def seed_maps(self):
        """Load seed mapping files"""
        from pathlib import Path
        seeds_dir = Path(__file__).parent.parent / "data" / "resources" / "seeds"

        # Load master seeds
        with (seeds_dir / "seeds.txt").open("r") as f:
            master_seeds = [int(line.strip()) for line in f if line.strip()]

        # Load ER seeds
        with (seeds_dir / "seeds_er.txt").open("r") as f:
            er_seeds = [int(line.strip()) for line in f if line.strip()]

        # Load WS seeds
        with (seeds_dir / "seeds_ws.txt").open("r") as f:
            ws_seeds = [int(line.strip()) for line in f if line.strip()]

        return {
            "er": dict(zip(master_seeds, er_seeds, strict=False)),
            "ws": dict(zip(master_seeds, ws_seeds, strict=False)),
            "master": dict(zip(master_seeds, master_seeds, strict=False)),
        }

    def test_seed_remap_with_none(self, seed_maps):
        """Test that None seed returns None"""
        from src.core.network import seed_remap

        result = seed_remap(None, seed_maps["er"])
        assert result is None

    def test_seed_remap_er_generator(self, seed_maps):
        """Test seed remapping for ER generator"""
        from src.core.network import seed_remap

        # Use a seed that exists in seeds.txt (first 8 seeds are mapped to ER)
        seed = 814183  # First seed in seeds.txt
        remapped = seed_remap(seed, seed_maps["er"])

        # Should be remapped to a seed from seeds_er.txt
        assert remapped != seed
        assert isinstance(remapped, int)

    def test_seed_remap_non_er_generator(self, seed_maps):
        """Test seed remapping for non-ER generator"""
        from src.core.network import seed_remap

        # Use a seed that exists in seeds.txt
        seed = 814183  # First seed in seeds.txt
        remapped = seed_remap(seed, seed_maps["master"])

        # For identity mapping, should return same seed
        assert remapped == seed
        assert isinstance(remapped, int)

    def test_seed_remap_missing_seed_uses_original(self, seed_maps):
        """Test that missing seed in lookup uses original seed"""
        from src.core.network import seed_remap

        # Use a seed that's not in the lookup
        seed = 999999999
        remapped = seed_remap(seed, seed_maps["er"])

        # Should return original seed
        assert remapped == seed

    def test_seed_remap_deterministic(self, seed_maps):
        """Test that seed remapping is deterministic"""
        from src.core.network import seed_remap

        seed = 814183
        result1 = seed_remap(seed, seed_maps["er"])
        result2 = seed_remap(seed, seed_maps["er"])

        assert result1 == result2


class TestNetworkSeedRemapping:
    """Test that Network uses remapped seed correctly"""

    @pytest.fixture
    def er_seed_map(self):
        """Load ER seed mapping"""
        from pathlib import Path
        seeds_dir = Path(__file__).parent.parent / "data" / "resources" / "seeds"

        with (seeds_dir / "seeds.txt").open("r") as f:
            master_seeds = [int(line.strip()) for line in f if line.strip()]

        with (seeds_dir / "seeds_er.txt").open("r") as f:
            er_seeds = [int(line.strip()) for line in f if line.strip()]

        return dict(zip(master_seeds, er_seeds, strict=False))

    def test_network_stores_remapped_seed(self, er_seed_map):
        """Test that Network stores the remapped seed"""
        # Use a seed from seeds.txt that will be remapped for ER
        cfg = {
            "seed": 814183,  # First seed in seeds.txt
            "network": {"generator": "ER", "params": {"n": 5, "p": 0.3}, "seed_map": er_seed_map},
        }

        network = Network(cfg)

        # Network should store the remapped seed
        assert network.seed != 814183  # Should be remapped
        assert isinstance(network.seed, int)

    def test_network_uses_remapped_seed_for_rng(self, er_seed_map):
        """Test that Network RNG uses remapped seed for determinism"""
        cfg1 = {
            "seed": 814183,  # Will be remapped
            "network": {"generator": "ER", "params": {"n": 10, "p": 0.3}, "seed_map": er_seed_map},
        }
        cfg2 = {
            "seed": 814183,  # Same original seed
            "network": {"generator": "ER", "params": {"n": 10, "p": 0.3}, "seed_map": er_seed_map},
        }

        network1 = Network(cfg1)
        network2 = Network(cfg2)

        # Networks should be identical because they use same remapped seed
        assert network1._edges == network2._edges
        assert network1.seed == network2.seed

    def test_network_with_none_seed(self):
        """Test that Network handles None seed gracefully"""
        cfg = {
            "seed": None,
            "network": {"generator": "ER", "params": {"n": 5, "p": 0.3}},
        }

        network = Network(cfg)

        # Should handle None seed
        assert network.seed is None

    def test_network_different_generators_different_remapping(self, er_seed_map):
        """Test that different generators can remap to different seeds"""
        from pathlib import Path

        from src.core.network import seed_remap

        seeds_dir = Path(__file__).parent.parent / "data" / "resources" / "seeds"
        with (seeds_dir / "seeds.txt").open("r") as f:
            master_seeds = [int(line.strip()) for line in f if line.strip()]
        with (seeds_dir / "seeds_ws.txt").open("r") as f:
            ws_seeds = [int(line.strip()) for line in f if line.strip()]
        ws_seed_map = dict(zip(master_seeds, ws_seeds, strict=False))

        seed = 814183
        er_remapped = seed_remap(seed, er_seed_map)
        ws_remapped = seed_remap(seed, ws_seed_map)

        # ER and WS generators should remap to different seeds
        assert isinstance(er_remapped, int)
        assert isinstance(ws_remapped, int)
        assert er_remapped != ws_remapped


class TestERGraphProperties:
    """Test Erdős-Rényi graph properties to confirm correct implementation."""

    def test_er_edge_count_p_zero(self):
        """Test ER with p=0 produces no edges."""
        cfg = {
            "seed": 42,
            "network": {"generator": "ER", "params": {"n": 10, "p": 0.0}},
        }
        network = Network(cfg)

        assert len(network._edges) == 0
        assert all(network.degree(i) == 0 for i in range(10))

    def test_er_edge_count_p_one(self):
        """Test ER with p=1 produces complete graph."""
        cfg = {
            "seed": 42,
            "network": {"generator": "ER", "params": {"n": 10, "p": 1.0}},
        }
        network = Network(cfg)

        expected_edges = 10 * 9 // 2  # Complete graph: n(n-1)/2
        assert len(network._edges) == expected_edges
        assert all(network.degree(i) == 9 for i in range(10))

    def test_er_edge_probability_distribution(self):
        """Test ER edge count follows expected distribution."""
        import numpy as np

        n = 50
        p = 0.3
        num_trials = 5

        edge_counts = []
        for trial in range(num_trials):
            cfg = {
                "seed": trial,
                "network": {"generator": "ER", "params": {"n": n, "p": p}},
            }
            network = Network(cfg)
            edge_counts.append(len(network._edges))

        # Expected number of edges: p * n(n-1)/2
        expected_edges = p * n * (n - 1) / 2

        # Mean should be close to expected (within reasonable variance)
        mean_edges = np.mean(edge_counts)
        # Allow for statistical variation (use generous bounds)
        assert 0.5 * expected_edges < mean_edges < 1.5 * expected_edges

    def test_er_no_self_loops(self):
        """Test ER graph contains no self-loops."""
        cfg = {
            "seed": 42,
            "network": {"generator": "ER", "params": {"n": 20, "p": 0.5}},
        }
        network = Network(cfg)

        # Check no self-loops
        for u, v in network.edges():
            assert u != v

        # Check diagonal of adjacency matrix
        adj_matrix = network.adjacency_matrix()
        import numpy as np

        assert np.all(np.diag(adj_matrix) == 0)

    def test_er_undirected_graph(self):
        """Test ER graph is undirected (symmetric adjacency)."""
        cfg = {
            "seed": 42,
            "network": {"generator": "ER", "params": {"n": 20, "p": 0.4}},
        }
        network = Network(cfg)

        # Check symmetry in adjacency list
        for u in range(network.n):
            for v in network.neighbors(u):
                assert u in network.neighbors(v)

        # Check adjacency matrix symmetry
        adj_matrix = network.adjacency_matrix()
        import numpy as np

        assert np.allclose(adj_matrix, adj_matrix.T)

    def test_er_degree_distribution(self):
        """Test ER degree distribution roughly follows binomial distribution."""
        import numpy as np

        n = 100
        p = 0.2
        cfg = {
            "seed": 42,
            "network": {"generator": "ER", "params": {"n": n, "p": p}},
        }
        network = Network(cfg)

        degrees = [network.degree(i) for i in range(n)]

        # For ER graph, expected degree is (n-1)*p
        expected_degree = (n - 1) * p
        mean_degree = np.mean(degrees)

        # Mean should be close to expected (within reasonable bounds)
        assert 0.5 * expected_degree < mean_degree < 1.5 * expected_degree

        # Variance should be roughly (n-1)*p*(1-p)
        expected_variance = (n - 1) * p * (1 - p)
        actual_variance = np.var(degrees)

        # Allow for reasonable variance in variance estimate
        assert 0.2 * expected_variance < actual_variance < 2.0 * expected_variance

    def test_er_connected_component_high_p(self):
        """Test ER with high p produces connected graph."""

        from src.core.metrics.network import connected_components

        # With p=0.5 and n=30, graph should be connected with high probability
        cfg = {
            "seed": 42,
            "network": {"generator": "ER", "params": {"n": 30, "p": 0.5}},
        }
        network = Network(cfg)

        adj_matrix = network.adjacency_matrix()
        components = connected_components(adj_matrix)

        # Should have 1 connected component (or at most 2-3 with small p)
        assert len(components) <= 3


class TestWSGraphProperties:
    """Test Watts-Strogatz graph properties to confirm correct implementation."""

    def test_ws_init_basic(self):
        """Test basic WS network initialization."""
        cfg = {
            "seed": 42,
            "network": {"generator": "WS", "params": {"n": 10, "k": 4, "beta": 0.0}},
        }
        network = Network(cfg)

        assert network.generator == "WS"
        assert network.n == 10
        assert network.k == 4
        assert network.beta == 0.0
        assert len(network.nodes) == 10

    def test_ws_missing_parameters(self):
        """Test that WS requires k and beta parameters."""
        cfg = {"seed": 42, "network": {"generator": "WS", "params": {"n": 10}}}
        with pytest.raises(
            ValueError, match="WS generator requires parameters k and beta"
        ):
            Network(cfg)

        cfg = {"seed": 42, "network": {"generator": "WS", "params": {"n": 10, "k": 4}}}
        with pytest.raises(
            ValueError, match="WS generator requires parameters k and beta"
        ):
            Network(cfg)

    def test_ws_k_must_be_even(self):
        """Test that k must be even for WS."""
        cfg = {
            "seed": 42,
            "network": {"generator": "WS", "params": {"n": 10, "k": 3, "beta": 0.0}},
        }
        with pytest.raises(ValueError, match="k must be even"):
            Network(cfg)

    def test_ws_beta_in_range(self):
        """Test that beta must be in [0,1]."""
        cfg = {
            "seed": 42,
            "network": {"generator": "WS", "params": {"n": 10, "k": 4, "beta": 1.5}},
        }
        with pytest.raises(ValueError, match="beta must be in"):
            Network(cfg)

        cfg = {
            "seed": 42,
            "network": {"generator": "WS", "params": {"n": 10, "k": 4, "beta": -0.1}},
        }
        with pytest.raises(ValueError, match="beta must be in"):
            Network(cfg)

    def test_ws_beta_zero_ring_lattice(self):
        """Test WS with beta=0 produces regular ring lattice."""
        n = 20
        k = 4
        cfg = {
            "seed": 42,
            "network": {"generator": "WS", "params": {"n": n, "k": k, "beta": 0.0}},
        }
        network = Network(cfg)

        # All nodes should have exactly k neighbors
        for i in range(n):
            assert network.degree(i) == k

        # Total edges should be n*k/2 (ring lattice)
        expected_edges = n * k // 2
        assert len(network._edges) == expected_edges

        # Each node should be connected to k/2 nodes on each side
        half_k = k // 2
        for i in range(n):
            neighbors_set = set(network.neighbors(i))
            expected_neighbors = {(i + d) % n for d in range(1, half_k + 1)}
            expected_neighbors.update({(i - d) % n for d in range(1, half_k + 1)})
            assert neighbors_set == expected_neighbors

    def test_ws_beta_one_random_rewiring(self):
        """Test WS with beta=1 rewires all edges."""
        n = 20
        k = 4
        cfg = {
            "seed": 42,
            "network": {"generator": "WS", "params": {"n": n, "k": k, "beta": 1.0}},
        }
        network = Network(cfg)

        # Should still have n*k/2 edges (rewiring preserves edge count)
        expected_edges = n * k // 2
        assert len(network._edges) == expected_edges

        # With beta=1, the network should differ from ring lattice
        # Create ring lattice for comparison
        cfg_ring = {
            "seed": 42,
            "network": {"generator": "WS", "params": {"n": n, "k": k, "beta": 0.0}},
        }
        network_ring = Network(cfg_ring)

        # Networks should have different edge sets
        assert network._edges != network_ring._edges

    def test_ws_no_self_loops(self):
        """Test WS graph contains no self-loops."""
        cfg = {
            "seed": 42,
            "network": {"generator": "WS", "params": {"n": 20, "k": 4, "beta": 0.5}},
        }
        network = Network(cfg)

        # Check no self-loops
        for u, v in network.edges():
            assert u != v

    def test_ws_undirected_graph(self):
        """Test WS graph is undirected."""
        cfg = {
            "seed": 42,
            "network": {"generator": "WS", "params": {"n": 20, "k": 4, "beta": 0.5}},
        }
        network = Network(cfg)

        # Check symmetry in adjacency list
        for u in range(network.n):
            for v in network.neighbors(u):
                assert u in network.neighbors(v)

        # Check adjacency matrix symmetry
        adj_matrix = network.adjacency_matrix()
        import numpy as np

        assert np.allclose(adj_matrix, adj_matrix.T)

    def test_ws_preserves_edge_count_after_rewiring(self):
        """Test WS preserves number of edges after rewiring."""
        n = 30
        k = 6
        expected_edges = n * k // 2

        for beta in [0.0, 0.2, 0.5, 0.8, 1.0]:
            cfg = {
                "seed": 42,
                "network": {
                    "generator": "WS",
                    "params": {"n": n, "k": k, "beta": beta},
                },
            }
            network = Network(cfg)
            assert len(network._edges) == expected_edges

    def test_ws_deterministic_with_seed(self):
        """Test WS generation is deterministic with same seed."""
        cfg1 = {
            "seed": 100,
            "network": {"generator": "WS", "params": {"n": 20, "k": 4, "beta": 0.3}},
        }
        cfg2 = {
            "seed": 100,
            "network": {"generator": "WS", "params": {"n": 20, "k": 4, "beta": 0.3}},
        }

        network1 = Network(cfg1)
        network2 = Network(cfg2)

        assert network1._edges == network2._edges

    def test_ws_different_with_different_seed(self):
        """Test WS generation differs with different seeds (for beta > 0)."""
        cfg1 = {
            "seed": 100,
            "network": {"generator": "WS", "params": {"n": 20, "k": 4, "beta": 0.5}},
        }
        cfg2 = {
            "seed": 200,
            "network": {"generator": "WS", "params": {"n": 20, "k": 4, "beta": 0.5}},
        }

        network1 = Network(cfg1)
        network2 = Network(cfg2)

        # With different seeds and beta > 0, networks should differ
        assert network1._edges != network2._edges

    def test_ws_clustering_higher_than_random(self):
        """Test WS with low beta has higher clustering than random graph."""

        from src.core.metrics.network import global_clustering_coefficient

        n = 50
        k = 6
        p_random = k / (n - 1)  # Equivalent edge density

        # WS with low beta (high clustering)
        cfg_ws = {
            "seed": 42,
            "network": {"generator": "WS", "params": {"n": n, "k": k, "beta": 0.1}},
        }
        network_ws = Network(cfg_ws)

        # Random graph (ER) with similar edge density
        cfg_er = {
            "seed": 42,
            "network": {"generator": "ER", "params": {"n": n, "p": p_random}},
        }
        network_er = Network(cfg_er)

        adj_ws = network_ws.adjacency_matrix()
        adj_er = network_er.adjacency_matrix()

        clustering_ws = global_clustering_coefficient(adj_ws)
        clustering_er = global_clustering_coefficient(adj_er)

        # WS should have higher clustering than random graph
        assert clustering_ws > clustering_er

    def test_ws_path_length_small_world(self):
        """Test WS with intermediate beta has small average path length."""

        from src.core.metrics.network import mean_shortest_path

        n = 50
        k = 6

        # Ring lattice (beta=0): high path length
        cfg_ring = {
            "seed": 42,
            "network": {"generator": "WS", "params": {"n": n, "k": k, "beta": 0.0}},
        }
        network_ring = Network(cfg_ring)

        # Small-world (beta=0.3): lower path length
        cfg_sw = {
            "seed": 42,
            "network": {"generator": "WS", "params": {"n": n, "k": k, "beta": 0.3}},
        }
        network_sw = Network(cfg_sw)

        adj_ring = network_ring.adjacency_matrix()
        adj_sw = network_sw.adjacency_matrix()

        path_ring = mean_shortest_path(adj_ring, directed=False, unweighted=True)
        path_sw = mean_shortest_path(adj_sw, directed=False, unweighted=True)

        # Small-world should have shorter average path length than ring
        assert path_sw < path_ring

    def test_ws_connected_graph(self):
        """Test WS produces connected graph."""

        from src.core.metrics.network import connected_components

        cfg = {
            "seed": 42,
            "network": {"generator": "WS", "params": {"n": 30, "k": 4, "beta": 0.5}},
        }
        network = Network(cfg)

        adj_matrix = network.adjacency_matrix()
        components = connected_components(adj_matrix)

        # Should have 1 connected component
        assert len(components) == 1
        assert len(components[0]) == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
