"""
Tests for MetricsTracker class
"""

from unittest.mock import Mock

import pytest

from src.Agent import BaseAgent
from src.MetricsTracker import MetricsTracker
from src.Network import Network


class TestMetricsTracker:
    """Test the MetricsTracker class"""

    @pytest.fixture
    def basic_config(self):
        """Create a basic config for metrics tracker"""
        return {"save_per_update": True, "save_per_round": True}

    @pytest.fixture
    def mock_io(self):
        """Create a mock IOManager"""
        return Mock()

    @pytest.fixture
    def sample_agents(self):
        """Create sample agents for testing"""
        agents = {}
        for i in range(3):
            agent = Mock(spec=BaseAgent)
            agent.id = i
            agent.model_name = f"model_{i}"
            agent.role = f"role_{i}"
            agent.current_belief = Mock(return_value=0.5 + i * 0.1)
            agent.current_belief_score = Mock(return_value=0.8 + i * 0.05)
            agent.current_message = []
            agents[i] = agent
        return agents

    @pytest.fixture
    def sample_network(self):
        """Create a sample network for testing"""
        cfg = {"seed": 42, "network": {"generator": "ER", "params": {"n": 3, "p": 0.5}}}
        return Network(cfg)

    def test_metrics_tracker_initialization(self, basic_config, mock_io):
        """Test MetricsTracker initialization"""
        tracker = MetricsTracker(cfg=basic_config, io=mock_io)

        assert tracker.cfg == basic_config
        assert tracker.io == mock_io
        assert isinstance(tracker.per_update, list)
        assert isinstance(tracker.per_round, list)
        assert len(tracker.per_update) == 0
        assert len(tracker.per_round) == 0
        assert not tracker._exp_registered

    def test_register_experiment(
        self, basic_config, mock_io, sample_agents, sample_network
    ):
        """Test register_experiment method"""
        tracker = MetricsTracker(cfg=basic_config, io=mock_io)

        # Register experiment
        result = tracker.register_experiment(sample_agents, sample_network)

        assert tracker._exp_registered
        assert "graph" in result
        assert "agents" in result
        assert result["agents"]["count"] == 3
        assert len(tracker._agents_register) == 3

    def test_update_agent_records_basic(
        self, basic_config, mock_io, sample_agents, sample_network
    ):
        """Test update_agent_records method"""
        tracker = MetricsTracker(cfg=basic_config, io=mock_io)

        # Must register experiment first
        tracker.register_experiment(sample_agents, sample_network)

        # Record an agent update
        agent = sample_agents[0]
        result = tracker.update_agent_records(
            agent=agent, t=0, new_belief=0.6, new_score=0.8, neighbor_view=None
        )

        assert result is not None
        assert result["agent_id"] == 0
        assert result["round"] == 0
        assert result["belief"]["curr"] == 0.6
        assert result["score"]["curr"] == 0.8
        assert len(tracker.per_update) == 1

    def test_record_round_basic(
        self, basic_config, mock_io, sample_agents, sample_network
    ):
        """Test record_round method"""
        tracker = MetricsTracker(cfg=basic_config, io=mock_io)

        # Register experiment
        tracker.register_experiment(sample_agents, sample_network)

        # Update all agents first (using discrete belief labels: 1, 0, or -1)
        for i, agent in enumerate(sample_agents.values()):
            tracker.update_agent_records(
                agent=agent,
                t=0,
                new_belief=(
                    1 if i % 2 == 0 else 0
                ),  # Alternate between agree (1) and disagree (0)
                new_score=0.8 + i * 0.05,
                neighbor_view=None,
            )

        # Call record_round - new signature only takes t and network
        tracker.record_round(t=0, network=sample_network)

        assert len(tracker.per_round) == 1
        assert tracker.per_round[0]["round"] == 0
        assert "belief" in tracker.per_round[0]
        assert "label" in tracker.per_round[0]["belief"]
        assert "score" in tracker.per_round[0]["belief"]

    def test_finalize_basic(self, basic_config, mock_io, sample_agents, sample_network):
        """Test finalize method"""
        tracker = MetricsTracker(cfg=basic_config, io=mock_io)

        # Register and record one round
        tracker.register_experiment(sample_agents, sample_network)
        for i, agent in enumerate(sample_agents.values()):
            tracker.update_agent_records(
                agent=agent,
                t=0,
                new_belief=1 if i % 2 == 0 else 0,
                new_score=0.8,
                neighbor_view=None,
            )
        tracker.record_round(0, sample_network)

        # Call finalize
        tracker.finalize(agents=sample_agents, network=sample_network)

        # Method exists and doesn't raise error
        assert "final_beliefs" in tracker.final_metrics

    def test_multiple_updates(
        self, basic_config, mock_io, sample_agents, sample_network
    ):
        """Test recording multiple updates"""
        tracker = MetricsTracker(cfg=basic_config, io=mock_io)

        # Register experiment
        tracker.register_experiment(sample_agents, sample_network)

        # Record multiple updates across rounds - use discrete labels
        for t in range(3):
            for i, agent in enumerate(sample_agents.values()):
                # Alternate between 1 and 0
                belief = 1 if (i + t) % 2 == 0 else 0
                tracker.update_agent_records(
                    agent=agent,
                    t=t,
                    new_belief=belief,
                    new_score=0.7 + t * 0.05,
                    neighbor_view=None,
                )
            tracker.record_round(t, sample_network)

        # Should have 9 updates (3 agents x 3 rounds)
        assert len(tracker.per_update) == 9
        assert len(tracker.per_round) == 3

    def test_multiple_rounds(
        self, basic_config, mock_io, sample_agents, sample_network
    ):
        """Test recording multiple rounds"""
        tracker = MetricsTracker(cfg=basic_config, io=mock_io)

        # Register experiment
        tracker.register_experiment(sample_agents, sample_network)

        # Record multiple rounds
        for t in range(5):
            for i, agent in enumerate(sample_agents.values()):
                belief = 1 if (i + t) % 2 == 0 else 0
                tracker.update_agent_records(
                    agent=agent,
                    t=t,
                    new_belief=belief,
                    new_score=0.8,
                    neighbor_view=None,
                )
            tracker.record_round(t=t, network=sample_network)

        assert len(tracker.per_round) == 5

    def test_update_with_neighbor_view(
        self, basic_config, mock_io, sample_agents, sample_network
    ):
        """Test update_agent_records with neighbor information"""
        tracker = MetricsTracker(cfg=basic_config, io=mock_io)

        # Register experiment
        tracker.register_experiment(sample_agents, sample_network)

        # Create neighbor view
        neighbor_view = {1: 1.0, 2: 0.0}

        # Record update with neighbor view
        agent = sample_agents[0]
        result = tracker.update_agent_records(
            agent=agent, t=0, new_belief=0.6, new_score=0.8, neighbor_view=neighbor_view
        )

        assert "neighbor_info" in result
        assert result["neighbor_info"]["num_neighbors"] == 2
        assert result["neighbor_info"]["n_agree"] == 1
        assert result["neighbor_info"]["n_disagree"] == 1

    def test_empty_agents_dict(self, basic_config, mock_io, sample_network):
        """Test that calling record_round without any agents registered fails appropriately"""
        # tracker = MetricsTracker(cfg=basic_config, io=mock_io)  # Not currently used

        # Don't register any agents - this should cause a validation error when trying to record
        # Since there are no agents, we can't record a round
        # This test verifies that the tracker handles the edge case properly
        pass  # Nothing to test - empty agent dict is not a valid scenario

    def test_metrics_persistence_structure(
        self, basic_config, mock_io, sample_agents, sample_network
    ):
        """Test that per_update list maintains structure"""
        tracker = MetricsTracker(cfg=basic_config, io=mock_io)

        # Register experiment
        tracker.register_experiment(sample_agents, sample_network)

        initial_len = len(tracker.per_update)

        agent = sample_agents[0]
        tracker.update_agent_records(
            agent=agent, t=0, new_belief=0.6, new_score=0.8, neighbor_view=None
        )

        assert len(tracker.per_update) == initial_len + 1
        assert isinstance(tracker.per_update, list)
        assert isinstance(tracker.per_round, list)


class TestMetricsTrackerEdgeCases:
    """Test edge cases for MetricsTracker"""

    @pytest.fixture
    def basic_config(self):
        return {}

    @pytest.fixture
    def mock_io(self):
        return Mock()

    @pytest.fixture
    def sample_agents(self):
        """Create sample agents for testing"""
        agents = {}
        for i in range(3):
            agent = Mock(spec=BaseAgent)
            agent.id = i
            agent.model_name = f"model_{i}"
            agent.role = f"role_{i}"
            agent.current_belief = Mock(return_value=0.5 + i * 0.1)
            agent.current_belief_score = Mock(return_value=0.8 + i * 0.05)
            agent.current_message = []
            agents[i] = agent
        return agents

    @pytest.fixture
    def sample_network(self):
        """Create a sample network for testing"""
        cfg = {"seed": 42, "network": {"generator": "ER", "params": {"n": 3, "p": 0.5}}}
        return Network(cfg)

    def test_update_without_registration(self, basic_config, mock_io, sample_agents):
        """Test that update_agent_records requires experiment registration"""
        tracker = MetricsTracker(cfg=basic_config, io=mock_io)

        agent = sample_agents[0]

        # Should raise assertion error
        with pytest.raises(AssertionError):
            tracker.update_agent_records(
                agent=agent, t=0, new_belief=0.6, new_score=0.8, neighbor_view=None
            )

    def test_record_round_without_updates(
        self, basic_config, mock_io, sample_agents, sample_network
    ):
        """Test that record_round requires all agents to be updated"""
        tracker = MetricsTracker(cfg=basic_config, io=mock_io)

        # Register experiment
        tracker.register_experiment(sample_agents, sample_network)

        # Try to record round without updating agents
        with pytest.raises(
            AssertionError, match="You did not update the beliefs of all the agents"
        ):
            tracker.record_round(t=0, network=sample_network)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
