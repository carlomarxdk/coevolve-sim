"""
Tests for numpy array serialization in checkpoint and save_artifacts functions.
"""

import json
import pathlib
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.core.metric_tracker import MetricsTracker
from src.core.network import Network
from src.utils import IOManager, convert_numpy_to_native


# Mock agent class for testing
@dataclass
class MockAgent:
    """Simplified mock agent for testing serialization."""
    id: int
    role: str = "test_role"
    model_name: str = "test_model"
    beliefs: dict[int, float] = field(default_factory=dict)
    _belief_score: dict[int, float] = field(default_factory=dict)
    _complete_scores: dict[int, list[float]] = field(default_factory=dict)
    current_message: str = ""

    def set_belief(self, label: float, t: int, score: float = None, complete_scores=None):
        """Set belief and scores for a given round."""
        if score is not None:
            self._belief_score[t] = float(score)
        if complete_scores is not None:
            self._complete_scores[t] = complete_scores
        self.beliefs[t] = float(label)

    def current_belief(self, t: int):
        """Get belief for round t."""
        return self.beliefs.get(t, None)

    def current_belief_score(self, t: int):
        """Get belief score for round t."""
        return self._belief_score.get(t, None)

    def current_complete_scores(self, t: int):
        """Get complete scores for round t."""
        return self._complete_scores.get(t, None)


def test_convert_numpy_to_native():
    """Test that convert_numpy_to_native handles various numpy types."""
    # Test numpy array
    arr = np.array([0.1, 0.2, 0.3])
    result = convert_numpy_to_native(arr)
    assert isinstance(result, list)
    assert result == [0.1, 0.2, 0.3]

    # Test numpy scalar
    scalar = np.float64(0.5)
    result = convert_numpy_to_native(scalar)
    assert isinstance(result, float)
    assert result == 0.5

    # Test nested dict with numpy arrays
    data = {
        "scores": np.array([1.0, 2.0, 3.0]),
        "meta": {"value": np.int32(42)},
    }
    result = convert_numpy_to_native(data)
    assert isinstance(result["scores"], list)
    assert isinstance(result["meta"]["value"], int)
    assert result["scores"] == [1.0, 2.0, 3.0]
    assert result["meta"]["value"] == 42

    # Test list with numpy arrays
    data = [np.array([1, 2]), np.array([3, 4])]
    result = convert_numpy_to_native(data)
    assert isinstance(result, list)
    assert all(isinstance(item, list) for item in result)
    assert result == [[1, 2], [3, 4]]


def test_checkpoint_with_numpy_arrays():
    """Test that checkpoint can handle agents with numpy array complete_scores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create IOManager
        experiment_cfg = {
            "statement": {"id": "test_stmt", "statement": "Test statement"},
            "prompt": {"type": "wR_L"},
            "seed": 42,
            "experiment": {"max_rounds": 10},
            "network": {"generator": "ER", "params": {"n": 3, "p": 0.5}},
        }

        cfg = {"save_activations": False, "save_text": True}
        io = IOManager(cfg, experiment_cfg=experiment_cfg)
        io.out_dir = pathlib.Path(tmpdir)

        # Create mock agents with numpy array complete_scores
        agents = {}
        for i in range(3):
            agent = MockAgent(id=i, role=f"role_{i}")
            # Set belief with numpy array complete_scores
            complete_scores = np.array([0.3, 0.5, 0.2])
            agent.set_belief(label=1.0, t=0, score=0.5, complete_scores=complete_scores)
            agents[i] = agent

        # Create mock metrics tracker
        network = Network(experiment_cfg)
        metrics = MetricsTracker(cfg={}, io=io)
        metrics.register_experiment(agents, network)

        # Record initial update for each agent
        for agent in agents.values():
            metrics.update_agent_records(
                agent=agent,
                t=0,
                new_belief=agent.current_belief(0),
                new_score=agent.current_belief_score(0),
                neighbor_view={},
            )

        # Test checkpoint - this should not raise an error
        io.checkpoint(0, agents, metrics)

        # Verify that beliefs.jsonl was created and is valid JSON
        beliefs_path = pathlib.Path(tmpdir) / "rounds" / "round_0" / "beliefs.jsonl"
        assert beliefs_path.exists()

        # Read and parse each line
        with open(beliefs_path, "r") as f:
            for line in f:
                data = json.loads(line)
                assert "complete_scores" in data
                # Verify it's a list, not a numpy array
                assert isinstance(data["complete_scores"], list)
                assert len(data["complete_scores"]) == 3

        # Verify beliefs_detailed.json is valid
        detailed_path = pathlib.Path(tmpdir) / "rounds" / "round_0" / "beliefs_detailed.json"
        assert detailed_path.exists()
        with open(detailed_path, "r") as f:
            data = json.load(f)
            assert isinstance(data, list)
            for record in data:
                if "complete_scores" in record and record["complete_scores"]["curr"] is not None:
                    assert isinstance(record["complete_scores"]["curr"], list)


def test_save_artifacts_with_numpy_arrays():
    """Test that save_artifacts can handle agents with numpy array complete_scores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create IOManager
        experiment_cfg = {
            "statement": {"id": "test_stmt", "statement": "Test statement"},
            "prompt": {"type": "wR_L"},
            "seed": 42,
            "experiment": {"max_rounds": 10},
            "network": {"generator": "ER", "params": {"n": 3, "p": 0.5}},
        }

        cfg = {"save_activations": False, "save_text": True}
        io = IOManager(cfg, experiment_cfg=experiment_cfg)
        io.out_dir = pathlib.Path(tmpdir)

        # Create mock agents with numpy array complete_scores
        agents = {}
        for i in range(3):
            agent = MockAgent(id=i, role=f"role_{i}", model_name="test_model")
            # Set multiple rounds with numpy arrays
            for t in range(3):
                complete_scores = np.array([0.3 * t, 0.5 * t, 0.2 * t])
                agent.set_belief(label=float(t), t=t, score=0.5 * t, complete_scores=complete_scores)
            agents[i] = agent

        # Create mock metrics tracker and network
        network = Network(experiment_cfg)
        metrics = MetricsTracker(cfg={}, io=io)
        metrics.register_experiment(agents, network)

        # Record updates for all agents at round 0
        for agent in agents.values():
            metrics.update_agent_records(
                agent=agent,
                t=0,
                new_belief=agent.current_belief(0),
                new_score=agent.current_belief_score(0),
                neighbor_view={},
            )
        metrics.record_round(0, network)

        # Test save_artifacts - this should not raise an error
        io.save_artifacts(agents, network, metrics)

        # Verify agents_data.json is valid
        agents_path = pathlib.Path(tmpdir) / "results" / "agents_data.json"
        assert agents_path.exists()
        with open(agents_path, "r") as f:
            data = json.load(f)
            for agent_id, agent_info in data.items():
                assert "complete_scores" in agent_info
                # Each round should have a list, not numpy array
                for t, scores in agent_info["complete_scores"].items():
                    assert isinstance(scores, list), f"Expected list but got {type(scores)}"
                    assert len(scores) == 3


if __name__ == "__main__":
    test_convert_numpy_to_native()
    test_checkpoint_with_numpy_arrays()
    test_save_artifacts_with_numpy_arrays()
    print("All tests passed!")
