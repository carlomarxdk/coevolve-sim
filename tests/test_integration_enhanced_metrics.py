"""
Integration test to verify the enhanced metrics tracking works end-to-end
"""

import json
import pathlib
import tempfile
from unittest.mock import Mock

from src.Agent import BaseAgent
from src.MetricsTracker import MetricsTracker
from src.Network import Network
from utils import IOManager


def test_enhanced_metrics_integration():
    """Test the complete enhanced metrics tracking workflow"""
    with tempfile.TemporaryDirectory() as tmpdir:
        import os

        original_cwd = pathlib.Path.cwd()
        try:
            os.chdir(tmpdir)

            # Setup
            cfg = {
                "seed": 42,
                "network": {"generator": "ER", "params": {"n": 5, "p": 0.6}},
            }

            experiment_cfg = {
                "statement": {"id": "test1", "statement": "Test statement"},
                "seed": 42,
                "agents": {
                    "catalog": [
                        {"id": 0, "name": "model1", "role": "Expert"},
                        {"id": 1, "name": "model1", "role": "Expert"},
                        {"id": 2, "name": "model2", "role": "LLM"},
                        {"id": 3, "name": "model2", "role": "LLM"},
                        {"id": 4, "name": "model1", "role": "Expert"},
                    ]
                },
            }

            # Create IO manager
            io = IOManager({}, experiment_cfg=experiment_cfg)

            # Create network
            network = Network(cfg)

            # Create metrics tracker
            metrics = MetricsTracker({}, io)

            # Create mock agents with proper attributes
            agents = {}
            for i in range(5):
                agent = Mock(spec=BaseAgent)
                agent.id = i
                agent.model_name = experiment_cfg["agents"]["catalog"][i]["name"]
                agent.role = experiment_cfg["agents"]["catalog"][i]["role"]
                agent.beliefs = {}
                agent._belief_score = {}
                agent.current_message = []

                # Set initial beliefs
                agent.beliefs[0] = 0.5
                agent._belief_score[0] = 0.7
                agent.current_belief = Mock(
                    side_effect=lambda t, beliefs=agent.beliefs: beliefs.get(t)
                )
                agent.current_belief_score = Mock(
                    side_effect=lambda t, scores=agent._belief_score: scores.get(t)
                )

                agents[i] = agent

            # Register experiment (this creates agent_register and exp_register)
            metrics.register_experiment(agents, network)

            # Verify agent register was created
            assert len(metrics._agents_register) == 5
            for agent_id, agent_info in metrics._agents_register.items():
                assert "agent_id" in agent_info
                assert "role" in agent_info
                assert "model" in agent_info
                assert "network" in agent_info
                assert "degree" in agent_info["network"]
                assert "betweenness_centrality" in agent_info["network"]
                assert "eigenvector_centrality" in agent_info["network"]
                assert "clustering_coefficient" in agent_info["network"]
                assert "triangles" in agent_info["network"]
                assert "k_core" in agent_info["network"]

            # Verify experiment register
            assert "graph" in metrics._exp_register
            assert "agents" in metrics._exp_register
            graph_metrics = metrics._exp_register["graph"]
            assert "num_nodes" in graph_metrics
            assert "num_edges" in graph_metrics
            assert "radius" in graph_metrics
            assert "diameter" in graph_metrics
            assert "degrees" in graph_metrics
            assert "triangles" in graph_metrics

            # Record agent updates for round 0
            for agent in agents.values():
                metrics.update_agent_records(
                    agent=agent,
                    t=0,
                    new_belief=agent.current_belief(0),
                    new_score=agent.current_belief_score(0),
                    neighbor_view=None,
                )

            # Record round 0
            metrics.record_round(0, network)

            # Verify per_round has the fields
            assert len(metrics.per_round) == 1
            round_0 = metrics.per_round[0]

            # Check basic fields
            assert "belief" in round_0
            assert "label" in round_0["belief"]
            assert "score" in round_0["belief"]
            assert "values" in round_0["belief"]["label"]
            assert len(round_0["belief"]["label"]["values"]) == 5

            # Simulate round 1 with neighbor information
            # Update agent beliefs for round 1
            for i, agent in agents.items():
                agent.beliefs[1] = 0.6 + i * 0.05
                agent._belief_score[1] = 0.75 + i * 0.03

            # Record updates with neighbor metadata
            for i, agent in agents.items():
                neighbors = network.neighbors(i)
                neighbor_beliefs = {nb: agents[nb].beliefs[0] for nb in neighbors}

                metrics.update_agent_records(
                    agent=agent,
                    t=1,
                    new_belief=agent.beliefs[1],
                    new_score=agent._belief_score[1],
                    neighbor_view=neighbor_beliefs,
                )

            # Record round 1
            metrics.record_round(1, network)

            # Verify per_update has neighbor information
            updates_round_1 = [u for u in metrics.per_update if u["round"] == 1]
            assert len(updates_round_1) == 5
            for update in updates_round_1:
                assert "neighbor_info" in update
                assert "num_neighbors" in update["neighbor_info"]
                assert "n_agree" in update["neighbor_info"]
                assert "n_disagree" in update["neighbor_info"]
                assert "n_neutral" in update["neighbor_info"]
                assert "roles_agree" in update["neighbor_info"]
                assert "roles_disagree" in update["neighbor_info"]
                assert "roles_neutral" in update["neighbor_info"]
                assert "belief" in update
                assert "score" in update

            # Finalize metrics
            metrics.finalize(agents, network)

            # Save artifacts
            io.save_artifacts(agents, network, metrics)

            # Verify files were created
            results_dir = io.out_dir / "results"
            assert results_dir.exists()

            # Check that files exist (based on current save_artifacts implementation)
            assert (results_dir / "per_round_metrics.json").exists()
            assert (results_dir / "final_metrics.json").exists()
            assert (results_dir / "agents_data.json").exists()
            assert (results_dir / "network_edges.json").exists()
            assert (results_dir / "agent_manifest.json").exists()
            assert (results_dir / "network_manifest.json").exists()

            # Load and verify per_round_metrics.json structure
            with open(results_dir / "per_round_metrics.json") as f:
                per_round_data = json.load(f)
                assert "per_round" in per_round_data
                assert len(per_round_data["per_round"]) == 2  # rounds 0 and 1

                # Verify round has new fields
                for round_data in per_round_data["per_round"]:
                    assert "belief" in round_data
                    assert "label" in round_data["belief"]
                    assert "score" in round_data["belief"]

            # Load and verify agent_manifest.json
            with open(results_dir / "agent_manifest.json") as f:
                agent_manifest = json.load(f)
                assert len(agent_manifest) == 5

                for agent_id, agent_info in agent_manifest.items():
                    assert "agent_id" in agent_info
                    assert "role" in agent_info
                    assert "model" in agent_info
                    assert "network" in agent_info

            # Load and verify network_manifest.json (experiment register)
            with open(results_dir / "network_manifest.json") as f:
                manifest = json.load(f)
                assert "graph" in manifest
                assert "agents" in manifest
                assert manifest["agents"]["count"] == 5

            # Load and verify agents_data.json
            with open(results_dir / "agents_data.json") as f:
                agents_data = json.load(f)
                assert len(agents_data) == 5

                for agent_id, agent_info in agents_data.items():
                    assert "beliefs" in agent_info
                    assert "belief_scores" in agent_info
                    assert "role" in agent_info
                    assert "model" in agent_info

            print("✓ Integration test passed!")

        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    test_enhanced_metrics_integration()
    print("All integration tests passed!")
