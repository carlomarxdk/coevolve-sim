"""Tests for experiment remapping behavior."""

import random

import pytest

from src.core.network import Network
from src.experiment import build_node_assignment, should_remap_agents


@pytest.mark.parametrize("catalog_name", ["experts", "random_experts"])
def test_should_remap_agents_for_ws_expert_catalogs(catalog_name):
    """Both experts variants should enable remapping on Watts-Strogatz."""
    choices = {"network": "watts-strogatz", "catalog": catalog_name}

    assert should_remap_agents(choices) is True


@pytest.mark.parametrize("network_name", ["ER", "barabasi-albert", "complete"])
def test_should_not_remap_agents_for_non_ws(network_name):
    """Non-WS networks should not enable agent remapping."""
    choices = {"network": network_name, "catalog": "experts"}

    assert should_remap_agents(choices) is False


def test_random_experts_remaps_like_experts_for_same_seed():
    """Random experts and experts should produce the same node-agent remap logic."""
    seed = 123
    agent_catalog = [{"id": i, "role": f"role-{i}"} for i in range(12)]
    network = Network(
        {
            "seed": seed,
            "network": {"generator": "ER", "params": {"n": 12, "p": 0.4}},
        }
    )

    remap_for_experts = should_remap_agents(
        {"network": "watts-strogatz", "catalog": "experts"}
    )
    remap_for_random_experts = should_remap_agents(
        {"network": "watts-strogatz", "catalog": "random_experts"}
    )

    experts_node_to_agent, experts_agent_to_node = build_node_assignment(
        agent_catalog=agent_catalog,
        network=network,
        seed=seed,
        remap=remap_for_experts,
    )
    random_experts_node_to_agent, random_experts_agent_to_node = build_node_assignment(
        agent_catalog=agent_catalog,
        network=network,
        seed=seed,
        remap=remap_for_random_experts,
    )

    assert experts_node_to_agent == random_experts_node_to_agent
    assert experts_agent_to_node == random_experts_agent_to_node

    # Verify deterministic remapping order for the chosen seed.
    expected_agent_ids = list(range(12))
    random.Random(seed).shuffle(expected_agent_ids)
    expected_node_to_agent = {
        node_id: agent_id
        for node_id, agent_id in zip(range(12), expected_agent_ids, strict=True)
    }

    assert experts_node_to_agent == expected_node_to_agent
