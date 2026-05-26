from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.core.agent import ExpertAgent, LLMAgent
from src.core.inference_scheduler import InferenceScheduler
from src.core.message import Message
from src.core.metric_tracker import MetricsTracker
from src.core.network import Network
from src.core.probe import Probe, ZeroShotProbe
from src.utils import (
    IOManager,
    build_catalog,
    check_experiment_completed,
    get_experiment_choices,
    load_model_config,
    move_incomplete_experiments,
    set_seed,
)

log = logging.getLogger("Experiment")

LEGAL_ORDERINGS = ["random", "degree_desc", "by_model", "fixed"]


def should_remap_agents(choices: dict) -> bool:
    # Return True only for WS experts/random_experts runs.
    return choices.get("network") == "watts-strogatz" and choices.get("catalog") in {
        "experts",
        "random_experts",
    }


def build_node_assignment(
    agent_catalog: list[dict], network: Network, seed: int, remap: bool
) -> tuple[dict[int, int], dict[int, int]]:
    # Build node<->agent assignment mappings.
    catalog_ids = [int(spec["id"]) for spec in agent_catalog]
    node_ids = list(network.nodes)

    if len(catalog_ids) != len(node_ids):
        raise ValueError(
            f"Catalog size ({len(catalog_ids)}) does not match number of nodes ({len(node_ids)})"
        )

    assigned_agent_ids = list(catalog_ids)
    if remap:
        log.info(
            "Randomly remapping agents to nodes (necessary for WS experts/random_experts)."
        )
        rng = random.Random(seed)
        rng.shuffle(assigned_agent_ids)

    node_to_a_id = {
        node_id: a_id
        for node_id, a_id in zip(node_ids, assigned_agent_ids, strict=True)
    }
    a_id_to_node = {a_id: node_id for node_id, a_id in node_to_a_id.items()}

    return node_to_a_id, a_id_to_node


def order_agents(network: Network, ordering: str, agents: dict[int, Any]) -> list[int]:
    # randomly order nodes
    assert ordering in LEGAL_ORDERINGS, f"Unknown ordering: {ordering}"
    if ordering == "random":
        ids = list(network.nodes)
        random.shuffle(ids)
        return ids

    # order nodes by degree (largest first)
    if ordering == "degree_desc":
        deg = {u: len(network.neighbors(u)) for u in network.nodes}
        return sorted(network.nodes, key=lambda u: deg[u], reverse=True)

    if ordering == "by_model":
        groups = defaultdict(list)
        for node_id, agent in agents.items():
            groups[agent.model_name].append(int(node_id))
        ids = [node_id for group in groups.values() for node_id in group]
        return ids

    # order nodes by id
    return list(network.nodes)


def initialize_beliefs(agents, statement, scheduler, probe, io):
    for i, agent in agents.items():
        # assign initial statement to agents
        agent.set_statement(statement.get("statement"))

        # For zeroshot probe, get logits; otherwise get embeddings
        if isinstance(probe, ZeroShotProbe):
            # Set tokenizer for zeroshot probe if not already set
            if probe.tokenizer is None:
                # Get tokenizer from scheduler's model
                with scheduler.ensure_loaded(agent.model_name) as mdl:
                    if hasattr(mdl, "_tokenizer"):
                        probe.set_tokenizer(mdl._tokenizer)

            logits = scheduler.get_logits(agent, t=0, verbose=True)
            b_0, s_0, fs_0 = probe.score(logits, t=0)
            # Cache logits instead of activations (activation caching not implemented)
            io.cache_activation(agent.id, t=0, activation=logits)
        else:
            # Original behavior for sawmil probe
            act_0 = scheduler.embed(agent, t=0, verbose=True)
            b_0, s_0, fs_0 = probe.score(act_0, t=0)
            # Cache activations (activation caching not implemented)
            io.cache_activation(agent.id, t=0, activation=act_0)

        # Store complete scores along with belief and confidence score
        # Complete scores are logged in agents_data.json and per_update metrics
        agent.set_belief(label=b_0, t=0, score=s_0, complete_scores=fs_0)
        log.info(f"Agent ({i}) initial belief: {b_0} (score: {s_0}) at round 0")


@hydra.main(version_base=None, config_path="configs", config_name="experiment.yaml")
def main(cfg: DictConfig):
    # Check if experiment is already completed before initializing
    seed = cfg.get("seed", 814183)
    max_rounds = cfg.get("experiment", {}).get("max_rounds", 10)
    choices = get_experiment_choices(cfg)

    base_dir = (
        cfg.get("callbacks", {}).get("io", {}).get("out_dir", "data/outputs/runs")
    )

    # Check for completed experiment
    if check_experiment_completed(
        catalog_choice=choices["catalog"],
        prompt_choice=choices["prompt"],
        statement_choice=choices["statement"],
        probe_choice=choices["probe"],
        network_choice=choices["network"],
        seed=seed,
        max_rounds=max_rounds,
        base_dir=base_dir,
    ):
        log.info(
            "\n" + "=" * 60 + "\n"
            "Experiment already completed!\n"
            f"Catalog: {choices['catalog']}\n"
            f"Prompt: {choices['prompt']}\n"
            f"Statement: {choices['statement']}\n"
            f"Probe: {choices['probe']}\n"
            f"Network: {choices['network']}\n"
            f"Seed: {seed}\n"
            f"Max Rounds: {max_rounds}\n" + "=" * 60 + "\n"
            "Skipping experiment execution."
        )
        return

    # Move any incomplete experiments with same seed to incomplete_runs directory
    moved_count = move_incomplete_experiments(
        catalog_choice=choices["catalog"],
        prompt_choice=choices["prompt"],
        statement_choice=choices["statement"],
        probe_choice=choices["probe"],
        network_choice=choices["network"],
        seed=seed,
        max_rounds=max_rounds,
        base_dir=base_dir,
    )
    if moved_count > 0:
        log.info(
            f"\n{'=' * 60}\n"
            f"Moved {moved_count} incomplete experiment(s) to outputs/incomplete_runs/\n"
            f"{'=' * 60}\n"
        )

    # initialize experiment
    set_seed(seed)
    io = IOManager(cfg.get("callbacks", {}).get("io", {}), experiment_cfg=cfg)

    log.info(
        "\n" + "=" * 60 + "\n"
        f"Starting Experiment: {io.experiment_name}\n"
        f"Output Directory: {io.out_dir}\n" + "=" * 60 + "\n"
    )

    scheduler = InferenceScheduler(cfg)
    network = Network(cfg)

    # Update config with remapped network seed
    # cfg["seed"] = network.seed # experiment config should use the original seed (not the remapped one)

    metrics = MetricsTracker(cfg.get("callbacks", {}).get("metrics", {}), io)
    stopper = instantiate(cfg.get("callbacks").get("stopping"), _convert_="all")

    # initialize agents
    agent_catalog = build_catalog(cfg.get("agents", {}).get("catalog", {}), seed=seed)
    if len(agent_catalog) != network.n:
        raise ValueError(
            f"Catalog size ({len(agent_catalog)}) does not match network size ({network.n})"
        )
    cfg["agents"]["catalog_used"] = agent_catalog  # store the actual catalog used

    # Because of WS initialization, we must sometimes randomize mapping between a_ids and node ids
    catalog_by_id = {int(spec["id"]): spec for spec in agent_catalog}

    remap_agents = should_remap_agents(choices)
    node_to_a_id, a_id_to_node = build_node_assignment(
        agent_catalog=agent_catalog,
        network=network,
        seed=seed,
        remap=remap_agents,
    )

    log.info(f"Remap agents to nodes: {remap_agents}")
    if remap_agents:
        log.info(f"node_to_a_id mapping: {node_to_a_id}")

    # instantiate agents keyed by NODE ID
    agents = {}
    for node_id in network.nodes:
        a_id = node_to_a_id[node_id]
        spec = catalog_by_id[a_id]

        model = spec.get("name", "expert")
        model_cfg = load_model_config(model)
        cls = ExpertAgent if model == "expert" else LLMAgent

        # it pools information from the global 'cfg'
        message = Message(cfg=cfg)

        # Choose probe type based on configuration
        probe_cfg = cfg.get("probe", {})
        probe_name = probe_cfg.get("name", "zeroshot").lower()
        if probe_name == "zeroshot":
            probe = ZeroShotProbe(cfg=probe_cfg, model_cfg=model_cfg, io=io)
        else:
            probe = Probe(cfg=probe_cfg, model_cfg=model_cfg, io=io)

        agent = cls(
            id=node_id,  # runtime ID should be the network node ID
            model_name=model,
            role=spec.get("role"),
            message=message,
            probe=probe,
            cfg=model_cfg,
        )

        # Save original catalog identity on the runtime agent
        agent.catalog_id = a_id
        agent.node_id = node_id

        agents[node_id] = agent

    # Register experiment (agents and network metrics)
    metrics.register_experiment(agents=agents, network=network)

    # Save the updated config with remapped seed and roles
    io.update_and_save_config()

    # Set initial statement
    statement = cfg.get("statement", {})

    # ROUND 0: INITIALIZE BELIEFS
    initialize_beliefs(agents, statement, scheduler, probe, io)

    # Record initial agent updates
    for agent in agents.values():
        metrics.update_agent_records(
            agent=agent,
            t=0,
            new_belief=agent.current_belief(0),
            new_score=agent.current_belief_score(0),
            neighbor_view=None,
        )

    # Record initial round metrics
    metrics.record_round(0, network)
    io.checkpoint(0, agents, metrics)

    # update beliefs in rounds
    T = cfg.get("experiment", {}).get("max_rounds", 1000)
    ordering = cfg.get("experiment", {}).get("ordering", "fixed")
    t = 0
    while t < T:
        order = order_agents(network, ordering, agents)

        # Build neighbor views
        neighbor_views = {}
        for a_id, _agent in agents.items():
            a_id = int(a_id)
            neighbors = network.neighbors(a_id)
            beliefs_t = {j: agents[j].current_belief(t) for j in neighbors}
            neighbor_views[a_id] = beliefs_t

        # Sequentially compute and commit belief updates
        for a_id in order:
            a_id = int(a_id)
            agent = agents[a_id]

            # For zeroshot probe, get logits; otherwise get embeddings
            if isinstance(probe, ZeroShotProbe):
                # Get logits for the agent's current context
                logits_new = scheduler.get_logits(
                    agent, t=t + 1, neighbor_view=neighbor_views[a_id], verbose=True
                )

                # Generate updated beliefs
                b_new, s_new, fs_new = probe.score(logits_new, t=t + 1)

                # Cache logits
                io.cache_activation(a_id, t + 1, logits_new)
            else:
                # Original behavior for sawmil probe
                # Get new embeddings
                act_new = scheduler.embed(
                    agent, t=t + 1, neighbor_view=neighbor_views[a_id], verbose=True
                )

                # Generate updated beliefs
                b_new, s_new, fs_new = probe.score(act_new, t=t + 1)

                # Cache activation
                io.cache_activation(a_id, t + 1, act_new)

            # Set the new belief for this agent with complete scores
            agent.set_belief(label=b_new, t=t + 1, score=s_new, complete_scores=fs_new)

            # Record agent update with neighbor information
            metrics.update_agent_records(
                agent=agent,
                t=t + 1,
                new_belief=b_new,
                new_score=s_new,
                neighbor_view=neighbor_views[a_id],
            )

            log.info(
                f"Agent ({a_id}) updated belief: {b_new} (score: {s_new}) at round {t + 1}"
            )

        # Record round-level metrics
        metrics.record_round(t + 1, network)

        # Save checkpoint
        io.checkpoint(t + 1, agents, metrics)
        log.info(f"Checkpoint saved for round {t + 1} → {io.out_dir}")

        if stopper.check(metrics, t):
            break
        t += 1

    # Finalize metrics
    metrics.finalize(agents, network)
    io.save_artifacts(agents, network, metrics)


if __name__ == "__main__":
    main()
