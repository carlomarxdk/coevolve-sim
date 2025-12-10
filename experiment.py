from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, List

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.Agent import ExpertAgent, LLMAgent
from src.InferenceScheduler import InferenceScheduler
from src.Message import Message
from src.MetricsTracker import MetricsTracker
from src.Network import Network
from src.Probe import Probe, ZeroShotProbe
from utils import (
    IOManager,
    check_experiment_completed,
    get_experiment_choices,
    load_model_config,
    move_incomplete_experiments,
    set_seed,
)

log = logging.getLogger("Experiment")

LEGAL_ORDERINGS = ["random", "degree_desc", "by_model", "fixed"]


def order_agents(
    network: Network, ordering: str, agent_catalog: Dict = None
) -> List[int]:
    # randomly order nodes
    assert ordering in LEGAL_ORDERINGS, f"Unknown ordering: {ordering}"
    if ordering == "random":
        ids = list(network.nodes)
        random.shuffle(ids)
        return ids

    # order nodes by degree (largest first)
    elif ordering == "degree_desc":
        deg = {u: len(network.neighbors(u)) for u in network.nodes}
        return sorted(network.nodes, key=lambda u: deg[u], reverse=True)

    elif ordering == "by_model":
        groups = defaultdict(list)
        for spec in agent_catalog:
            a_id = int(spec["id"])
            model = spec.get("name")
            groups[model].append(a_id)
        ids = [a_id for ids in groups.values() for a_id in ids]

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
            b_0, s_0 = probe.score(logits, t=0)
            # Cache logits instead of activations
            io.cache_activation(agent.id, t=0, activation=logits)
        else:
            # Original behavior for sawmil probe
            act_0 = scheduler.embed(agent, t=0, verbose=True)
            b_0, s_0 = probe.score(act_0, t=0)
            io.cache_activation(agent.id, t=0, activation=act_0)

        agent.set_belief(label=b_0, t=0, score=s_0)
        log.info(f"Agent ({i}) initial belief: {b_0} (score: {s_0}) at round 0")


@hydra.main(version_base=None, config_path="configs", config_name="experiment.yaml")
def main(cfg: DictConfig):
    # Check if experiment is already completed before initializing
    seed = cfg.get("seed", 42)
    max_rounds = cfg.get("experiment", {}).get("max_rounds", 10)
    catalog_choice, prompt_choice, statement_choice, probe_choice = (
        get_experiment_choices(cfg)
    )
    base_dir = cfg.get("callbacks", {}).get("io", {}).get("out_dir", "outputs/runs")

    # Check for completed experiment
    if check_experiment_completed(
        catalog_choice,
        prompt_choice,
        statement_choice,
        probe_choice,
        seed,
        max_rounds,
        base_dir,
    ):
        log.info(
            "\n" + "=" * 60 + "\n"
            "Experiment already completed!\n"
            f"Catalog: {catalog_choice}\n"
            f"Prompt: {prompt_choice}\n"
            f"Statement: {statement_choice}\n"
            f"Probe: {probe_choice}\n"
            f"Seed: {seed}\n"
            f"Max Rounds: {max_rounds}\n" + "=" * 60 + "\n"
            "Skipping experiment execution."
        )
        return

    # Move any incomplete experiments with same seed to incomplete_runs directory
    moved_count = move_incomplete_experiments(
        catalog_choice,
        prompt_choice,
        statement_choice,
        probe_choice,
        seed,
        max_rounds,
        base_dir,
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
    metrics = MetricsTracker(cfg.get("callbacks", {}).get("metrics", {}), io)
    stopper = instantiate(cfg.get("callbacks").get("stopping"), _convert_="all")

    # initialize agents
    agent_catalog = cfg.get("agents", {}).get("catalog", [])
    agents = {}
    if cfg.get("agents", {}).get("random_roles"):
        log.info("Randomizing agent roles...")
        roles = [spec.get("role") for spec in agent_catalog]
        log.debug(f"Original roles: {roles}")
        random.shuffle(roles)
        for i, _ in enumerate(agent_catalog):
            agent_catalog[i]["role"] = roles[i]
            cfg.agents.catalog[i]["role"] = roles[i]
    for spec in agent_catalog:
        a_id = int(spec["id"])
        model = spec.get("name", "expert")
        model_cfg = load_model_config(model)
        cls = ExpertAgent if model == "expert" else LLMAgent
        # it pools information from the global 'cfg'
        message = Message(cfg=cfg)

        # Choose probe type based on configuration
        probe_cfg = cfg.get("probe", {})
        probe_name = probe_cfg.get("name", "sawmil")
        if probe_name == "zeroshot":
            probe = ZeroShotProbe(cfg=probe_cfg, model_cfg=model_cfg, io=io)
        else:
            probe = Probe(cfg=probe_cfg, model_cfg=model_cfg, io=io)

        agents[a_id] = cls(
            id=a_id,
            model_name=model,
            role=spec.get("role"),
            message=message,
            probe=probe,
            cfg=model_cfg,
        )  # where should we store role

    # Register experiment (agents and network metrics)
    metrics.register_experiment(agents=agents, network=network)

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
        order = order_agents(network, ordering, agent_catalog)

        # Build neighbor views
        neighbor_views = {}
        for a_id, agent in agents.items():
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
                b_new, s_new = probe.score(logits_new, t=t + 1)

                # Cache logits
                io.cache_activation(a_id, t + 1, logits_new)
            else:
                # Original behavior for sawmil probe
                # Get new embeddings
                act_new = scheduler.embed(
                    agent, t=t + 1, neighbor_view=neighbor_views[a_id], verbose=True
                )

                # Generate updated beliefs
                b_new, s_new = probe.score(act_new, t=t + 1)

                # Cache activation
                io.cache_activation(a_id, t + 1, act_new)

            # Set the new belief for this agent
            agent.set_belief(label=b_new, t=t + 1, score=s_new)

            # Record agent update with neighbor information
            metrics.update_agent_records(
                agent=agent,
                t=t + 1,
                new_belief=b_new,
                new_score=s_new,
                neighbor_view=neighbor_views[a_id],
            )

            log.info(
                f"Agent ({a_id}) updated belief: {b_new} (score: {s_new}) at round {t+1}"
            )

        # Record round-level metrics
        metrics.record_round(t + 1, network)

        # Save checkpoint
        io.checkpoint(t + 1, agents, metrics)
        log.info(f"Checkpoint saved for round {t+1} → {io.out_dir}")

        if stopper.check(metrics, t):
            break
        t += 1

    # Finalize metrics
    metrics.finalize(agents, network)
    io.save_artifacts(agents, network, metrics)


if __name__ == "__main__":
    main()
