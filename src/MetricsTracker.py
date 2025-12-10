"""
MetricsTracker - Comprehensive metrics tracking for LLM debate simulations.

This module tracks and computes three types of metrics:
    1. Per-update metrics: Individual agent belief changes
    2. Per-round metrics: Aggregate statistics for each simulation round
    3. Final metrics: Post-run summary including consensus, clustering, and network analysis

The tracker also computes influence and information metrics including:
    - Pairwise mutual information
    - Transfer entropy estimates
    - Influence centrality (correlation-based)
    - Neighbor influence proxies

Optimized for small networks (N<=16) with exact computations for triangles and shortest paths.
"""

import math
from collections import Counter, deque
from typing import Any, Dict, List, Optional

import numpy as np

from src.metrics.general import (
    belief_change_metrics,
    binary_entropy_normalized,
    binary_magnetization,
    binary_polarization,
    consensus_fraction,
    entropy_multiclass_normalized,
    granger_influence_score,
    influence_matrix,
    information_leadership,
    mutual_information_matrix,
    simple_influence_scores,
    transfer_entropy_matrix,
)
from src.metrics.network import (
    assortativity_categorical,
    assortativity_continuous,
    betweenness_centrality,
    connected_components,
    cross_belief_fraction_continuous,
    cross_belief_fraction_discrete,
    edge_disagreement,
    eigenvector_centrality,
    global_clustering_coefficient,
    graph_radius_diameter,
    k_core_decomposition,
    local_agreement_metrics,
    local_clustering_coefficients,
    modularity_categorical,
    network_density,
    node_triangles,
    node_wedges,
    shortest_path_lengths,
    total_triangles,
    total_wedges,
)


def compute_belief_clusters(
    beliefs: np.ndarray, adjacency_matrix: np.ndarray
) -> Dict[str, Any]:
    """Compute belief clustering metrics from final beliefs and network structure.

    Identifies connected components in the network where all nodes share the same
    discrete belief value (determined by threshold at 0.5).

    Args:
        beliefs: Array of continuous belief values (one per agent).
        adjacency_matrix: Binary adjacency matrix for the network.

    Returns:
        Dictionary containing:
            - num_belief_clusters: Number of belief-homogeneous components
            - cluster_sizes: List of sizes for each cluster
            - cluster_purity: Always 1.0 (clusters are pure by construction)
    """
    N = len(beliefs)
    A = adjacency_matrix
    final_b_bin = (beliefs > 0.5).astype(int)

    # Count belief clusters (connected subgraphs of same discrete belief)
    visited = set()
    clusters = []
    for i in range(N):
        if i in visited:
            continue

        # BFS but only traverse same-belief neighbors
        q = deque([i])
        comp = set()
        while q:
            u = q.popleft()
            if u in comp:
                continue
            comp.add(u)
            visited.add(u)
            for v in np.where(A[u] > 0)[0]:
                if final_b_bin[v] == final_b_bin[u] and v not in comp:
                    q.append(v)
        clusters.append(comp)

    cluster_sizes = [len(c) for c in clusters]
    cluster_purity = 1.0  # by construction, all clusters are pure

    return {
        "num_belief_clusters": len(clusters),
        "cluster_sizes": cluster_sizes,
        "cluster_purity": cluster_purity,
    }


class MetricsTracker:
    """Tracks and computes metrics throughout the simulation lifecycle.

    Manages three levels of metrics tracking:
        1. Per-update: Individual agent belief changes
        2. Per-round: Aggregate statistics after each round
        3. Final: Summary metrics after simulation completes

    Attributes:
        cfg: Configuration dictionary.
        io: IOManager for saving metrics.
        per_update: List of per-update metric dictionaries.
        per_round: List of per-round metric dictionaries.
        final_metrics: Dictionary of final summary metrics.
        cross_run: Placeholder for cross-run aggregation.
        _belief_history: History of belief vectors by round.
        _rounds_recorded: Number of rounds recorded.
        _agents_register: Agent-specific network metrics.
        _exp_register: Experiment-level static metrics.
        _exp_registered: Flag for experiment registration.
    """

    def __init__(self, cfg: Dict, io):
        """Initialize metrics tracker.

        Args:
            cfg: Configuration dictionary for metrics settings.
            io: IOManager instance for file operations.
        """
        # config
        self.cfg = cfg or {}
        self.io = io

        # LOGS:
        # each agent update (agent_id, round, prev/new/change)
        self.per_update: List[Dict] = []
        # metrics per round + raw belief vector
        self.per_round: List[Dict] = []
        # post-run summary
        self.final_metrics: Dict = {}
        # NOTE: placeholder for cross-run aggregation
        self.cross_run: List[Dict] = []

        # internal storage
        # beliefs[t] = array of length <num_agents> (run t: from 0..T)
        self._belief_history: List[np.ndarray] = []
        self._rounds_recorded = 0

        # store the network and agent summaries
        self._agents_register: Dict[int, Dict] = {}
        # static experiment-level metrics and graph metrics
        self._exp_register: Dict[str, Any] = {}

        # flags
        self._exp_registered = False

    def _register_agent(self, agent, network) -> Dict:
        """Register agent and compute network-based metrics.

        Computes and caches agent-specific network metrics including degree,
        centrality measures, clustering coefficient, and k-core membership.

        Args:
            agent: Agent instance to register.
            network: Network instance for computing metrics.

        Returns:
            Dictionary containing agent summary with network metrics.
        """
        if agent.id in self._agents_register:
            return self._agents_register[agent.id]
        else:
            # Node-level network metrics
            A = network.adjacency_matrix()
            betweenness = betweenness_centrality(A)
            eigenvector = eigenvector_centrality(A)
            k_cores = k_core_decomposition(A)
            wedges = node_wedges(A)
            triangles = node_triangles(A)
            clustering_coeffs = local_clustering_coefficients(A)
            summary = {
                "agent_id": int(agent.id),
                "role": agent.role,
                "model": agent.model_name,
                "network": {
                    "degree": int(np.sum(A[agent.id])),
                    "betweenness_centrality": float(betweenness[agent.id]),
                    "eigenvector_centrality": float(eigenvector[agent.id]),
                    "clustering_coefficient": float(clustering_coeffs[agent.id]),
                    "triangles": int(triangles[agent.id]),
                    "wedges": int(wedges[agent.id]),
                    "k_core": int(k_cores[agent.id]),
                    "adjacent_agents": [
                        int(nbr) for nbr in network.neighbors(agent.id)
                    ],
                },
            }

            self._agents_register[agent.id] = summary
            return summary

    def register_experiment(self, agents: Dict, network) -> Dict:
        """Register experiment-level and graph-level metrics.

        Computes and caches static metrics about the experiment setup including
        network topology, degree distributions, and connectivity properties.

        Args:
            agents: Dictionary mapping agent IDs to Agent instances.
            network: Network instance.

        Returns:
            Dictionary containing experiment summary metrics.
        """
        summary = {}

        # GRAPH-LEVEL NETWORK METRICS
        A = network.adjacency_matrix()
        degrees = np.sum(A, axis=1)
        total_edges = int(np.sum(np.triu(A, 1)))

        # Compute shortest paths for radius and diameter
        spl = shortest_path_lengths(A)
        radius, diameter = graph_radius_diameter(spl)
        density = network_density(A)

        # Degree distribution
        degree_dist = Counter(degrees.astype(int).tolist())
        degree_distribution = {int(k): int(v) for k, v in degree_dist.items()}

        clustering_coeffs = local_clustering_coefficients(A)
        avg_clustering = float(np.mean(clustering_coeffs))
        global_clust_coeff = global_clustering_coefficient(A)

        components = connected_components(A)
        num_components = len(components)
        component_sizes = [len(c) for c in components]

        spl = shortest_path_lengths(A)
        # Extract finite path lengths for average calculation
        finite_lengths = spl[np.triu_indices_from(spl, k=1)]
        finite_lengths = finite_lengths[np.isfinite(finite_lengths)]
        avg_path_length = (
            float(np.mean(finite_lengths))
            if len(finite_lengths) > 0
            else float(math.inf)
        )

        summary = {
            "graph": {
                "generator": network.generator,
                "num_nodes": network.n,
                "num_edges": total_edges,
                "radius": radius,
                "diameter": diameter,
                "degrees": {
                    "min": int(np.min(degrees)) if network.n > 0 else 0,
                    "max": int(np.max(degrees)) if network.n > 0 else 0,
                    "mean": float(np.mean(degrees)) if network.n > 0 else 0,
                    "median": float(np.median(degrees)) if network.n > 0 else 0,
                    "std": float(np.std(degrees)) if network.n > 0 else 0,
                    "distribution": degree_distribution,
                },
                "triangles": total_triangles(A),
                "wedges": total_wedges(A),
                "avg_path_length": avg_path_length,
                "num_components": num_components,
                "component_sizes": component_sizes,
                "avg_clustering": avg_clustering,
                "global_clustering_coefficient": global_clust_coeff,
                "density": density,
            }
        }

        # AGENT-LEVEL SUMMARY
        # Extract agent composition
        agents_list = []
        role_counts = {}
        model_counts = {}
        for ia, agent in agents.items():
            self._register_agent(agent, network)
            model_name = agent.model_name
            role_name = agent.role

            agents_list.append({"id": ia, "model": model_name, "role": role_name})

            # Count roles and models
            role_counts[role_name] = role_counts.get(role_name, 0) + 1
            model_counts[model_name] = model_counts.get(model_name, 0) + 1

        summary["agents"] = {
            "count": len(agents_list),
            "role_distribution": role_counts,
            "model_distribution": model_counts,
        }
        self._exp_register = summary
        self._exp_registered = True
        return summary

    # RECORDING EACH AGENT'S UPDATE PER ROUND
    def update_agent_records(
        self,
        agent: Any,
        t: int,
        new_belief: float | int,
        new_score: float | int,
        neighbor_view: Dict = None,
        meta: Dict = None,
    ) -> Dict:
        """Record an agent's update for the current round.

        Called after each agent's belief is committed for the round.
        Note: 't' here is the current round index.

        Args:
            agent: Agent object whose belief was updated.
            t: Round number (current round).
            new_belief: New belief value (0, 1, or -1).
            new_score: New belief score (continuous value).
            neighbor_view: Dictionary of neighbor_id -> belief values.
            meta: Additional metadata to include in the record.

        Returns:
            Dictionary containing the update record with belief changes and neighbor info.
        """
        assert (
            self._exp_registered
        ), "The experiment is not registered, run `register_experiment` first "
        agent_id = agent.id
        prev_record = None
        for record in self.per_update:
            if record["agent_id"] == agent_id and record["round"] == t - 1:
                prev_record = record
                break
        if prev_record is not None:
            prev_belief = prev_record["belief"]["curr"]
            prev_score = prev_record["score"]["curr"]
        else:
            prev_belief = None
            prev_score = None

        entry = {
            "agent_id": int(agent_id),
            "round": int(t),  # the round used in experiment call
            "belief": {
                "prev": float(prev_belief) if prev_belief is not None else None,
                "curr": float(new_belief),
                "delta": (
                    (float(new_belief) - float(prev_belief))
                    if (prev_belief is not None)
                    else None
                ),
            },
            "score": {
                "prev": float(prev_score) if prev_score is not None else None,
                "curr": float(new_score) if new_score is not None else None,
                "delta": (
                    (float(new_score) - float(prev_score))
                    if (prev_score is not None and new_score is not None)
                    else None
                ),
            },
            "message": agent.current_message,
        }

        if neighbor_view is not None:
            # Collect neighbor information for metrics
            neighbors = list(neighbor_view.keys())
            # Count agree/disagree/neutral neighbors
            n_agree = sum(
                1
                for nb_belief in neighbor_view.values()
                if nb_belief is not None and nb_belief == 1
            )
            n_disagree = sum(
                1
                for nb_belief in neighbor_view.values()
                if nb_belief is not None and nb_belief == 0
            )
            n_neutral = sum(
                1
                for nb_belief in neighbor_view.values()
                if nb_belief is None or (nb_belief != 0 and nb_belief != 1)
            )

            agents = self._agents_register
            # Get roles of neighbors by stance
            agree_roles = [
                agents[nb_id].get("role")
                for nb_id, nb_belief in neighbor_view.items()
                if nb_belief is not None and nb_belief == 1
            ]
            disagree_roles = [
                agents[nb_id].get("role")
                for nb_id, nb_belief in neighbor_view.items()
                if nb_belief is not None and nb_belief == 0
            ]
            neutral_roles = [
                agents[nb_id].get("role")
                for nb_id, nb_belief in neighbor_view.items()
                if nb_belief is None or (nb_belief != 0 and nb_belief != 1)
            ]

            # Create metadata with neighbor information
            entry["neighbor_info"] = {
                "num_neighbors": len(neighbors),
                "n_agree": n_agree,
                "n_disagree": n_disagree,
                "n_neutral": n_neutral,
                "roles_agree": agree_roles,
                "roles_disagree": disagree_roles,
                "roles_neutral": neutral_roles,
            }
        else:
            entry["neighbor_info"] = {
                "num_neighbors": None,
                "n_agree": None,
                "n_disagree": None,
                "n_neutral": None,
                "roles_agree": None,
                "roles_disagree": None,
                "roles_neutral": None,
            }
        if meta:
            entry.update(meta)
        self.per_update.append(entry)

        return entry

    def _validate_round(self, t: int) -> bool:
        """
        Make sure that all agents recieved an update
        :params t: current round
        """
        # Get all agent IDs that have been updated for round t
        updated_agents = set()
        for record in self.per_update:
            if record["round"] == t:
                updated_agents.add(record["agent_id"])

        existing_agents = set()
        for record in self._agents_register.values():
            existing_agents.add(record["agent_id"])

        # Find missing agents
        missing_agents = existing_agents - updated_agents

        if len(missing_agents) == 0:
            return True
        else:
            return False

    # RECORDING AFTER ALL UPDATES FOR ROUND

    def record_round(self, t: int, network: Any) -> Dict:
        """
        Record aggregated metrics for a round after all agent updates are committed.

        This method computes round-level metrics including:
        - Belief distributions (mean, variance, entropy)
        - Consensus and polarization measures
        - Local agreement and edge disagreement
        - Change rates from previous round

        :param t: Round index for which beliefs have just been set
        :param network: Network instance
        :return: Dictionary of computed round metrics
        Example return structure:
        {
            "round": 1,
            "belief": {
                "label": {
                    "values": [...],
                    "median": ...,
                    "mode": ...,
                    "var": ...,
                    "mean": ...,
                },
                "score": {
                    "values": [...],
                    "median": ...,
                    "mode": ...,
                    "mean": ...,
                    "var": ...,
                },
            },
            "consensus": {
                "magnetization": ...,
                "polarity": ...,
                "consensus_fraction": ...,
            },
            "temporal": {
                "flip_rate": ...,
                "change_l1": ...,
                "change_l2": ...,
            },
            "local": {
                "local_agreement": {

        }
        """
        # Validate that all agents have been updated for this round
        assert self._validate_round(
            t
        ), "You did not update the beliefs of all the agents"

        # Collect agent data for this round
        agents = [a for a in self.per_update if a["round"] == t]

        # Extract belief labels and scores
        # 1=agree, 0=disagree, -1=neutral
        _belief_labels = np.asarray([a["belief"]["curr"] for a in agents], dtype=int)
        _belief_scores = np.asarray([a["score"]["curr"] for a in agents], dtype=float)

        # Initialize entry dictionary
        entry = {"round": int(t)}

        # OVERALL BELIEF DISTRIBUTION METRICS
        entry["belief"] = {
            "label": {
                "values": _belief_labels.tolist(),  # Convert to list for JSON serialization
                "median": (
                    float(np.median(_belief_labels))
                    if len(_belief_labels) > 0
                    else None
                ),
                "mode": (
                    float(Counter(_belief_labels).most_common(1)[0][0])
                    if len(_belief_labels) > 0
                    else None
                ),
                "var": (
                    float(np.var(_belief_labels)) if len(_belief_labels) > 0 else None
                ),
                "mean": (
                    float(np.mean(_belief_labels)) if len(_belief_labels) > 0 else None
                ),
            },
            "score": {
                "values": _belief_scores.tolist(),  # Convert to list for JSON serialization
                "median": (
                    float(np.median(_belief_scores))
                    if len(_belief_scores) > 0
                    else None
                ),
                "mode": (
                    float(Counter(_belief_scores).most_common(1)[0][0])
                    if len(_belief_scores) > 0
                    else None
                ),
                "mean": (
                    float(np.mean(_belief_scores)) if len(_belief_scores) > 0 else None
                ),
                "var": (
                    float(np.var(_belief_scores)) if len(_belief_scores) > 0 else None
                ),
            },
        }

        # Convert to binary labels for some metrics (treating only 1 as positive)
        binary_labels = (_belief_labels == 1).astype(int)

        # Save beliefs for history tracking
        self._belief_history.append(_belief_labels.copy())

        # TEMPORAL DYNAMICS

        # Flip rate: fraction of agents whose belief changed vs previous round
        flip_rate, change_l1, change_l2 = belief_change_metrics(
            prev=self._belief_history[-2] if len(self._belief_history) >= 2 else None,
            curr=_belief_labels,
        )

        # LOCAL / NETWORK-BASED METRICS
        A = network.adjacency_matrix()
        # Local agreement per node: fraction of neighbors with same belief
        local_agreement, mean_local_agreement, neighbor_mean_beliefs = (
            local_agreement_metrics(A, _belief_labels)
        )

        # Increment round counter
        self._rounds_recorded += 1

        # COMPILE ALL METRICS
        entry.update(
            {
                "entropy": binary_entropy_normalized(binary_labels),
                "multiclass_entropy": entropy_multiclass_normalized(_belief_labels),
                "consensus": {
                    "magnetization": binary_magnetization(binary_labels),
                    "polarity": binary_polarization(binary_labels),
                    "consensus_fraction": consensus_fraction(_belief_labels),
                },
                "temporal": {
                    "flip_rate": flip_rate,
                    "change_l1": change_l1,
                    "change_l2": change_l2,
                },
                "local": {
                    "agreement_mean": mean_local_agreement,
                    "edge_disagreement": edge_disagreement(A, _belief_labels),
                    "assortativity": assortativity_categorical(binary_labels, A),
                    "assortativity_con": assortativity_continuous(_belief_scores, A),
                    "cross_belief_fraction_cat": cross_belief_fraction_discrete(
                        _belief_labels, A
                    ),
                    "cross_belief_fraction_con": cross_belief_fraction_continuous(
                        _belief_scores, A
                    ),
                    "modularity": modularity_categorical(A, binary_labels),
                    "neighbor_mean": neighbor_mean_beliefs.tolist(),
                },
            }
        )

        # Store the entry
        self.per_round.append(entry)

        # Return the stored entry
        return entry

    # COMPUTE FINAL METRICS AFTER ALL ROUNDS ARE FINISHED
    def finalize(self, agents: Dict[int, Any], network: Any) -> None:
        """
        compute final and global metrics after the experiment ends
        uses self._belief_history (list of np arrays) saved by record_round
        """
        # basic checks to make sure rounds recorded
        if len(self._belief_history) == 0:
            self.final_metrics = {"error": "Mo rounds recorded"}
            return

        agent_ids = sorted(agents.keys())
        N = len(agent_ids)
        A = network.adjacency_matrix()

        # --- last-round beliefs
        final_belief_labels = self._belief_history[-1]
        T = len(self._belief_history)

        mi_matrix = mutual_information_matrix(self._belief_history)
        te_matrix = transfer_entropy_matrix(self._belief_history, A)

        cluster_metrics = compute_belief_clusters(final_belief_labels, A)

        # BUILD FINAL METRICS DICTIONARY
        self.final_metrics = {
            # --- final state
            "final_belief_labels": final_belief_labels.tolist(),
            # DEPRECATED: Use final_belief_labels instead (kept for backward compatibility)
            "final_beliefs": final_belief_labels.tolist(),
            "consensus_fraction": consensus_fraction(final_belief_labels),
            "final_magnetization": binary_magnetization(final_belief_labels),
            "num_belief_clusters": cluster_metrics["num_belief_clusters"],
            "cluster_sizes": cluster_metrics["cluster_sizes"],
            "cluster_purity": cluster_metrics["cluster_purity"],
            # --- temporal & information
            "T": T,
            "mi_matrix": mi_matrix.tolist() if mi_matrix.size > 0 else [],
            "influence_scores_granger": (
                granger_influence_score(self._belief_history).tolist() if N > 1 else []
            ),
            "influence_scores_simple": (
                simple_influence_scores(self._belief_history).tolist() if N > 1 else []
            ),
            "te_matrix": te_matrix.tolist() if te_matrix.size > 0 else [],
            "influence_matrix": influence_matrix(te_matrix).tolist() if N > 1 else [],
            "information_leadership": (
                information_leadership(te_matrix).tolist() if N > 1 else []
            ),
            # --- time-series summary
            "per_round": self.per_round,  # include the per-round logs for downstream analysis
            "per_update": self.per_update,
        }

    # ----------------------
    # export summary to JSON-like dict
    # ----------------------
    def summary(self) -> Dict:
        return {
            "per_round": self.per_round,
            "per_update": self.per_update,
            "final_metrics": self.final_metrics,
            "cross_run": self.cross_run,
        }

    # ----------------------
    # optional: record a run summary (for cross-run aggregation)
    # ----------------------

    def record_run_summary(self, run_id: int, extra: Optional[Dict] = None) -> None:
        rec = {"run_id": int(run_id)}
        if extra:
            rec.update(extra)
        self.cross_run.append(rec)
