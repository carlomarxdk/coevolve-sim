from __future__ import annotations

import json
import logging
import pathlib
import random
from typing import Any

import numpy as np
from hydra import compose
from omegaconf import DictConfig

log = logging.getLogger("utils")


def build_catalog(spec: dict, seed: int | None = None) -> list[dict]:
    """Build a catalog of agents based on a specification.
    The configuration files are stored in 'configs/catalog/'.

    Args:
        spec: Specification dictionary containing the catalog details
        seed: Optional integer for deterministic shuffling.

    Returns:
        A list of agent dictionaries with unique IDs.
        A single agent specification is a dictionary with keys:
            - id: Unique integer identifier for the agent.
            - name: Name of the LLM models to use.
            - role: Role (social label) of the agent.
            - prompt: Prompt template for the agent.
    """
    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)

    catalog = []
    next_id = 0

    mode = spec.get("mode")

    # --------------------------------------------------
    # 1. Build catalog (procedural or explicit)
    # --------------------------------------------------

    if mode == "procedural":
        log.info("Building PROCEDURAL agent catalog...")

        for key, n in spec.get("counts", {}).items():
            tmpl = spec["role_templates"][key]
            for _ in range(n):
                catalog.append(
                    {
                        "id": next_id,
                        "name": tmpl["name"],
                        "role": tmpl["role"],
                        "prompt": tmpl["prompt"],
                    }
                )
                next_id += 1

    elif mode == "explicit":
        log.info("Building EXPLICIT agent catalog...")

        for agent in spec.get("explicit", []):
            agent = dict(agent)
            catalog.append(agent)
            next_id += 1

    else:
        raise ValueError(f"Unknown catalog mode: {mode}")

    # --------------------------------------------------
    # 2. Procedural random role reassignment
    # --------------------------------------------------

    if spec.get("random_roles", False):
        log.info("Applying PROCEDURAL random role assignment...")

        random_roles_spec = spec.get("random_roles_spec", {})

        if not random_roles_spec:
            raise ValueError("random_roles is enabled but random_roles_spec is empty.")

        # Build role multiset
        roles = []
        for role, count in random_roles_spec.items():
            roles.extend([role] * count)

        if len(roles) != len(catalog):
            raise ValueError(
                "Sum of random_roles_spec counts does not match "
                "number of agents in catalog."
            )

        # Deterministic shuffle
        rng.shuffle(roles)

        # Assign roles
        for agent, role in zip(catalog, roles, strict=True):
            agent["role"] = role

    return catalog


def convert_numpy_to_native(obj: Any) -> Any:
    """Convert numpy arrays and types to native Python types for JSON serialization.

    Recursively processes nested data structures and converts:
    - numpy.ndarray to Python lists via .tolist()
    - numpy scalar types (np.float64, np.int32, etc.) to Python primitives via .item()
    - Nested dicts and lists are processed recursively

    Args:
        obj: Object to convert (can be nested dict, list, numpy array, or scalar)

    Returns:
        Object with all numpy arrays/types converted to native Python types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_to_native(item) for item in obj]
    return obj


# ---- Loading configs ----
def load_config_from_file(config_path: str, config_name: str) -> DictConfig:
    """Load a configuration from a Hydra config file.

    Args:
        config_path: Path to the config directory.
        config_name: Name of the config file without extension.

    Returns:
        Configuration as a DictConfig object.
    """
    cfg = compose(config_name=f"{config_path}/{config_name}")
    return cfg


def load_model_config(model_name: str) -> DictConfig:
    """Load a model configuration given its name.

    Args:
        model_name: Name of the model.

    Returns:
        Model configuration as a DictConfig object.
    """
    # For simplicity, assume model configs are stored in 'configs/models/{model_name}.yaml'
    return load_config_from_file("model", model_name)


def get_experiment_choices(experiment_cfg: dict | DictConfig | None = None) -> dict:
    """Extract catalog, prompt, and statement choices from Hydra configuration.

    Args:
        experiment_cfg: Full experiment configuration.

    Returns:
        Dictionary of choices including catalog, prompt, statement, network, and probe.
    """

    # TODO: Simplify this function - it has too many fallback mechanisms

    catalog_choice = "unk_catalog"
    prompt_choice = "unk_prompt"
    statement_choice = "unk_statement"
    network_choice = "unk_network"
    probe_choice = "unk_probe"

    # Try to get choices from HydraConfig (when running under Hydra)
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            hydra_cfg = HydraConfig.get()
            hydra_choices = hydra_cfg.runtime.choices
            catalog_choice = hydra_choices.get("catalog", catalog_choice)
            prompt_choice = hydra_choices.get("prompt", prompt_choice)
            statement_choice = hydra_choices.get("statement", statement_choice)
            network_choice = hydra_choices.get("network", network_choice)
            probe_choice = hydra_choices.get("probe", probe_choice)
            return {
                "catalog": catalog_choice,
                "network": network_choice,
                "prompt": prompt_choice,
                "statement": statement_choice,
                "probe": probe_choice,
            }
    except (ImportError, ModuleNotFoundError, AttributeError):
        # HydraConfig not available or not initialized
        pass

    # Fallback: Try to get from experiment_cfg if HydraConfig not available
    if experiment_cfg:
        # Try to get choices from Hydra runtime metadata in config
        try:
            cfg_hydra_choices = (
                experiment_cfg.get("hydra", {}).get("runtime", {}).get("choices", {})
            )
            if cfg_hydra_choices:
                catalog_choice = cfg_hydra_choices.get("catalog", catalog_choice)
                prompt_choice = cfg_hydra_choices.get("prompt", prompt_choice)
                statement_choice = cfg_hydra_choices.get("statement", statement_choice)
                network_choice = cfg_hydra_choices.get("network", network_choice)
                probe_choice = cfg_hydra_choices.get("probe", probe_choice)
                return {
                    "catalog": catalog_choice,
                    "network": network_choice,
                    "prompt": prompt_choice,
                    "statement": statement_choice,
                    "probe": probe_choice,
                }
        except (AttributeError, KeyError):
            pass

        # Additional fallback: Try to get from config structure
        # Get statement ID
        statement_info = experiment_cfg.get("statement", {})
        statement_choice = statement_info.get("id", statement_choice)

        # Get prompt choice from the prompt config
        prompt_info = experiment_cfg.get("prompt", {})
        prompt_choice = prompt_info.get("type", prompt_choice)

        # Try to infer catalog from agents structure
        catalog_info = experiment_cfg.get("catalog", {})
        catalog_choice = catalog_info.get("name", catalog_choice)

        # Get probe choice
        probe_info = experiment_cfg.get("probe", {})
        probe_choice = probe_info.get("name", probe_choice)

        network_info = experiment_cfg.get("network", {})
        network_choice = network_info.get("generator", network_choice)

    return {
        "catalog": catalog_choice,
        "network": network_choice,
        "prompt": prompt_choice,
        "statement": statement_choice,
        "probe": probe_choice,
    }


# ---- Utility Functions ----


def move_incomplete_experiments(
    catalog_choice: str,
    prompt_choice: str,
    statement_choice: str,
    network_choice: str,
    probe_choice: str,
    seed: int,
    max_rounds: int = 10,
    base_dir: str = "data/outputs/runs",
    incomplete_dir: str = "data/outputs/incomplete_runs",
) -> int:
    """Move incomplete experiments with the given seed to the incomplete_runs directory.

    Args:
        catalog_choice: Catalog configuration name.
        prompt_choice: Prompt configuration name.
        statement_choice: Statement configuration name.
        network_choice: Network configuration name.
        probe_choice: Probe configuration name.
        seed: Random seed used for the experiment.
        max_rounds: Maximum number of rounds for a completed experiment.
        base_dir: Base directory for experiment outputs.
        incomplete_dir: Directory to move incomplete experiments to.

    Returns:
        Number of incomplete experiments moved.
    """
    import shutil

    base_path = pathlib.Path(base_dir)
    experiment_dir = (
        base_path
        / probe_choice
        / catalog_choice
        / network_choice
        / prompt_choice
        / statement_choice
    )

    # If the experiment directory doesn't exist, no incomplete experiments
    if not experiment_dir.exists():
        return 0

    moved_count = 0

    # Check all subdirectories (timestamp folders) in the experiment directory
    for timestamp_dir in experiment_dir.iterdir():
        if not timestamp_dir.is_dir():
            continue

        # Check if config.json exists
        config_path = timestamp_dir / "config.json"
        if not config_path.exists():
            continue

        # Load the config and check if seed matches
        try:
            with open(config_path) as f:
                config = json.load(f)

            # Check if the seed matches
            if config.get("seed") != seed:
                continue

            # Check if the experiment is incomplete
            rounds_dir = timestamp_dir / "rounds"

            # If rounds directory doesn't exist, the experiment is incomplete
            if not rounds_dir.exists():
                all_rounds_complete = False
            else:
                # Check that all rounds from 0 to max_rounds-1 exist with beliefs.jsonl
                all_rounds_complete = True
                for round_num in range(max_rounds):
                    round_dir = rounds_dir / f"round_{round_num}"
                    beliefs_path = round_dir / "beliefs.jsonl"
                    if not round_dir.exists() or not beliefs_path.exists():
                        all_rounds_complete = False
                        break

            # If incomplete, move to incomplete_runs directory
            if not all_rounds_complete:
                incomplete_path = pathlib.Path(incomplete_dir)
                dest_dir = (
                    incomplete_path
                    / probe_choice
                    / catalog_choice
                    / network_choice
                    / prompt_choice
                    / statement_choice
                )
                dest_dir.mkdir(parents=True, exist_ok=True)

                dest_path = dest_dir / timestamp_dir.name
                # If destination already exists, add a suffix
                if dest_path.exists():
                    counter = 1
                    while dest_path.exists():
                        dest_path = dest_dir / f"{timestamp_dir.name}_{counter}"
                        counter += 1

                shutil.move(str(timestamp_dir), str(dest_path))
                moved_count += 1

        except (json.JSONDecodeError, OSError):
            # If we can't read the config, skip this directory
            continue

    return moved_count


def check_experiment_completed(
    catalog_choice: str,
    prompt_choice: str,
    statement_choice: str,
    network_choice: str,
    probe_choice: str,
    seed: int,
    max_rounds: int = 10,
    base_dir: str = "data/outputs/runs",
) -> bool:
    """Check if an experiment with the given parameters has already been completed.

    Args:
        catalog_choice: Catalog configuration name.
        prompt_choice: Prompt configuration name.
        statement_choice: Statement configuration name.
        network_choice: Network configuration name.
        probe_choice: Probe configuration name.
        seed: Random seed used for the experiment.
        max_rounds: Maximum number of rounds for a completed experiment.
        base_dir: Base directory for experiment outputs.

    Returns:
        True if a completed experiment exists, False otherwise.
    """
    base_path = pathlib.Path(base_dir)
    experiment_dir = (
        base_path
        / probe_choice
        / catalog_choice
        / network_choice
        / prompt_choice
        / statement_choice
    )

    # If the experiment directory doesn't exist, no completed experiments
    if not experiment_dir.exists():
        return False

    # Check all subdirectories (timestamp folders) in the experiment directory
    for timestamp_dir in experiment_dir.iterdir():
        if not timestamp_dir.is_dir():
            continue

        # Check if config.json exists
        config_path = timestamp_dir / "config.json"
        if not config_path.exists():
            continue

        # Load the config and check if seed matches
        try:
            with open(config_path) as f:
                config = json.load(f)

            # Check if the seed matches
            if config.get("seed") != seed:
                continue

            # Check if the experiment completed the required number of rounds
            # Rounds are 0-indexed, so round_0 to round_{max_rounds-1}
            rounds_dir = timestamp_dir / "rounds"
            if not rounds_dir.exists():
                continue

            cmpltn_path = timestamp_dir / "results" / "final_metrics.json"
            if not cmpltn_path.exists():
                continue

            # Check that all rounds from 0 to max_rounds-1 exist with beliefs.jsonl
            all_rounds_complete = True
            for round_num in range(max_rounds):
                round_dir = rounds_dir / f"round_{round_num}"
                beliefs_path = round_dir / "beliefs.jsonl"
                if not round_dir.exists() or not beliefs_path.exists():
                    all_rounds_complete = False
                    break

            if all_rounds_complete:
                return True

        except (json.JSONDecodeError, OSError):
            # If we can't read the config, skip this directory
            continue

    return False


class IOManager:
    def __init__(self, cfg: dict, experiment_cfg: dict | None = None):
        """Initialize IOManager with experiment-specific directories and database tracking.

        Args:
            cfg: IO configuration dictionary.
            experiment_cfg: Full experiment configuration for recording.
        """
        # Extract experiment name components
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        choices = get_experiment_choices(experiment_cfg)
        catalog_choice = choices["catalog"]
        network_choice = choices["network"]
        prompt_choice = choices["prompt"]
        statement_choice = choices["statement"]
        probe_choice = choices["probe"]

        # Store choices for later use (for _save_experiment_config)
        hydra_choices = None
        try:
            from hydra.core.hydra_config import HydraConfig

            if HydraConfig.initialized():
                hydra_cfg = HydraConfig.get()
                hydra_choices = hydra_cfg.runtime.choices
        except (ImportError, ModuleNotFoundError, AttributeError):
            pass

        # Store the choices for later use in _save_experiment_config
        self._hydra_choices = hydra_choices

        # Create experiment name with timestamp
        self.experiment_name = timestamp

        # Create hierarchical directory structure: data/outputs/runs/<catalog>/<network>/<prompt>/<statement>/<timestamp>
        base_dir = pathlib.Path(cfg.get("out_dir", "data/outputs/runs"))
        self.base_dir = base_dir
        self.out_dir = (
            base_dir
            / probe_choice
            / catalog_choice
            / network_choice
            / prompt_choice
            / statement_choice
            / self.experiment_name
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Store configuration
        self.save_activations = cfg.get("save_activations", False)
        self.save_text = cfg.get("save_text", True)
        self.experiment_cfg = experiment_cfg

        # Save experiment configuration
        if experiment_cfg:
            self._save_experiment_config()

    def _save_experiment_config(self):
        """Save the experiment HYDRA configuration to the experiment directory.

        Preserves Hydra runtime metadata including choices.
        """
        config_path = self.out_dir / "config.json"
        # Convert OmegaConf to dict if needed
        config_dict: dict[str, Any]
        try:
            from omegaconf import OmegaConf

            if hasattr(self.experiment_cfg, "__dict__"):
                # Resolve most values but try to preserve hydra metadata
                container = OmegaConf.to_container(self.experiment_cfg, resolve=True)
                # OmegaConf.to_container can return dict, list, or primitive types
                # Convert to regular dict for type safety (OmegaConf returns special dict type)
                config_dict = (
                    dict(container)  # type: ignore[arg-type]
                    if isinstance(container, dict)
                    else {}
                )

                # Add hydra runtime choices if we captured them earlier
                if self._hydra_choices:
                    if "hydra" not in config_dict:
                        config_dict["hydra"] = {}
                    if "runtime" not in config_dict["hydra"]:
                        config_dict["hydra"]["runtime"] = {}
                    config_dict["hydra"]["runtime"]["choices"] = dict(
                        self._hydra_choices
                    )
            else:
                config_dict = self.experiment_cfg  # type: ignore[assignment]
        except ImportError:
            config_dict = self.experiment_cfg  # type: ignore[assignment]

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def update_and_save_config(self) -> None:
        """Update and re-save the experiment configuration.

        This should be called after any modifications to the config
        (e.g., after network seed remapping or role randomization).
        """
        self._save_experiment_config()

    def save_json(self, filename: str, data: dict) -> pathlib.Path:
        """Save a JSON file inside the experiment's output directory.

        Automatically creates directories if needed.

        Args:
            filename: Relative path to the file within the experiment directory.
            data: Dictionary to save as JSON.

        Returns:
            Path to the saved file.
        """

        path: pathlib.Path = self.out_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def _round_path(self, t: int) -> pathlib.Path:
        """Generate the file path for a given round.

        Now uses the rounds subdirectory for better organization.

        Args:
            t: Round number.

        Returns:
            Path to the round directory.
        """
        rounds_dir: pathlib.Path = self.out_dir / "rounds"
        rounds_dir.mkdir(parents=True, exist_ok=True)

        p: pathlib.Path = rounds_dir / f"round_{t}"
        p.mkdir(parents=True, exist_ok=True)

        return p

    def cache_activation(
        self, agent_id: int, t: int, activation: Any
    ) -> pathlib.Path | None:
        """Save the activations of a given agent at a given round.

        Args:
            agent_id: Agent ID.
            t: Round number.
            activation: Activation generated by the agent (numpy array).

        Returns:
            The file path where the activation is saved, or None if not saved.
        """
        if self.save_activations:
            raise NotImplementedError("Activation saving not implemented yet.")
        return None
        # if not self.save_activations:
        #     return self._round_path(t) / f'{agent_id}.activation.ignore'

        # # TODO: decide tensor format (npy/pt) and save; placeholder json meta
        # p = self._round_path(t) / f'{agent_id}.activation.json'
        # with open(p, 'w') as f:
        #     json.dump({'shape': getattr(activation, 'shape', None), 'meta': 'TODO'}, f)

        # return p

    def checkpoint(self, t: int, agents: dict[int, Any], metrics: Any) -> None:
        """Save a checkpoint for round t, including beliefs of agents and round metrics.

        Organizes files in a structured way within the experiment directory.

        Args:
            t: Round number.
            agents: Dictionary of agents.
            metrics: MetricsTracker object.
        """
        # Create rounds subdirectory for better organization
        rounds_dir = self.out_dir / "rounds"
        rounds_dir.mkdir(parents=True, exist_ok=True)

        round_dir = rounds_dir / f"round_{t}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # save beliefs for all agents
        beliefs_path = round_dir / "beliefs.jsonl"
        with open(beliefs_path, "w") as f:
            for a in agents.values():
                b = a.current_belief(t)
                s = a.current_belief_score(t)
                fs = a.current_complete_scores(t)
                rec = {
                    "round": t,
                    "belief": b,
                    "agent": a.id,
                    "node_id": getattr(a, "node_id", a.id),
                    "catalog_id": getattr(a, "catalog_id", a.id),
                    "role": a.role,
                    "score": s,
                    "complete_scores": fs,
                }
                # Convert numpy arrays to native types before JSON serialization
                rec = convert_numpy_to_native(rec)
                f.write(json.dumps(rec) + "\n")

        # detailed belief scores per agent
        scores_path = round_dir / "beliefs_detailed.json"
        a = metrics.per_update
        detailed_scores = [u for u in a if u.get("round") == t]
        # Convert numpy arrays to native types before JSON serialization
        detailed_scores = convert_numpy_to_native(detailed_scores)
        with open(scores_path, "w") as f:
            json.dump(detailed_scores, f, indent=2)

        # save the most recent per-round metrics
        if metrics.per_round:
            metrics_path = round_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics.per_round[-1], f, indent=2)

    def save_artifacts(
        self, agents: dict[int, Any], network: Any, metrics: Any
    ) -> None:
        """Save different artifacts in an organized manner.

        Creates separate subdirectories for different types of outputs.

        Args:
            agents: Dictionary of agents.
            network: Network object.
            metrics: MetricsTracker object.
        """

        # Create organized subdirectories
        results_dir = self.out_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        ## AGENT DATA
        with open(results_dir / "agent_manifest.json", "w") as f:
            json.dump(metrics._agents_register, f, indent=2)

        ## EXPERIMENT REGISTER
        with open(results_dir / "network_manifest.json", "w") as f:
            json.dump(metrics._exp_register, f, indent=2)

        # Save all per-round metrics
        all_metrics_path = results_dir / "per_round_metrics.json"
        with open(all_metrics_path, "w") as f:
            json.dump({"per_round": metrics.per_round}, f, indent=2)

        # Save final metrics
        final_metrics_path = results_dir / "final_metrics.json"
        with open(final_metrics_path, "w") as f:
            json.dump(metrics.final_metrics, f, indent=2)

        # network structure
        net_path = results_dir / "network_edges.json"
        with open(net_path, "w") as f:
            json.dump({"edges": list(network._edges), "n": network.n}, f, indent=2)

        # agents final beliefs and scores
        agent_data = {}
        for a in agents.values():
            agent_data[a.id] = {
                "node_id": getattr(a, "node_id", a.id),
                "catalog_id": getattr(a, "catalog_id", a.id),
                "beliefs": a.beliefs,
                "belief_scores": a._belief_score,
                "complete_scores": a._complete_scores,
                "role": a.role if hasattr(a, "role") else "unknown",
                "model": a.model_name if hasattr(a, "model_name") else "unknown",
            }

        # Convert numpy arrays to native types before JSON serialization
        agent_data = convert_numpy_to_native(agent_data)
        agents_path = results_dir / "agents_data.json"
        with open(agents_path, "w") as f:
            json.dump(agent_data, f, indent=2)


class StoppingCriteria:
    def __init__(self, max_rounds: int = 20, eps: float = 0.0, patience: int = 0):
        self.max_rounds = max_rounds
        self.eps = eps
        self.patience = patience
        self._stable_streak = 0

    def check(self, metrics: Any, t: int) -> bool:
        """Check if agent beliefs are stable.

        Args:
            metrics: MetricsTracker object.
            t: Round number.

        Returns:
            True if stopping criteria is met, False otherwise.
        """

        # check if we've exceeded the max number of rounds
        if t + 1 >= self.max_rounds:
            return True

        # check if the beliefs are stable
        updates = [u for u in metrics.per_update if u["round"] == t]
        if not updates:
            return False

        # Extract deltas from the new structure
        deltas = []
        for u in updates:
            if "belief" in u and "delta" in u["belief"]:
                delta = u["belief"]["delta"]
                if delta is not None:
                    deltas.append(abs(delta))

        if (
            deltas and sum(deltas) / len(deltas) < self.eps
        ):  # check against some threshold
            self._stable_streak += 1
        else:
            self._stable_streak = 0

        return (
            self._stable_streak >= self.patience
        )  # see if we've exceeded our patience rounds


def move_invalid_runs(
    runs_dir: pathlib.Path | str | None = None,
    incomplete_dir: pathlib.Path | str | None = None,
    dry_run: bool = False,
) -> tuple(dict[str, int], Any):
    """Move incomplete or unvalidated runs to an incomplete_runs directory.

    Uses the analysis loaders to identify runs that are either:
    - Incomplete (missing final_metrics.json)
    - Unvalidated (invalid seed, statement_id, prompt, probe, graph_type, or setting)

    Args:
        runs_dir: Path to the runs directory. If None, uses 'data/outputs/runs'.
        incomplete_dir: Path to move invalid runs to. If None, uses 'data/outputs/incomplete_runs'.
        dry_run: If True, only print what would be moved without actually moving files.

    Returns:
        Dictionary with counts:
            - 'incomplete': Number of incomplete runs moved
            - 'unvalidated': Number of unvalidated runs moved
            - 'total': Total number of runs moved

    Example:
        >>> from src.utils import move_invalid_runs
        >>> # Preview what would be moved
        >>> stats = move_invalid_runs(dry_run=True)
        >>> # Actually move the files
        >>> stats = move_invalid_runs(dry_run=False)
        >>> print(f"Moved {stats['total']} invalid runs")
    """
    import shutil

    from src.analysis.loaders import load_runs_metadata

    # Set default paths
    if runs_dir is None:
        runs_dir = pathlib.Path("data") / "outputs" / "runs"
    else:
        runs_dir = pathlib.Path(runs_dir)

    if incomplete_dir is None:
        incomplete_dir = pathlib.Path("data") / "outputs" / "incomplete_runs"
    else:
        incomplete_dir = pathlib.Path(incomplete_dir)

    # Load metadata for all runs
    log.info(f"Loading run metadata from {runs_dir}...")
    df_runs = load_runs_metadata(runs_dir)

    if df_runs.empty:
        log.warning(f"No runs found in {runs_dir}")
        return {"incomplete": 0, "unvalidated": 0, "total": 0}

    # Filter for invalid runs
    invalid_runs = df_runs[~(df_runs["completed"] & df_runs["validated"])].copy()

    if invalid_runs.empty:
        log.info("No invalid runs found. All runs are complete and validated.")
        return {"incomplete": 0, "unvalidated": 0, "total": 0}

    # Count types of invalid runs
    incomplete_count = (~invalid_runs["completed"]).sum()
    unvalidated_count = (~invalid_runs["validated"]).sum()

    log.info(
        f"Found {len(invalid_runs)} invalid runs "
        f"({incomplete_count} incomplete, {unvalidated_count} unvalidated)"
    )

    if dry_run:
        log.info("DRY RUN: No files will be moved")
        log.info("\nRuns that would be moved:")
        for _, run in invalid_runs.iterrows():
            status = []
            if not run["completed"]:
                status.append("INCOMPLETE")
            if not run["validated"]:
                status.append("UNVALIDATED")
            log.info(f"  [{', '.join(status)}] {run['run_path']}")
        return {
            "incomplete": incomplete_count,
            "unvalidated": unvalidated_count,
            "total": len(invalid_runs),
        }, invalid_runs

    # Move invalid runs
    moved_count = 0
    for _, run in invalid_runs.iterrows():
        source_path = pathlib.Path(run["run_path"])

        if not source_path.exists():
            log.warning(f"Source path does not exist: {source_path}")
            continue

        # Reconstruct the directory structure in incomplete_runs
        # Structure: {probe}/{setting}/{graph_type}/{prompt}/{statement_id}/{datetime}/
        rel_path = source_path.relative_to(runs_dir)
        dest_path = incomplete_dir / rel_path

        # Create parent directories
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle existing destination
        if dest_path.exists():
            counter = 1
            original_dest = dest_path
            while dest_path.exists():
                dest_path = original_dest.parent / f"{original_dest.name}_{counter}"
                counter += 1

        try:
            shutil.move(str(source_path), str(dest_path))
            status = []
            if not run["completed"]:
                status.append("INCOMPLETE")
            if not run["validated"]:
                status.append("UNVALIDATED")
            log.info(f"Moved [{', '.join(status)}]: {source_path} -> {dest_path}")
            moved_count += 1
        except (OSError, shutil.Error) as e:
            log.error(f"Failed to move {source_path}: {e}")

    log.info(f"Successfully moved {moved_count} invalid runs to {incomplete_dir}")

    return (
        {
            "incomplete": incomplete_count,
            "unvalidated": unvalidated_count,
            "total": moved_count,
        },
        invalid_runs,
    )


def set_seed(seed: int) -> None:
    """Set seed for reproducibility across random, numpy, and other libraries."""
    random.seed(seed)
