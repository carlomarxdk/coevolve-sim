from __future__ import annotations

import json
import pathlib
import random
from typing import Any, Dict, Optional

from hydra import compose


# ---- Loading configs ----
def load_config_from_file(config_path: str, config_name: str) -> Dict:
    """
    Load a configuration from a Hydra config file.

    :param config_path: path to the config directory (str)
    :param config_name: name of the config file without extension (str)
    :return: configuration as a dictionary
    """
    cfg = compose(config_name=f"{config_path}/{config_name}")
    return cfg


def load_model_config(model_name: str) -> Dict:
    """
    Load a model configuration given its name.

    :param model_name: name of the model (str)
    :return: model configuration as a dictionary
    """
    # For simplicity, assume model configs are stored in 'configs/models/{model_name}.yaml'
    return load_config_from_file("model", model_name)


def get_experiment_choices(experiment_cfg: Dict = None) -> tuple:
    """
    Extract catalog, prompt, and statement choices from Hydra configuration.

    :param experiment_cfg: Full experiment configuration
    :return: tuple of (catalog_choice, prompt_choice, statement_choice)
    """
    catalog_choice = "unknown_catalog"
    prompt_choice = "unknown_prompt"
    statement_choice = "unknown_statement"

    # Try to get choices from HydraConfig (when running under Hydra)
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            hydra_cfg = HydraConfig.get()
            hydra_choices = hydra_cfg.runtime.choices
            catalog_choice = hydra_choices.get("catalog", "unknown_catalog")
            prompt_choice = hydra_choices.get("prompt", "unknown_prompt")
            statement_choice = hydra_choices.get("statement", "unknown_statement")
            probe = hydra_choices.get("probe", "default_probe")
            return catalog_choice, prompt_choice, statement_choice, probe
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
                probe = cfg_hydra_choices.get("probe", "default_probe")
                return catalog_choice, prompt_choice, statement_choice, probe
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
        catalog_choice = catalog_info.get("name", "unknown_catalog")

    return catalog_choice, prompt_choice, statement_choice


# ---- Utility Functions ----


def move_incomplete_experiments(
    catalog_choice: str,
    prompt_choice: str,
    statement_choice: str,
    probe: str,
    seed: int,
    max_rounds: int = 10,
    base_dir: str = "outputs/runs",
    incomplete_dir: str = "outputs/incomplete_runs",
) -> int:
    """
    Move incomplete experiments with the given seed to the incomplete_runs directory.

    :param catalog_choice: catalog configuration name
    :param prompt_choice: prompt configuration name
    :param statement_choice: statement configuration name
    :param seed: random seed used for the experiment
    :param max_rounds: maximum number of rounds for a completed experiment
    :param base_dir: base directory for experiment outputs
    :param incomplete_dir: directory to move incomplete experiments to
    :return: number of incomplete experiments moved
    """
    import shutil

    base_path = pathlib.Path(base_dir)
    experiment_dir = (
        base_path / probe / catalog_choice / prompt_choice / statement_choice
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
            with open(config_path, "r") as f:
                config = json.load(f)

            # Check if the seed matches
            if config.get("seed") != seed:
                continue

            # Check if the experiment is incomplete
            rounds_dir = timestamp_dir / "rounds"
            if not rounds_dir.exists():
                continue

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
                    incomplete_path / catalog_choice / prompt_choice / statement_choice
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
    probe: str,
    seed: int,
    max_rounds: int = 10,
    base_dir: str = "outputs/runs",
) -> bool:
    """
    Check if an experiment with the given parameters has already been completed.

    :param catalog_choice: catalog configuration name
    :param prompt_choice: prompt configuration name
    :param statement_choice: statement configuration name
    :param seed: random seed used for the experiment
    :param max_rounds: maximum number of rounds for a completed experiment
    :param base_dir: base directory for experiment outputs
    :return: True if a completed experiment exists, False otherwise
    """
    base_path = pathlib.Path(base_dir)
    experiment_dir = (
        base_path / probe / catalog_choice / prompt_choice / statement_choice
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
            with open(config_path, "r") as f:
                config = json.load(f)

            # Check if the seed matches
            if config.get("seed") != seed:
                continue

            # Check if the experiment completed the required number of rounds
            # Rounds are 0-indexed, so round_0 to round_{max_rounds-1}
            rounds_dir = timestamp_dir / "rounds"
            if not rounds_dir.exists():
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


# ---- Utility Classes ----

# class Logger:
#     def __init__(self, cfg: Dict):
#         self.cfg = cfg

#     def info(self, msg):
#         """
#         Log the message.

#         :param msg: message (str)
#         :return: None
#         """
#         print(msg)


class IOManager:
    def __init__(self, cfg: Dict, experiment_cfg: Dict = None):
        """
        Initialize IOManager with experiment-specific directories and database tracking.

        :param cfg: IO configuration dictionary
        :param experiment_cfg: Full experiment configuration for recording
        """
        # Extract experiment name components
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Get experiment choices using the helper function
        catalog_choice, prompt_choice, statement_choice, probe_choice = (
            get_experiment_choices(experiment_cfg)
        )

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

        # Create hierarchical directory structure: outputs/runs/<catalog>/<prompt>/<statement>/<timestamp>
        base_dir = pathlib.Path(cfg.get("out_dir", "outputs/runs"))
        self.base_dir = base_dir
        self.out_dir = (
            base_dir
            / probe_choice
            / catalog_choice
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
        """
        Save the experiment HYDRA configuration to the experiment directory.
        Preserves Hydra runtime metadata including choices.
        """
        config_path = self.out_dir / "config.json"
        # Convert OmegaConf to dict if needed
        try:
            from omegaconf import OmegaConf

            if hasattr(self.experiment_cfg, "__dict__"):
                # Resolve most values but try to preserve hydra metadata
                config_dict = OmegaConf.to_container(self.experiment_cfg, resolve=True)

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
                config_dict = self.experiment_cfg
        except ImportError:
            config_dict = self.experiment_cfg

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def save_json(self, filename: str, data: dict) -> None:
        """
        save a JSON file inside the experiment's output directory.
        automatically creates directories if needed.
        """

        path = self.out_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def _round_path(self, t: int) -> str | pathlib.Path:
        """
        Generate the file path for a given round.
        Now uses the rounds subdirectory for better organization.

        :param t: round (int)
        :return: the file path
        """
        rounds_dir = self.out_dir / "rounds"
        rounds_dir.mkdir(parents=True, exist_ok=True)

        p = rounds_dir / f"round_{t}"
        p.mkdir(parents=True, exist_ok=True)

        return p

    def cache_activation(
        self, agent_id: int, t: int, activation: Any
    ) -> Optional[pathlib.Path]:
        """
        Save the activations of a given agent at a given round.

        :param agent_id: agent id (int)
        :param t: round (int)
        :param activation: activation generated by the agent (np array)
        :return: the file path
        """
        if self.save_activations:
            raise NotImplementedError("Activation saving not implemented yet.")
        else:
            return None
        # if not self.save_activations:
        #     return self._round_path(t) / f'{agent_id}.activation.ignore'

        # # TODO: decide tensor format (npy/pt) and save; placeholder json meta
        # p = self._round_path(t) / f'{agent_id}.activation.json'
        # with open(p, 'w') as f:
        #     json.dump({'shape': getattr(activation, 'shape', None), 'meta': 'TODO'}, f)

        # return p

    def checkpoint(self, t: int, agents: Dict[int, Any], metrics: Any) -> None:
        """
        Save a checkpoint for round t, including beliefs of agents and round metrics.
        Organizes files in a structured way within the experiment directory.

        :param t: round (int)
        :param agents: dictionary of agents (Dict[int, Agent])
        :param metrics: MetricsTracker object
        :return: None
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
                rec = {
                    "round": t,
                    "belief": b,
                    "agent": a.id,
                    "role": a.role,
                    "score": s,
                }
                f.write(json.dumps(rec) + "\n")

        # detailed belief scores per agent
        scores_path = round_dir / "beliefs_detailed.json"
        a = metrics.per_update
        detailed_scores = [u for u in a if u.get("round") == t]
        with open(scores_path, "w") as f:
            json.dump(detailed_scores, f, indent=2)

        # save the most recent per-round metrics
        if metrics.per_round:
            metrics_path = round_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics.per_round[-1], f, indent=2)

    def save_artifacts(
        self, agents: Dict[int, Any], network: Any, metrics: Any
    ) -> None:
        """
        Save different artifacts in an organized manner.
        Creates separate subdirectories for different types of outputs.

        :param agents: dictionary of agents (Dict[int, Agent])
        :param network: Network object
        :param metrics: MetricsTracker object
        :return:
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
                "beliefs": a.beliefs,
                "belief_scores": a._belief_score,
                "role": a.role if hasattr(a, "role") else "unknown",
                "model": a.model_name if hasattr(a, "model_name") else "unknown",
            }

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
        """
        Check if agent beliefs are stable.

        :param metrics: MetricsTracker object
        :param t: round (int)
        :return:
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


# class Aggregators:
#     @staticmethod
#     def aggregate(neighbors, beliefs, method, k=None):
#         """
#         Returns a neighbor view that PromptTemplate knows how to render.

#         :param neighbors: list of neighbor ids
#         :param beliefs: dictionary of neighbor beliefs (Dict[str, float])
#         :param method: type of aggregation in ['list_all'] (str)
#         :param k: optional, number of neighbors to consider (int)
#         :return: neighbor view
#         """

#         # todo: do we want others?

#         if method == "list_all":
#             # todo
#             raise NotImplementedError
#         else:
#             raise NotImplementedError(f"aggregation method '{method}'")


def set_seed(seed: int) -> None:
    random.seed(seed)
