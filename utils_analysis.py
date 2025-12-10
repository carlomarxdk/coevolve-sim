#!/usr/bin/env python3
"""
Analyze experiments stored in outputs/runs directory.

This script scans all experiment runs and extracts key attributes including:
- Experiment timestamp and ID
- Random seed for reproducibility
- Models used (e.g., llama-base, llama-biomed)
- Statement being evaluated
- Prompt configuration (type and version)
- Network configuration (generator, nodes, edges, probability)
- Number of rounds completed
- Final consensus metrics and network statistics

The output can be saved as CSV or JSON for further analysis.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_experiment_config(experiment_dir: Path) -> Dict[str, Any]:
    """Load experiment configuration from config.json file.

    Args:
        experiment_dir: Path to the experiment directory.

    Returns:
        Dictionary containing the experiment configuration, or empty dict if not found.
    """
    config_path = experiment_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


@dataclass
class ExpData:
    """Data structure to hold experiment attributes."""

    catalog: str
    prompt: str
    statement: str

    @property
    def path(self) -> Path:
        """Path to the experiment outputs (that contains all the runs)"""
        return Path("outputs/runs/") / self.catalog / self.prompt / self.statement

    def get_paths_to_runs(self) -> list[Path]:
        """Get all folder paths within the experiment runs path"""
        return [p for p in self.path.iterdir() if p.is_dir()]

    @property
    def config_name(self) -> str:
        """Get a string representing the configuration name."""
        return f"{self.catalog}_{self.prompt}_{self.statement}"


def load_experiment_results(experiment_dir: Path) -> Dict[str, Any]:
    """Load experiment results if available."""
    results = {}

    # Load per-round metrics
    metrics_path = experiment_dir / "results" / "per_round_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            results["per_round_metrics"] = json.load(f)

    # Load final metrics
    final_path = experiment_dir / "results" / "final_metrics.json"
    if final_path.exists():
        with open(final_path, "r") as f:
            results["final_metrics"] = json.load(f)

    # Load agents data
    agents_path = experiment_dir / "results" / "agents_data.json"
    if agents_path.exists():
        with open(agents_path, "r") as f:
            results["agents_data"] = json.load(f)

    # Load network manifest
    network_path = experiment_dir / "results" / "network_manifest.json"
    if network_path.exists():
        with open(network_path, "r") as f:
            results["network_manifest"] = json.load(f)

    return results


def extract_experiment_attributes(experiment_dir: Path) -> Dict[str, Any]:
    """Extract key attributes from an experiment."""
    config = load_experiment_config(experiment_dir)
    results = load_experiment_results(experiment_dir)

    attributes = {
        "experiment_id": experiment_dir.name,
        "path": str(experiment_dir),
    }

    # Extract configuration attributes
    if config:
        attributes["seed"] = config.get("seed")

        # Statement information
        statement_info = config.get("statement", {})
        attributes["statement_id"] = statement_info.get("id")
        attributes["statement_text"] = statement_info.get("statement")

        # Experiment settings
        exp_config = config.get("experiment", {})
        attributes["max_rounds"] = exp_config.get("max_rounds")
        attributes["ordering"] = exp_config.get("ordering")

        # Network configuration
        network_config = config.get("network", {})
        attributes["network_generator"] = network_config.get("generator")
        attributes["network_n"] = network_config.get("params", {}).get("n")
        attributes["network_p"] = network_config.get("params", {}).get("p")

        # Agent catalog - extract models and roles
        catalog = config.get("agents", {}).get("catalog", [])
        models = [agent.get("name", "unknown") for agent in catalog]
        roles = [agent.get("role", "unknown") for agent in catalog]
        attributes["num_agents"] = len(catalog)
        attributes["models"] = list(set(models))
        attributes["roles"] = list(set(roles))

        # Prompt configuration
        if catalog and len(catalog) > 0:
            prompt_info = catalog[0].get("prompt", {})
            attributes["prompt_type"] = prompt_info.get("type")
            attributes["prompt_version"] = prompt_info.get("version")

    # Extract results attributes
    if results:
        # Number of rounds completed
        if "per_round_metrics" in results:
            per_round = results["per_round_metrics"].get("per_round", [])
            attributes["rounds_completed"] = len(per_round)

            # Final round metrics
            if per_round:
                final_round = per_round[-1]
                attributes["final_consensus_fraction"] = final_round.get(
                    "consensus", {}
                ).get("consensus_fraction")
                attributes["final_magnetization"] = final_round.get(
                    "consensus", {}
                ).get("magnetization")
                attributes["final_entropy"] = final_round.get("entropy")

        # Final metrics
        if "final_metrics" in results:
            final = results["final_metrics"]
            attributes["final_majority_fraction"] = final.get("majority_fraction")
            attributes["num_belief_clusters"] = final.get("num_belief_clusters")

        # Network metrics
        if "network_manifest" in results:
            graph_info = results["network_manifest"].get("graph", {})
            attributes["num_nodes"] = graph_info.get("num_nodes")
            attributes["num_edges"] = graph_info.get("num_edges")
            attributes["avg_clustering"] = graph_info.get("avg_clustering")
            attributes["density"] = graph_info.get("density")

    return attributes


def list_experiments(runs_dir: str = "outputs/runs") -> List[Dict[str, Any]]:
    """
    List all experiments in the runs directory with their attributes.

    Args:
        runs_dir: Path to the runs directory

    Returns:
        List of dictionaries containing experiment attributes
    """
    runs_path = Path(runs_dir)

    if not runs_path.exists():
        print(f"Warning: Runs directory '{runs_dir}' does not exist")
        return []

    experiments = []

    # Iterate through all subdirectories in runs
    for exp_dir in sorted(runs_path.iterdir()):
        if exp_dir.is_dir():
            try:
                attributes = extract_experiment_attributes(exp_dir)
                experiments.append(attributes)
            except Exception as e:
                print(
                    f"Warning: Failed to process {exp_dir.name}: {e}", file=sys.stderr
                )

    return experiments


def save_experiments_to_csv(
    experiments: List[Dict[str, Any]], output_path: str = "experiments_summary.csv"
):
    """Save experiment list to CSV file."""
    if not experiments:
        print("No experiments to save")
        return

    df = pd.DataFrame(experiments)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(experiments)} experiments to {output_path}")


def save_experiments_to_json(
    experiments: List[Dict[str, Any]], output_path: str = "experiments_summary.json"
):
    """Save experiment list to JSON file."""
    if not experiments:
        print("No experiments to save")
        return

    with open(output_path, "w") as f:
        json.dump(experiments, f, indent=2)
    print(f"Saved {len(experiments)} experiments to {output_path}")


def print_experiments_summary(experiments: List[Dict[str, Any]]):
    """Print a summary of experiments."""
    if not experiments:
        print("No experiments found")
        return

    print(f"\n{'='*80}")
    print(f"Found {len(experiments)} experiments")
    print(f"{'='*80}\n")

    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['experiment_id']}")
        print(f"   Seed: {exp.get('seed', 'N/A')}")
        print(f"   Statement: {exp.get('statement_text', 'N/A')[:60]}...")
        print(f"   Models: {', '.join(exp.get('models', []))}")
        print(f"   Agents: {exp.get('num_agents', 'N/A')}")
        print(
            f"   Rounds: {exp.get('rounds_completed', 'N/A')}/{exp.get('max_rounds', 'N/A')}"
        )
        print(
            f"   Network: {exp.get('network_generator', 'N/A')} (n={exp.get('network_n', 'N/A')}, p={exp.get('network_p', 'N/A')})"
        )
        print(
            f"   Prompt: {exp.get('prompt_type', 'N/A')} v{exp.get('prompt_version', 'N/A')}"
        )
        if exp.get("final_consensus_fraction") is not None:
            print(
                f"   Final consensus: {exp.get('final_consensus_fraction', 'N/A'):.2%}"
            )
        print()


def main():
    """Main function to run experiment analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze LLM debate simulation experiments"
    )
    parser.add_argument(
        "--runs-dir",
        default="outputs/runs",
        help="Path to runs directory (default: outputs/runs)",
    )
    parser.add_argument(
        "--output-csv",
        default="experiments_summary.csv",
        help="Output CSV file path (default: experiments_summary.csv)",
    )
    parser.add_argument(
        "--output-json",
        default="experiments_summary.json",
        help="Output JSON file path (default: experiments_summary.json)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output files, only print summary",
    )

    args = parser.parse_args()

    # List all experiments
    experiments = list_experiments(args.runs_dir)

    # Print summary
    print_experiments_summary(experiments)

    # Save to files if requested
    if not args.no_save and experiments:
        save_experiments_to_csv(experiments, args.output_csv)
        save_experiments_to_json(experiments, args.output_json)


if __name__ == "__main__":
    main()
