"""Belief evolution plotting for main manuscript figures.

This module generates publication-ready figures showing belief evolution over
rounds for multi-run experiments. Figures display collective accuracy and
belief distributions organized by statement type and experiment condition.

Output figures follow Nature Publishing Group (NPG) guidelines with the
repository's font standards (Georgia, Cabin, Menlo) and PDF-only export.
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontManager

from src.metrics.general import collective_accuracy_label, label_distribution

# Import cmcrameri for scientific colormaps (colorblind-safe)
try:
    import cmcrameri.cm as cmc  # type: ignore

    CMAP_AVAILABLE = True
except ImportError:
    CMAP_AVAILABLE = False


def _get_batlowS_colors() -> dict:
    """Get colors from cmcrameri batlowS categorical colormap.

    Uses the batlowS discrete/categorical colormap which provides
    perceptually uniform and colorblind-safe colors for categorical data.

    Returns:
        Dictionary mapping color names to hex color codes.
    """
    if CMAP_AVAILABLE:
        # batlowS is a categorical colormap with discrete colors
        batlowS = cmc.batlowS
        colors = {
            "primary": mpl.colors.to_hex(batlowS(3)),  # First color
            "secondary": mpl.colors.to_hex(batlowS(4)),  # Fourth color
            "tertiary": mpl.colors.to_hex(batlowS(2)),  # Third color
            "accent": mpl.colors.to_hex(batlowS(1)),  # Second color
        }
    else:
        # Fallback to Okabe-Ito colors if cmcrameri not available
        colors = {
            "primary": "#0072B2",  # blue
            "secondary": "#E69F00",  # orange
            "tertiary": "#009E73",  # green
            "accent": "#CC79A7",  # purple
        }
    return colors


BATLOWS_COLORS = _get_batlowS_colors()


def _configure_fonts() -> None:
    """Configure matplotlib fonts with fallback to defaults if specified fonts are unavailable."""
    # Default fallback fonts (widely available on most systems)
    default_fallbacks = {
        "serif": ["DejaVu Serif"],
        "sans-serif": ["DejaVu Sans"],
        "monospace": ["DejaVu Sans Mono"],
    }

    # Desired fonts with fallbacks (these may not be available on all systems)
    desired_fonts = {
        "serif": ["Georgia"] + default_fallbacks["serif"] + ["Times New Roman"],
        "sans-serif": ["Cabin"]
        + default_fallbacks["sans-serif"]
        + ["Helvetica", "Arial"],
        "monospace": ["Menlo"] + default_fallbacks["monospace"] + ["Courier New"],
    }

    # Get list of available fonts
    fm = FontManager()
    available_fonts = {f.name for f in fm.ttflist}

    # Build font lists with only available fonts, warn if primary font unavailable
    configured_fonts = {}
    for family, fonts in desired_fonts.items():
        available = [f for f in fonts if f in available_fonts]
        if not available:
            # If no fonts available, use default fallbacks
            warnings.warn(
                f"None of the desired {family} fonts {fonts} found. "
                f"Using system default {family} font.",
                UserWarning,
            )
            configured_fonts[family] = default_fallbacks[family]
        else:
            if fonts[0] not in available_fonts:
                warnings.warn(
                    f"Primary {family} font '{fonts[0]}' not found. "
                    f"Using fallback: {available[0]}.",
                    UserWarning,
                )
            configured_fonts[family] = available

    # Apply configuration with fallback behavior
    mpl.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.serif": configured_fonts["serif"],
            "font.sans-serif": configured_fonts["sans-serif"],
            "font.monospace": configured_fonts["monospace"],
            # Sizes tuned for LaTeX paper inclusion
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            # Nature-style tick direction
            "xtick.direction": "in",
            "ytick.direction": "in",
            # Remove top and right spines globally
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


_configure_fonts()


def find_multi_run_experiments(runs_dir: str = "outputs/runs") -> Dict[str, List[Path]]:
    """Discover multi-run experiments organized by configuration.

    Experiments are grouped by: probe/catalog/prompt/statement

    Args:
        runs_dir: Path to the runs directory.

    Returns:
        Dictionary mapping configuration keys to list of run directories.
    """
    runs_path = Path(runs_dir)

    if not runs_path.exists():
        print(f"Warning: Runs directory '{runs_dir}' does not exist")
        return {}

    multi_run_configs: Dict[str, List[Path]] = defaultdict(list)

    # Navigate the directory structure: probe/catalog/prompt/statement/timestamp/
    for probe_dir in sorted(runs_path.iterdir()):
        if not probe_dir.is_dir() or probe_dir.name.startswith("."):
            continue

        for catalog_dir in sorted(probe_dir.iterdir()):
            if not catalog_dir.is_dir() or catalog_dir.name.startswith("."):
                continue

            for prompt_dir in sorted(catalog_dir.iterdir()):
                if not prompt_dir.is_dir() or prompt_dir.name.startswith("."):
                    continue

                for statement_dir in sorted(prompt_dir.iterdir()):
                    if not statement_dir.is_dir() or statement_dir.name.startswith("."):
                        continue

                    # Each timestamp directory is a run
                    run_dirs = [
                        d
                        for d in sorted(statement_dir.iterdir())
                        if d.is_dir()
                        and (d / "config.json").exists()
                        and not (
                            len(d.name) >= 3
                            and d.name[-3] == "_"
                            and len(d.name[-2:]) == 2
                        )
                    ]

                    if len(run_dirs) > 1:  # Multi-run experiment
                        config_key = (
                            f"{probe_dir.name}/"
                            f"{catalog_dir.name}/"
                            f"{prompt_dir.name}/"
                            f"{statement_dir.name}"
                        )
                        multi_run_configs[config_key] = run_dirs

    return dict(multi_run_configs)


def load_multi_run_belief_data(
    run_dirs: List[Path],
) -> Tuple[np.ndarray, int, np.ndarray, List[int]]:
    """Load belief labels and scores across multiple runs.

    Args:
        run_dirs: List of run directories to load data from.

    Returns:
        Tuple containing:
            - labels: Array of shape (num_runs, num_rounds, num_agents).
            - correct_label: Integer indicating the correct belief label.
            - scores: Array of shape (num_runs, num_rounds, num_agents).
            - rounds: List of round numbers.
    """
    labels_list: List[np.ndarray] = []
    scores_list: List[np.ndarray] = []
    rounds_ref: Optional[List[int]] = None
    correct_label: Optional[int] = None

    for run_dir in run_dirs:
        metrics_path = run_dir / "results" / "per_round_metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        per_round = metrics.get("per_round", [])
        if not per_round:
            continue

        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue

        with open(config_path, "r") as f:
            config = json.load(f)
            correct_label_run = int(
                config.get("statement", {}).get("label", {}).get("correct")
            )

        # Extract rounds, labels, scores
        rounds = [r["round"] for r in per_round]
        labels_run = np.array(
            [r["belief"]["label"]["values"] for r in per_round], dtype=float
        )
        scores_run = np.array(
            [r["belief"]["score"]["values"] for r in per_round], dtype=float
        )

        # Consistency check on rounds
        if rounds_ref is None:
            rounds_ref = rounds
        elif rounds != rounds_ref:
            print(
                f"Warning: run {run_dir.name} has different rounds; skipping.",
                file=sys.stderr,
            )
            continue

        # Consistency check on correct_label
        if correct_label is None:
            correct_label = correct_label_run
        elif correct_label != correct_label_run:
            print(
                f"Warning: run {run_dir.name} has different correct_label; skipping.",
                file=sys.stderr,
            )
            continue

        labels_list.append(labels_run)
        scores_list.append(scores_run)

    if not labels_list or rounds_ref is None or correct_label is None:
        return np.array([]), 0, np.array([]), []

    labels = np.stack(labels_list, axis=0)  # (R, T, A)
    scores = np.stack(scores_list, axis=0)  # (R, T, A)

    return labels, correct_label, scores, rounds_ref


def parse_experiment_and_statement(config_name: str) -> Tuple[str, str]:
    """Map a config_name string to human-readable experiment and statement labels.

    Args:
        config_name: Configuration path string (e.g., 'probe/catalog/prompt/stmt').

    Returns:
        Tuple of (experiment_label, statement_label) for display in plots.
    """
    # Experiment type
    if "random_experts" in config_name:
        experiment = "Experts (Randomized Roles)"
    elif "experts" in config_name:
        experiment = "Experts (Matching Roles)"
    elif "only_llms" in config_name:
        experiment = "Generalists (No Roles)"
    elif "random_roles" in config_name:
        experiment = "Generalists (Randomized Roles)"
    else:
        experiment = "Experiment"

    # Statement type
    if "true_doc_1" in config_name:
        statement = "True Statement (Doctor is Correct)"
    elif "true_rest_1" in config_name:
        statement = "True Statement (Doctor is Incorrect)"
    elif "false_doc_1" in config_name:
        statement = "False Statement (Doctor is Correct)"
    elif "false_rest_1" in config_name:
        statement = "False Statement (Doctor is Incorrect)"
    else:
        statement = "STATEMENT"

    return experiment, statement


def slugify_statement(statement: str) -> str:
    """Turn a human-readable statement label into a filesystem-safe slug.

    Converts display labels to lowercase strings safe for use in filenames.
    For example: "True (Doc. Correct)" -> "true_doc_correct".

    Args:
        statement: Human-readable statement label.

    Returns:
        Filesystem-safe slug string.
    """
    s = statement.lower()
    for ch in ["(", ")", ".", ","]:
        s = s.replace(ch, "")
    s = s.replace(" ", "_").replace("/", "_")
    return s


def plot_statement_multi_experiment(
    statement_label: str,
    statement_slug: str,
    experiment_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
    show_belief_score: bool = False,
) -> None:
    """Create a single figure for a given statement type, with experiments as columns.

    For each statement (e.g., "True (Doc. Correct)"):
        - Top row: Collective Accuracy for experts, only_llms, random_roles
        - Middle row: Belief Distributions for experts, only_llms, random_roles
        - Bottom row (optional): Score Progression if show_belief_score is True

    Args:
        statement_label: Pretty label for title (e.g., "True (Doc. Correct)").
        statement_slug: Filesystem-safe slug for filename.
        experiment_data: Mapping from experiment_name to data dict containing
            labels, scores, rounds, and correct_label. Valid experiment names
            are "experts", "only_llms", and "random_roles".
        output_dir: Directory to save the figure.
        show_belief_score: If True, include a third row showing belief score
            progression.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments_order = ["experts", "random_roles", "only_llms"]
    experiment_labels = {
        "experts": "Experts\n(Matching Roles)",
        "only_llms": "Generalists\n(No Roles)",
        "random_roles": "Generalists\n(Randomized Roles)",
    }

    # Configure grid layout based on whether belief score row is shown
    if show_belief_score:
        # 3 rows: title, exp_titles, CA, CA_legend, Dist, Dist_legend, Score, Score_legend
        fig = plt.figure(figsize=(7.2, 8.0))
        gs = gridspec.GridSpec(
            8,
            3,
            height_ratios=[0.04, 0.04, 0.26, 0.06, 0.26, 0.06, 0.26, 0.06],
            width_ratios=[1, 1, 1],
            hspace=0.70,
            wspace=0.45,
            figure=fig,
        )
        letters_bot = ["G", "H", "I"]
    else:
        # 2 rows: title, exp_titles, CA, CA_legend, Dist, Dist_legend
        fig = plt.figure(figsize=(7.2, 5.5))
        gs = gridspec.GridSpec(
            6,
            3,
            height_ratios=[0.04, 0.04, 0.36, 0.08, 0.36, 0.08],
            width_ratios=[1, 1, 1],
            hspace=0.75,
            wspace=0.45,
            figure=fig,
        )

    # Main title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.5,
        f"Belief Evolution over Rounds: {statement_label}",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        fontfamily="sans-serif",
        color="#444444",
    )

    # Panel letters
    letters_top = ["A", "B", "C"]
    letters_mid = ["D", "E", "F"]

    # To build shared legends
    ca_handles, ca_labels = None, None
    dist_handles, dist_labels = None, None
    score_handles, score_labels = None, None

    for col, exp_name in enumerate(experiments_order):
        pretty_exp = experiment_labels[exp_name]

        # Experiment Title Row
        ax_exp = fig.add_subplot(gs[1, col])
        ax_exp.axis("off")
        # Top row: Collective Accuracy
        ax_ca = fig.add_subplot(gs[2, col])
        # Middle row: Belief Distributions
        ax_dist = fig.add_subplot(gs[4, col])
        # Bottom row: Score Progression (only if enabled)
        ax_score = None
        if show_belief_score:
            ax_score = fig.add_subplot(gs[6, col])

        # Experiment Title
        ax_exp.text(
            0.5,
            0.5,
            pretty_exp,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#555555",
        )

        data = experiment_data.get(exp_name, None)

        if data is None or data["labels"].size == 0:
            # No data for this experiment & statement
            axes_to_clear = [ax_ca, ax_dist]
            if ax_score is not None:
                axes_to_clear.append(ax_score)
            for ax in axes_to_clear:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"No data\n{pretty_exp}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#777777",
                )
            continue

        labels = data["labels"]
        scores = data["scores"]
        rounds = data["rounds"]
        correct_label = data["correct_label"]
        rounds_arr = np.array(rounds)
        num_runs, num_rounds, _ = labels.shape

        # Collective Accuracy
        ax_ca.text(
            -0.15,
            1.08,
            f"{letters_top[col]}",
            transform=ax_ca.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#555555",
        )

        collective_acc_score = collective_accuracy_label(labels, correct_label).mean(
            axis=2
        )  # (R, T)
        ca_mean = collective_acc_score.mean(axis=0)
        ca_std = collective_acc_score.std(axis=0)

        ax_ca.plot(
            rounds_arr,
            ca_mean,
            linewidth=1.5,
            label="Mean",
            color=BATLOWS_COLORS["primary"],
        )
        ax_ca.fill_between(
            rounds_arr,
            ca_mean - ca_std,
            ca_mean + ca_std,
            alpha=0.3,
            label="±1 std",
            color=BATLOWS_COLORS["primary"],
        )

        ax_ca.set_xlabel("Round")
        ax_ca.set_ylabel("Group Accuracy")
        ax_ca.tick_params(axis="both", direction="in")
        ax_ca.spines["top"].set_visible(False)
        ax_ca.spines["right"].set_visible(False)
        ax_ca.grid(True, alpha=0.3)
        ax_ca.set_ylim([0.0, 1.05])
        ax_ca.set_xlim(rounds_arr[0], rounds_arr[-1])
        ax_ca.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Capture legend handles/labels from first CA axis
        if ca_handles is None:
            ca_handles, ca_labels = ax_ca.get_legend_handles_labels()

        # Belief Distributions
        ax_dist.text(
            -0.15,
            1.08,
            f"{letters_mid[col]}",
            transform=ax_dist.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#555555",
        )

        avg_fractions_final, std_fractions_final = label_distribution(labels[:, -1, :])
        avg_fractions_initial, std_fractions_initial = label_distribution(
            labels[:, 0, :]
        )

        order = [1, 0, 2]
        avg_fractions_final = avg_fractions_final[order]
        std_fractions_final = std_fractions_final[order]
        avg_fractions_initial = avg_fractions_initial[order]
        std_fractions_initial = std_fractions_initial[order]

        label_names = ["True", "False", "Neither"]
        x = np.arange(len(label_names))
        width = 0.35

        ax_dist.bar(
            x - width / 2,
            avg_fractions_initial,
            width,
            alpha=0.7,
            edgecolor="black",
            yerr=std_fractions_initial,
            capsize=3,
            label="Initial",
            color=BATLOWS_COLORS["primary"],
        )
        ax_dist.bar(
            x + width / 2,
            avg_fractions_final,
            width,
            alpha=0.7,
            edgecolor="black",
            yerr=std_fractions_final,
            capsize=3,
            label="Final",
            color=BATLOWS_COLORS["secondary"],
        )

        ax_dist.set_xlabel("Belief Label")
        ax_dist.set_ylabel("Fraction")
        ax_dist.tick_params(axis="both", direction="in")
        ax_dist.spines["top"].set_visible(False)
        ax_dist.spines["right"].set_visible(False)
        ax_dist.set_xticks(x)
        ax_dist.set_xticklabels(label_names)
        ax_dist.set_ylim(0, 1.1)
        ax_dist.grid(True, axis="y", alpha=0.3)

        # Capture legend handles/labels from first dist axis
        if dist_handles is None:
            dist_handles, dist_labels = ax_dist.get_legend_handles_labels()

        # Score Progression (only if enabled)
        if ax_score is not None:
            ax_score.text(
                -0.15,
                1.08,
                f"{letters_bot[col]}",
                transform=ax_score.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#555555",
            )

            # Calculate mean scores over time
            score_mean_per_round = scores.mean(axis=2).mean(axis=0)
            score_std_per_round = scores.mean(axis=2).std(axis=0)

            ax_score.plot(
                rounds_arr,
                score_mean_per_round,
                color=BATLOWS_COLORS["accent"],
                linewidth=1.5,
                label="Mean P(True)",
            )
            ax_score.fill_between(
                rounds_arr,
                score_mean_per_round - score_std_per_round,
                score_mean_per_round + score_std_per_round,
                alpha=0.3,
                color=BATLOWS_COLORS["accent"],
            )

            ax_score.set_xlabel("Round")
            ax_score.set_ylabel("Belief Score")
            ax_score.tick_params(axis="both", direction="in")
            ax_score.spines["top"].set_visible(False)
            ax_score.spines["right"].set_visible(False)
            ax_score.set_ylim(0.0, 1.05)
            ax_score.grid(True, alpha=0.3)
            ax_score.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            # Capture legend handles/labels from first score axis
            if score_handles is None:
                score_handles, score_labels = ax_score.get_legend_handles_labels()

    # Shared legends
    ax_ca_legend = fig.add_subplot(gs[3, :])
    ax_ca_legend.axis("off")
    if ca_handles and ca_labels:
        ax_ca_legend.legend(
            ca_handles,
            ca_labels,
            loc="center",
            ncol=2,
            fontsize=8,
            frameon=False,
        )

    ax_dist_legend = fig.add_subplot(gs[5, :])
    ax_dist_legend.axis("off")
    if dist_handles and dist_labels:
        ax_dist_legend.legend(
            dist_handles,
            dist_labels,
            loc="center",
            ncol=2,
            fontsize=8,
            frameon=False,
        )

    if show_belief_score:
        ax_score_legend = fig.add_subplot(gs[7, :])
        ax_score_legend.axis("off")
        if score_handles and score_labels:
            ax_score_legend.legend(
                score_handles,
                score_labels,
                loc="center",
                ncol=1,
                fontsize=8,
                frameon=False,
            )

    out_path = output_dir / f"{statement_slug}_belief_evolution_by_experiment.pdf"
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    print(
        f"Saved statement-level belief evolution plot for '{statement_label}' to {out_path}"
    )
    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for belief evolution analysis.

    Returns:
        Namespace containing parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Analyze belief evolution across multiple experiment runs"
    )
    parser.add_argument(
        "--runs-dir",
        default="outputs/runs",
        help="Path to runs directory (default: outputs/runs)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/plots",
        help="Directory to save analysis outputs (default: outputs/plots)",
    )
    parser.add_argument(
        "--config",
        help=(
            "Specific configuration to analyze (e.g., "
            "sawmil/random_roles/wR_L/trueV_2). "
            "If provided, uses the original single-config plotting."
        ),
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=1,
        help="Minimum number of runs required for analysis (default: 1)",
    )
    parser.add_argument(
        "--show-belief-score",
        action="store_true",
        help="Include belief score progression as a third row (default: off)",
    )

    return parser.parse_args()


def main() -> None:
    """Run the belief evolution analysis and generate publication-ready figures."""
    args = parse_args()
    output_dir = Path(args.output_dir)

    # Discover multi-run experiments
    print("Discovering multi-run experiments...")
    multi_run_configs = find_multi_run_experiments(args.runs_dir)

    # Filter by minimum runs
    multi_run_configs = {
        k: v for k, v in multi_run_configs.items() if len(v) >= args.min_runs
    }

    if not multi_run_configs:
        print(f"No multi-run experiments found with at least {args.min_runs} runs.")
        return

    print(f"\nFound {len(multi_run_configs)} multi-run configurations:")
    for config_name, run_dirs in multi_run_configs.items():
        print(f"  {config_name}: {len(run_dirs)} runs")

    # statements[(probe, statement_pretty)][experiment_name] -> data dict
    statements: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)

    allowed_experiments = {"experts", "only_llms", "random_roles"}

    for config_name, run_dirs in multi_run_configs.items():
        print(f"\n{'=' * 60}")
        print(f"Loading: {config_name}")
        print(f"{'=' * 60}")

        labels, correct_label, scores, rounds = load_multi_run_belief_data(run_dirs)
        if labels.size == 0:
            print("No valid data found for this config, skipping.")
            continue

        parts = config_name.split("/")
        if len(parts) != 4:
            print(f"Unexpected config format: {config_name}, skipping.")
            continue

        probe, experiment_name, _prompt, _statement_raw = parts

        # Skip experiments we don't want (e.g., random_experts)
        if experiment_name not in allowed_experiments:
            print(f"Skipping experiment '{experiment_name}' (not in allowed set).")
            continue

        # Use helper to get statement label
        _pretty_experiment, statement_pretty = parse_experiment_and_statement(
            config_name
        )

        key = (probe, statement_pretty)
        statements[key][experiment_name] = {
            "labels": labels,
            "scores": scores,
            "rounds": rounds,
            "correct_label": correct_label,
            "config_name": config_name,
        }

    if not statements:
        print("No statement-level data available after loading; exiting.")
        return

    # Create one figure per (probe, statement)
    for (probe, statement_label), exp_data in statements.items():
        print(
            f"\nCreating statement-level figure for probe={probe}, statement='{statement_label}'"
        )

        statement_slug = slugify_statement(statement_label)
        # Save under outputs/plots/<probe>/
        output_dir_stmt = output_dir / probe

        plot_statement_multi_experiment(
            statement_label=statement_label,
            statement_slug=statement_slug,
            experiment_data=exp_data,
            output_dir=output_dir_stmt,
            show_belief_score=args.show_belief_score,
        )


if __name__ == "__main__":
    main()
