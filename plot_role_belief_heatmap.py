#!/usr/bin/env python
"""Role-based belief strength heatmap grids grouped by experiment.

For each experiment type (e.g., experts, only_llms, random_roles, random_experts),
this script creates a single figure with a 2x2 grid of heatmaps:

    Columns:  True Statement        |  False Statement
    Rows:     Doctor is Correct     |  Doctor is Incorrect

Each panel shows average belief strength (or collective accuracy) by role across
rounds, averaged over multiple runs for a given statement.

Directory structure assumed:
    outputs/runs/probe/experiment/prompt/statement/timestamp/

Multi-run configs are grouped by:
    probe/experiment/prompt/statement
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontManager

try:
    import cmcrameri.cm as cmc  # type: ignore  # noqa: F401

    CMAP_AVAILABLE = True
except ImportError:
    CMAP_AVAILABLE = False


# Okabe-Ito color palette for accessibility (colorblind-safe)
OKABE_ITO_COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "yellow": "#F0E442",
    "sky_blue": "#56B4E9",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}


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
            # Note: Tick direction not set globally; heatmaps use specific
            # settings (no x ticks, y ticks outward) set per-axis in plotting.
            # Remove top and right spines globally
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


_configure_fonts()


# ---------------------------------------------------------------------------
# Basic helpers (adapted from existing scripts)
# ---------------------------------------------------------------------------


def _compute_collective_accuracy(scores: np.ndarray, correct_label: int) -> np.ndarray:
    """
    Compute collective accuracy from belief scores.

    Args:
        scores: Belief scores array (any shape)
        correct_label: The correct belief label (0 or 1)

    Returns:
        Collective accuracy values (same shape as input)
    """
    return 1 - np.abs(scores - correct_label)


def find_multi_run_experiments(runs_dir: str = "outputs/runs") -> Dict[str, List[Path]]:
    """
    Discover multi-run experiments organized by configuration.

    Experiments are grouped by: probe/experiment/prompt/statement

    Args:
        runs_dir: Path to the runs directory

    Returns:
        Dictionary mapping configuration keys to list of run directories:
            { "probe/experiment/prompt/statement": [run_dir1, run_dir2, ...], ... }
    """
    runs_path = Path(runs_dir)

    if not runs_path.exists():
        print(f"Warning: Runs directory '{runs_dir}' does not exist")
        return {}

    multi_run_configs: Dict[str, List[Path]] = defaultdict(list)

    # Navigate directory structure: probe/experiment/prompt/statement/timestamp/
    for probe_dir in sorted(runs_path.iterdir()):
        if not probe_dir.is_dir() or probe_dir.name.startswith("."):
            continue

        for experiment_dir in sorted(probe_dir.iterdir()):
            if not experiment_dir.is_dir() or experiment_dir.name.startswith("."):
                continue

            for prompt_dir in sorted(experiment_dir.iterdir()):
                if not prompt_dir.is_dir() or prompt_dir.name.startswith("."):
                    continue

                for statement_dir in sorted(prompt_dir.iterdir()):
                    if not statement_dir.is_dir() or statement_dir.name.startswith("."):
                        continue

                    run_dirs = [
                        d
                        for d in sorted(statement_dir.iterdir())
                        if d.is_dir() and (d / "config.json").exists()
                        # skip slurm "_xx" re-run dirs
                        and not (
                            len(d.name) >= 3
                            and d.name[-3] == "_"
                            and len(d.name[-2:]) == 2
                        )
                    ]

                    if len(run_dirs) > 1:
                        key = (
                            f"{probe_dir.name}/"
                            f"{experiment_dir.name}/"
                            f"{prompt_dir.name}/"
                            f"{statement_dir.name}"
                        )
                        multi_run_configs[key] = run_dirs

    return dict(multi_run_configs)


def load_role_belief_data(
    run_dirs: List[Path],
    metric: str = "belief_score",
) -> Tuple[Dict[str, np.ndarray], List[int], Optional[int]]:
    """
    Load belief scores or collective accuracy grouped by role across multiple runs.

    Args:
        run_dirs: List of run directories to load data from
        metric: Type of metric to use - "belief_score" or "collective_accuracy"

    Returns:
        Tuple of (role_scores, rounds, correct_label):
        - role_scores: Dict mapping role name to array of shape (num_runs, num_rounds)
        - rounds: List of round numbers
        - correct_label: The correct belief label (1 for true, -1 or 0/1; passed through)
    """
    from collections import defaultdict as ddict

    role_scores_by_run: Dict[str, List[np.ndarray]] = ddict(list)
    rounds_ref: Optional[List[int]] = None
    correct_label: Optional[int] = None

    for run_dir in run_dirs:
        metrics_path = run_dir / "results" / "per_round_metrics.json"
        agent_manifest_path = run_dir / "results" / "agent_manifest.json"
        config_path = run_dir / "config.json"

        if not all(
            p.exists() for p in [metrics_path, agent_manifest_path, config_path]
        ):
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        with open(agent_manifest_path, "r") as f:
            agent_manifest = json.load(f)

        with open(config_path, "r") as f:
            config = json.load(f)
            if correct_label is None:
                correct_label = int(
                    config.get("statement", {}).get("label", {}).get("correct", 1)
                )

        per_round = metrics.get("per_round", [])
        if not per_round:
            continue

        rounds = [r["round"] for r in per_round]

        if rounds_ref is None:
            rounds_ref = rounds
        elif rounds != rounds_ref:
            # If rounds don't match, skip this run for simplicity
            print(
                f"  Warning: round indices differ in {run_dir}, "
                f"skipping this run for role belief data."
            )
            continue

        scores_per_round = np.array(
            [r["belief"]["score"]["values"] for r in per_round],
            dtype=float,
        )

        # Convert to collective accuracy if requested
        if metric == "collective_accuracy" and correct_label is not None:
            scores_per_round = _compute_collective_accuracy(
                scores_per_round, correct_label
            )

        agent_roles = {
            int(agent_id): info["role"] for agent_id, info in agent_manifest.items()
        }

        role_to_agents: Dict[str, List[int]] = ddict(list)
        for agent_id, role in agent_roles.items():
            role_to_agents[role].append(agent_id)

        num_agents = scores_per_round.shape[1]
        for role, agent_ids in role_to_agents.items():
            valid_agent_ids = [aid for aid in agent_ids if aid < num_agents]
            if not valid_agent_ids:
                continue
            role_scores_this_run = scores_per_round[:, valid_agent_ids].mean(axis=1)
            role_scores_by_run[role].append(role_scores_this_run)

    role_scores: Dict[str, np.ndarray] = {}
    for role, run_scores_list in role_scores_by_run.items():
        if run_scores_list:
            role_scores[role] = np.stack(run_scores_list, axis=0)

    return role_scores, (rounds_ref if rounds_ref else []), correct_label


def parse_experiment_and_statement(config_name: str) -> Tuple[str, str]:
    """
    Map a config_name string to human-readable experiment and statement labels.

    Expected config_name format:
        probe/experiment/prompt/statement
    """
    # Experiment type (2nd path component)
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

    # Statement type (based on filename)
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


def slugify_experiment(experiment: str) -> str:
    """
    Turn a human-readable experiment label into a filesystem-safe slug.
    E.g., "Experts (Matching Roles)" -> "experts_matching_roles".
    """
    s = experiment.lower()
    for ch in ["(", ")", ".", ","]:
        s = s.replace(ch, "")
    s = s.replace(" ", "_").replace("/", "_")
    return s


# ---------------------------------------------------------------------------
# Plotting: 2x2 heatmap grid per experiment
# ---------------------------------------------------------------------------


def _build_heatmap_matrix_for_statement(
    all_roles: List[str],
    role_scores: Dict[str, np.ndarray],
    rounds: List[int],
) -> np.ndarray:
    """
    Build a (n_roles, n_rounds) matrix of mean scores for a given statement,
    aligning rows to all_roles. Roles missing for this statement are NaN.
    """
    n_roles = len(all_roles)
    n_rounds = len(rounds)
    data = np.full((n_roles, n_rounds), np.nan, dtype=float)

    for i, role in enumerate(all_roles):
        if role not in role_scores:
            continue
        scores = role_scores[role]  # shape: (num_runs, num_rounds)
        if scores.size == 0:
            continue
        # Average over runs
        data[i, :] = scores.mean(axis=0)

    return data


def plot_experiment_role_belief_grid(
    experiment_label: str,
    experiment_slug: str,
    experiment_data: Dict[str, Dict[str, Any]],
    metric: str,
    output_dir: Path,
) -> None:
    """
    Create a single 2x2 figure for a given experiment type, aggregated across statements.

    Layout:
        Columns: True Statement | False Statement
        Rows:    Doctor Correct | Doctor Incorrect

    experiment_data maps:
        statement_pretty -> {
            "role_scores": Dict[str, np.ndarray],
            "rounds": List[int],
            "correct_label": Optional[int],
            "config_name": str,
        }
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # The four statement labels we care about
    stmt_true_doc_correct = "True Statement (Doctor is Correct)"
    stmt_true_doc_incorrect = "True Statement (Doctor is Incorrect)"
    stmt_false_doc_correct = "False Statement (Doctor is Correct)"
    stmt_false_doc_incorrect = "False Statement (Doctor is Incorrect)"

    statements_order = [
        stmt_true_doc_correct,
        stmt_false_doc_correct,
        stmt_true_doc_incorrect,
        stmt_false_doc_incorrect,
    ]

    # Figure out the union of roles across all four statements for this experiment
    all_roles_set = set()
    for _stmt_label, data in experiment_data.items():
        role_scores = data.get("role_scores", {})
        all_roles_set.update(role_scores.keys())

    if not all_roles_set:
        print(f"  No roles found for experiment '{experiment_label}', skipping figure.")
        return

    all_roles = sorted(all_roles_set)

    # We keep color scale fixed in [0, 1] because scores are between 0 and 1
    # Use cmcrameri's bam colormap for diverging data (colorblind-safe)
    if CMAP_AVAILABLE:
        low, high = 0.1, 0.9

        new_colors = cmc.bam(np.linspace(low, high, 256))
        cmap = LinearSegmentedColormap.from_list("bam_trimmed", new_colors)
    else:
        cmap = "PiYG"
    metric_label = (
        "Collective Accuracy"
        if metric == "collective_accuracy"
        else "Avg. Belief Score"
    )

    fig = plt.figure(figsize=(7.2, 6.5))
    gs = gridspec.GridSpec(
        5,
        2,
        height_ratios=[0.01, 0.01, 0.46, 0.46, 0.06],
        width_ratios=[0.50, 0.50],
        hspace=0.5,
        wspace=0.2,
        figure=fig,
    )

    # Title row
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        -0.25,
        0.0,
        f"Belief Scores Across Rounds: {experiment_label}",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        fontfamily="sans-serif",
        color="#444444",
    )

    # Column titles
    ax_true_title = fig.add_subplot(gs[1, 0])
    ax_true_title.axis("off")
    ax_true_title.text(
        0.5,
        0.0,
        "True Statement",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#555555",
    )

    ax_false_title = fig.add_subplot(gs[1, 1])
    ax_false_title.axis("off")
    ax_false_title.text(
        0.5,
        0.0,
        "False Statement",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#555555",
    )

    # Panel letters in the 2x2 (row-major)
    panel_letters = ["A", "B", "C", "D"]

    # Map idx -> which gs cell to use
    # idx: 0,1 → row 2 (Doctor Correct), 2,3 → row 3 (Doctor Incorrect)
    idx_to_gs = {
        0: (2, 0),  # True, Doc Correct
        1: (2, 1),  # False, Doc Correct
        2: (3, 0),  # True, Doc Incorrect
        3: (3, 1),  # False, Doc Incorrect
    }

    # Plot each of the four panels
    for idx, stmt_label in enumerate(statements_order):
        data = experiment_data.get(stmt_label, None)
        row_gs, col_gs = idx_to_gs[idx]

        ax_plot = fig.add_subplot(gs[row_gs, col_gs])

        # Panel letter
        ax_plot.text(
            -0.15,
            1.05,
            panel_letters[idx],
            transform=ax_plot.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#555555",
        )

        # Left-side row labels
        if idx in (0, 1):
            # First row (Doctor Correct)
            if col_gs == 0:
                ax_plot.text(
                    -0.55,
                    0.5,
                    "Doctor is Correct",
                    transform=ax_plot.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="#555555",
                    rotation=90,
                )
        else:
            # Second row (Doctor Incorrect)
            if col_gs == 0:
                ax_plot.text(
                    -0.55,
                    0.5,
                    "Doctor is Incorrect",
                    transform=ax_plot.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="#555555",
                    rotation=90,
                )

        # Handle missing data gracefully
        if data is None or not data.get("role_scores"):
            ax_plot.axis("off")
            ax_plot.text(
                0.5,
                0.5,
                f"No data for\n{stmt_label}",
                ha="center",
                va="center",
                fontsize=7,
                color="#777777",
            )
            continue

        role_scores = data["role_scores"]
        rounds = data["rounds"]

        if not rounds:
            ax_plot.axis("off")
            ax_plot.text(
                0.5,
                0.5,
                f"No rounds for\n{stmt_label}",
                ha="center",
                va="center",
                fontsize=7,
                color="#777777",
            )
            continue

        # Build heatmap matrix aligned to all_roles
        heatmap_data = _build_heatmap_matrix_for_statement(
            all_roles=all_roles,
            role_scores=role_scores,
            rounds=rounds,
        )

        # Draw the heatmap without a per-panel colorbar
        sns.heatmap(
            heatmap_data,
            ax=ax_plot,
            cmap=cmap,
            xticklabels=[str(r) for r in rounds],
            yticklabels=all_roles,
            vmin=0.0,
            vmax=1.0,
            cbar=False,
        )

        # Heatmap axis configuration:
        # - X-axis: no tick marks, labels only on bottom row
        # - Y-axis: tick marks outward, labels only on left column

        # Bottom row = idx 2,3
        is_bottom_row = idx in (2, 3)
        # Left column = idx 0,2
        is_left_col = idx in (0, 2)

        ax_plot.tick_params(axis="x", length=0)
        ax_plot.tick_params(axis="y", direction="out")

        if is_bottom_row:
            ax_plot.set_xlabel("Round", fontsize=8)
            ax_plot.tick_params(axis="x", labelsize=7)
        else:
            ax_plot.set_xlabel("")
            ax_plot.set_xticklabels([])

        if is_left_col:
            ax_plot.tick_params(axis="y", labelsize=7)
        else:
            ax_plot.set_yticklabels([])

    # Shared colorbar at bottom
    ax_colorbar = fig.add_subplot(gs[4, :])

    pos = ax_colorbar.get_position()
    new_height = pos.height * 0.6  # tweak this factor (0.3, 0.5, etc.) to taste
    ax_colorbar.set_position(
        [
            pos.x0,
            pos.y0 + (pos.height - new_height) / 2,  # re-center vertically
            pos.width,
            new_height,
        ]
    )

    norm = Normalize(vmin=0.0, vmax=1.0)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        cax=ax_colorbar,
        orientation="horizontal",
    )

    # Put label below the bar, left-to-right
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.set_xlabel(metric_label, fontsize=9, labelpad=6)

    cb_ax = cbar.ax
    trans = cb_ax.get_xaxis_transform()
    y_arrow = -2.2
    y_text = -2.7

    cb_ax.annotate(
        "",
        xy=(0.00, y_arrow),
        xytext=(0.15, y_arrow),
        xycoords=trans,
        textcoords=trans,
        arrowprops=dict(
            arrowstyle="->",
            mutation_scale=8,
            linewidth=0.8,
        ),
        clip_on=False,
    )
    cb_ax.text(
        0.08,
        y_text,
        "Believes False",
        transform=trans,
        ha="center",
        va="top",
        fontsize=8,
        clip_on=False,
    )

    cb_ax.annotate(
        "",
        xy=(1.00, y_arrow),
        xytext=(0.85, y_arrow),
        xycoords=trans,
        textcoords=trans,
        arrowprops=dict(
            arrowstyle="->",
            mutation_scale=8,
            linewidth=0.8,
        ),
        clip_on=False,
    )
    cb_ax.text(
        0.92,
        y_text,
        "Believes True",
        transform=trans,
        ha="center",
        va="top",
        fontsize=8,
        clip_on=False,
    )

    out_path = output_dir / f"{experiment_slug}_role_belief_heatmap_grid.pdf"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(
        f"Saved role belief heatmap grid for experiment '{experiment_label}' to {out_path}"
    )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def plot_experiment_heatmap_grids(
    runs_dir: str = "outputs/runs",
    output_dir: Path = Path("outputs") / "plots",
    min_runs: int = 1,
    metric: str = "belief_score",
    experiment_filter: Optional[str] = None,
) -> None:
    """
    Discover multi-run configs, group them by experiment and statement, and
    generate one 2x2 heatmap grid per experiment.
    """
    print("Discovering multi-run experiments...")
    multi_run_configs = find_multi_run_experiments(runs_dir)

    # Filter by min-runs
    multi_run_configs = {
        k: v for k, v in multi_run_configs.items() if len(v) >= min_runs
    }

    if not multi_run_configs:
        print(f"No multi-run experiments found with at least {min_runs} runs.")
        return

    # experiments[(probe, experiment_pretty)][statement_pretty] -> dict with data
    experiments: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)

    # Only include these experiment types by default
    allowed_experiments = {"experts", "only_llms", "random_roles", "random_experts"}

    for config_name, run_dirs in multi_run_configs.items():
        print(f"\n{'=' * 60}")
        print(f"Loading: {config_name}")
        print(f"{'=' * 60}")

        parts = config_name.split("/")
        if len(parts) != 4:
            print(f"  Unexpected config format: {config_name}, skipping.")
            continue

        probe, experiment_name, _prompt, _statement_raw = parts

        if experiment_name not in allowed_experiments:
            print(f"  Skipping experiment '{experiment_name}' (not in allowed set).")
            continue

        if experiment_filter is not None and experiment_filter not in experiment_name:
            print(
                f"  Skipping '{experiment_name}' (does not match filter '{experiment_filter}')."
            )
            continue

        # Load role belief data once per config (statement)
        role_scores, rounds, correct_label = load_role_belief_data(
            run_dirs, metric=metric
        )
        if not role_scores:
            print("  No valid role score data found, skipping.")
            continue

        # Pretty labels
        experiment_pretty, statement_pretty = parse_experiment_and_statement(
            config_name
        )
        key = (probe, experiment_pretty)

        experiments[key][statement_pretty] = {
            "role_scores": role_scores,
            "rounds": rounds,
            "correct_label": correct_label,
            "config_name": config_name,
        }

    if not experiments:
        print("No experiment-level data available after loading; exiting.")
        return

    # Create one figure per (probe, experiment)
    for (probe, experiment_label), exp_data in experiments.items():
        print(
            f"\nCreating role belief heatmap grid for probe={probe}, "
            f"experiment='{experiment_label}'"
        )
        experiment_slug = slugify_experiment(experiment_label)
        out_dir_exp = output_dir / probe
        plot_experiment_role_belief_grid(
            experiment_label=experiment_label,
            experiment_slug=experiment_slug,
            experiment_data=exp_data,
            metric=metric,
            output_dir=out_dir_exp,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate role-based belief strength heatmap grids grouped by experiment "
            "(Experts, Generalists, etc.), with a 2x2 layout over statement types."
        )
    )
    parser.add_argument(
        "--runs-dir",
        default="outputs/runs",
        help="Path to runs directory (default: outputs/runs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "plots",
        help="Base directory to save plots (default: outputs/plots)",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=1,
        help="Minimum number of runs required for a config (default: 1)",
    )
    parser.add_argument(
        "--metric",
        choices=["belief_score", "collective_accuracy"],
        default="belief_score",
        help="Metric to use: 'belief_score' or 'collective_accuracy' (default: belief_score)",
    )
    parser.add_argument(
        "--experiment-filter",
        type=str,
        default=None,
        help="Optional substring filter for experiment name (e.g., 'experts', 'only_llms')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plot_experiment_heatmap_grids(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        min_runs=args.min_runs,
        metric=args.metric,
        experiment_filter=args.experiment_filter,
    )


if __name__ == "__main__":
    main()
