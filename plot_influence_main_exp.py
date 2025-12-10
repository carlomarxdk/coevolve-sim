"""Influence analysis plotting for main manuscript figures.

This module generates publication-ready figures showing agent influence on
others' beliefs across multi-run experiments. Figures display influence vs.
plasticity scatter plots and transfer entropy by agent role.
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
import pandas as pd
from matplotlib.font_manager import FontManager
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from src.metrics.general import information_leadership

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
            "primary": mpl.colors.to_hex(batlowS(3)),  # First color (other agents)
            "secondary": mpl.colors.to_hex(batlowS(4)),  # Fourth color (HM agent)
            "tertiary": mpl.colors.to_hex(batlowS(0)),  # Third color
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


def compute_te_stats(df_info: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute mean and SEM of information_leadership per role,
    returning a DataFrame with index=role and columns [mean, count, std, sem],
    sorted alphabetically by role.

    Returns None if there is no usable TE data.
    """
    if "information_leadership" not in df_info.columns:
        return None

    df = df_info.dropna(subset=["information_leadership"]).copy()
    if df.empty:
        return None

    grouped = df.groupby("role")["information_leadership"]
    stats = grouped.agg(
        mean="mean",
        count="size",
        std=lambda x: x.std(ddof=1),
    )

    stats["sem"] = stats.apply(
        lambda row: (
            0.0
            if row["count"] <= 1 or np.isnan(row["std"])
            else float(row["std"] / np.sqrt(row["count"]))
        ),
        axis=1,
    )

    # Sort alphabetically by role
    stats = stats.sort_index()

    return stats


def find_multi_run_experiments(runs_dir: str = "outputs/runs") -> Dict[str, List[Path]]:
    """
    Discover multi-run experiments organized by configuration.

    Experiments are grouped by: probe/catalog/prompt/statement

    Args:
        runs_dir: Path to the runs directory

    Returns:
        Dictionary mapping configuration keys to list of run directories:
            { "probe/catalog/prompt/statement": [run_dir1, run_dir2, ...], ... }
    """
    runs_path = Path(runs_dir)

    if not runs_path.exists():
        print(f"Warning: Runs directory '{runs_dir}' does not exist")
        return {}

    multi_run_configs: Dict[str, List[Path]] = defaultdict(list)

    # Navigate directory structure: probe/catalog/prompt/statement/timestamp/
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
                        key = f"{probe_dir.name}/{catalog_dir.name}/{prompt_dir.name}/{statement_dir.name}"
                        multi_run_configs[key] = run_dirs

    return dict(multi_run_configs)


def adjacency_matrix(edges, n_nodes: int) -> np.ndarray:
    """Build undirected adjacency matrix from edge list."""
    A = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A


def analyze_influence_scores_across_runs(run_dirs: List[Path]) -> pd.DataFrame:
    """
    Analyze influence scores across multiple runs, aggregated per agent.

    Uses:
        - results/agent_manifest.json
        - results/final_metrics.json (te_matrix, influence scores)
        - results/network_edges.json (for adjacency & masking)

    Returns:
        DataFrame with columns (when available):
        - run_id, run_name, role, agent_id
        - influence_granger, influence_simple, information_leadership
    """
    all_records: List[Dict[str, Any]] = []

    for run_idx, run_dir in enumerate(run_dirs):
        agent_manifest_path = run_dir / "results" / "agent_manifest.json"
        final_metrics_path = run_dir / "results" / "final_metrics.json"
        adjacency_path = run_dir / "results" / "network_edges.json"

        if (
            not agent_manifest_path.exists()
            or not final_metrics_path.exists()
            or not adjacency_path.exists()
        ):
            continue

        with agent_manifest_path.open("r") as f:
            agent_manifest = json.load(f)

        with final_metrics_path.open("r") as f:
            final_metrics = json.load(f)

        with adjacency_path.open("r") as f:
            data = json.load(f)
            n_agents = data["n"]
            edges = data["edges"]
            A = adjacency_matrix(edges, n_agents)

        te_matrix = np.asarray(final_metrics.get("te_matrix", []))
        if te_matrix.size == 0:
            info_leadership = None
        else:
            te_matrix = te_matrix * A  # mask by adjacency
            info_leadership = np.asarray(information_leadership(te_matrix))

        influence_granger = final_metrics.get("influence_scores_granger", [])
        influence_simple = final_metrics.get("influence_scores_simple", [])

        # If literally no metrics, skip run
        if not influence_granger and not influence_simple and info_leadership is None:
            continue

        for agent_id_str, agent_info in agent_manifest.items():
            agent_id = int(agent_id_str)
            role = agent_info.get("role", "unknown")

            record: Dict[str, Any] = {
                "run_id": run_idx,
                "run_name": run_dir.name,
                "role": role,
                "agent_id": agent_id,
            }

            if influence_granger and agent_id < len(influence_granger):
                record["influence_granger"] = influence_granger[agent_id]
            if influence_simple and agent_id < len(influence_simple):
                record["influence_simple"] = influence_simple[agent_id]
            if info_leadership is not None and agent_id < len(info_leadership):
                record["information_leadership"] = float(info_leadership[agent_id])

            all_records.append(record)

    return pd.DataFrame.from_records(all_records)


def load_summary_for_config(
    summary_root: Path,
    config_name: str,
) -> pd.DataFrame:
    """
    Load opinion_leaders_summary.csv for a given config.

    We expect summary_root to point to 'outputs/plots' (and we will append
    probe/experiment/prompt/statement to it), so that the full path is:

        summary_root / probe / experiment / prompt / statement / opinion_leaders_summary.csv
    """
    parts = config_name.split("/")
    if len(parts) != 4:
        raise ValueError(
            f"Config name should have 4 parts: probe/experiment/prompt/statement, got: {config_name}"
        )

    probe, experiment, prompt, statement = parts

    summary_path = (
        summary_root
        / probe
        / experiment
        / prompt
        / statement
        / "opinion_leaders_summary.csv"
    )
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found at: {summary_path}")

    return pd.read_csv(summary_path)


def parse_experiment_and_statement(config_name: str) -> Tuple[str, str]:
    """
    Map a config_name string to human-readable experiment and statement labels.
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


def slugify_experiment(experiment: str) -> str:
    """
    Turn a human-readable experiment label into a filesystem-safe slug.
    E.g., "Base LLMs (Randomized)" -> "base_llms_randomized".
    """
    s = experiment.lower()
    for ch in ["(", ")", ".", ","]:
        s = s.replace(ch, "")
    s = s.replace(" ", "_").replace("/", "_")
    return s


def plot_single_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    doctor_role: str,
) -> List[Line2D]:
    """Influence vs Plasticity scatter plot.

    Highlights different agent roles with distinct markers and colors from
    the batlowS colormap:
      - Clinical Physician (star, accent color)
      - Biomedical Researcher (pentagon, accent color)
      - Chemist (square, accent color)
      - Human Participant (diamond, secondary color)
      - LLM (triangle, tertiary color)
      - Other agents (circles, primary color)

    Args:
        ax: Matplotlib axes object to plot on.
        df: DataFrame with influence_mean and plasticity_mean columns.
        doctor_role: Role name to treat as the primary doctor role.

    Returns:
        List of Line2D handles for building a shared legend.
    """
    handles: List[Line2D] = []

    if df.empty or "influence_mean" not in df or "plasticity_mean" not in df:
        ax.text(
            0.5,
            0.5,
            "No influence/plasticity data",
            ha="center",
            va="center",
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return handles

    x = df["influence_mean"].to_numpy()
    y = df["plasticity_mean"].to_numpy()

    # Color palette using batlowS colormap
    color_other = BATLOWS_COLORS["primary"]
    color_doctor = BATLOWS_COLORS["accent"]
    color_human = BATLOWS_COLORS["secondary"]
    color_llm = BATLOWS_COLORS["tertiary"]

    # Define special roles (excluded from "Other Agents")
    special_roles = {
        doctor_role,
        "Biomedical Researcher",
        "Chemist",
        "Human Participant",
        "LLM",
    }

    # "Other" agents = everyone not in the special set
    mask_other = ~df["role"].isin(special_roles)
    if mask_other.any():
        ax.scatter(
            x[mask_other],
            y[mask_other],
            s=25,
            alpha=0.6,
            color=color_other,
            label="Other Agents",
        )

    # Clinical Physician
    mask_doc = df["role"] == doctor_role
    if mask_doc.any():
        doc = df.loc[mask_doc]
        ax.scatter(
            doc["influence_mean"],
            doc["plasticity_mean"],
            marker="*",
            s=70,
            color=color_doctor,
            edgecolor="black",
            linewidth=0.8,
            label="Clinical Physician",
        )

    # Biomedical Researcher
    mask_bio = df["role"] == "Biomedical Researcher"
    if mask_bio.any():
        bio = df.loc[mask_bio]
        ax.scatter(
            bio["influence_mean"],
            bio["plasticity_mean"],
            marker="p",
            s=40,
            color=color_doctor,
            edgecolor="black",
            linewidth=0.8,
            label="Biomedical Researcher",
        )

    # Chemist
    mask_chem = df["role"] == "Chemist"
    if mask_chem.any():
        chem = df.loc[mask_chem]
        ax.scatter(
            chem["influence_mean"],
            chem["plasticity_mean"],
            marker="s",
            s=35,
            color=color_doctor,
            edgecolor="black",
            linewidth=0.8,
            label="Chemist",
        )

    # Human Participant
    mask_human = df["role"] == "Human Participant"
    if mask_human.any():
        human = df.loc[mask_human]
        ax.scatter(
            human["influence_mean"],
            human["plasticity_mean"],
            marker="D",  # diamond
            s=40,
            color=color_human,
            edgecolor="black",
            linewidth=0.8,
            label="Human Participant",
        )

    # LLM
    mask_llm = df["role"] == "LLM"
    if mask_llm.any():
        llm = df.loc[mask_llm]
        ax.scatter(
            llm["influence_mean"],
            llm["plasticity_mean"],
            marker="^",  # triangle up
            s=45,
            color=color_llm,
            edgecolor="black",
            linewidth=0.8,
            label="LLM",
        )

    # Correlation box (upper-left)
    if len(x) > 1 and x.std() > 1e-10 and y.std() > 1e-10:
        corr = spearmanr(x, y).correlation
        ax.text(
            0.05,
            0.95,
            f"$\\rho = {corr:.3f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=7,
            color="#333333",
        )

    ax.set_xlabel("Avg. Influence", fontsize=8)
    ax.set_ylabel("Avg. Plasticity", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.3)

    # Build handles for shared legend (matching the above colors/markers)
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8,
            color=color_other,
            label="Other Agents",
        )
    )
    if mask_human.any():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="None",
                markersize=6,
                markerfacecolor=color_human,
                markeredgecolor="black",
                label="Human Participant",
            )
        )
    if mask_llm.any():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="None",
                markersize=7,
                markerfacecolor=color_llm,
                markeredgecolor="black",
                label="LLM",
            )
        )
    if mask_bio.any():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="p",
                linestyle="None",
                markersize=8,
                markerfacecolor=color_doctor,
                markeredgecolor="black",
                label="Biomedical Researcher",
            )
        )
    if mask_doc.any():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="None",
                markersize=10,
                markerfacecolor=color_doctor,
                markeredgecolor="black",
                label="Clinical Physician",
            )
        )
    if mask_chem.any():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                markersize=7,
                markerfacecolor=color_doctor,
                markeredgecolor="black",
                label="Chemist",
            )
        )

    return handles


def plot_info_leadership_bar(
    ax: plt.Axes,
    df_info: pd.DataFrame,
    *,
    te_ylim: Optional[Tuple[float, float]] = None,
    show_xticklabels: bool = True,
) -> None:
    """Bar chart of mean information_leadership by role across runs.

    Displays mean transfer entropy values with standard error bars, using
    colors from the batlowS colormap:
      - Doctor roles (Clinical Physician, Biomedical Researcher, Chemist): accent color
      - Human Participant: secondary color
      - LLM: tertiary color
      - All others: primary color

    Roles are ordered alphabetically on the x-axis.

    Args:
        ax: Matplotlib axes object to plot on.
        df_info: DataFrame with role and information_leadership columns.
        te_ylim: Optional y-axis limits for consistent scaling across panels.
        show_xticklabels: If False, x tick labels are hidden (for non-bottom rows).
    """
    stats = compute_te_stats(df_info)
    if stats is None:
        ax.text(
            0.5,
            0.5,
            "No information_leadership data",
            ha="center",
            va="center",
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    roles = stats.index.tolist()
    means = stats["mean"].to_numpy(dtype=float)
    sems = stats["sem"].to_numpy(dtype=float)

    positions = np.arange(len(roles))

    # Base color mapping using batlowS colormap
    default_color = BATLOWS_COLORS["primary"]
    role_colors: Dict[str, str] = {r: default_color for r in roles}

    # Doctors (Clinical Physician, Biomedical Researcher, Chemist)
    doctor_roles = {
        "Clinical Physician",
        "Biomedical Researcher",
        "Chemist",
    }
    for r in doctor_roles:
        if r in role_colors:
            role_colors[r] = BATLOWS_COLORS["accent"]

    # Human Participant
    if "Human Participant" in role_colors:
        role_colors["Human Participant"] = BATLOWS_COLORS["secondary"]

    # LLM
    if "LLM" in role_colors:
        role_colors["LLM"] = BATLOWS_COLORS["tertiary"]

    colors = [role_colors[r] for r in roles]

    ax.bar(
        positions,
        means,
        yerr=sems,
        color=colors,
        edgecolor="black",
        alpha=0.7,
        capsize=3,
    )

    ax.set_xticks(positions)
    if show_xticklabels:
        ax.set_xticklabels(roles, rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xticklabels([])

    ax.set_ylabel("Transfer Entropy", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Apply shared TE y-limits if provided
    if te_ylim is not None:
        ax.set_ylim(*te_ylim)


def plot_experiment_influence_grid(
    experiment_label: str,
    experiment_slug: str,
    experiment_data: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: Path,
    doctor_role: str,
    annotate_zero_te: bool = False,
) -> None:
    """
    Create a single figure for a given experiment type, aggregated across statements.

    Layout:
        - Title: "Agent Influence on Others' Beliefs: {experiment_label}"
        - 4 rows (statements):
            1) True (Doc. Correct)
            2) True (Doc. Incorrect)
            3) False (Doc. Correct)
            4) False (Doc. Incorrect)
        - 2 columns:
            (1) Avg. Plasticity vs. Influence (scatter)
            (2) Transfer Entropy by role (bar)

    For a given experiment, all scatter plots share the same x/y limits,
    and all TE bar charts share the same y-limit. Limits are computed
    ONLY from the statements actually plotted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statement order and row labels
    statements_order = [
        "True Statement (Doctor is Correct)",
        "True Statement (Doctor is Incorrect)",
        "False Statement (Doctor is Correct)",
        "False Statement (Doctor is Incorrect)",
    ]

    row_labels = {
        "True Statement (Doctor is Correct)": "True Statement\n(Doctor is Correct)",
        "True Statement (Doctor is Incorrect)": "True Statement\n(Doctor is Incorrect)",
        "False Statement (Doctor is Correct)": "False Statement\n(Doctor Correct)",
        "False Statement (Doctor is Incorrect)": "False Statement\n(Doctor is Incorrect)",
    }

    # ---------- FIRST PASS: compute shared axis limits for this experiment ----------
    global_x_min = np.inf
    global_x_max = -np.inf
    global_y_min = np.inf
    global_y_max = -np.inf
    global_te_max = 0.0

    for stmt_label in statements_order:
        data = experiment_data.get(stmt_label, None)
        if data is None:
            continue

        # Scatter limits
        df_summary = data.get("summary_df", pd.DataFrame())
        if (
            not df_summary.empty
            and "influence_mean" in df_summary
            and "plasticity_mean" in df_summary
        ):
            x = df_summary["influence_mean"].to_numpy()
            y = df_summary["plasticity_mean"].to_numpy()
            if x.size > 0 and y.size > 0:
                global_x_min = min(global_x_min, float(x.min()))
                global_x_max = max(global_x_max, float(x.max()))
                global_y_min = min(global_y_min, float(y.min()))
                global_y_max = max(global_y_max, float(y.max()))

        # TE limits based on bar means ± SEM
        df_info = data.get("info_df", pd.DataFrame())
        te_stats = compute_te_stats(df_info)
        if te_stats is not None and not te_stats.empty:
            local_max = float((te_stats["mean"] + te_stats["sem"]).max())
            global_te_max = max(global_te_max, local_max)

    # Build scatter limits if we saw any valid data
    if np.isfinite(global_x_min) and np.isfinite(global_x_max):
        x_range = global_x_max - global_x_min
        x_pad = 0.05 * (x_range if x_range > 0 else 1.0)
        scatter_xlim = (global_x_min - x_pad, global_x_max + x_pad)
    else:
        scatter_xlim = None

    if np.isfinite(global_y_min) and np.isfinite(global_y_max):
        y_range = global_y_max - global_y_min
        y_pad = 0.05 * (y_range if y_range > 0 else 1.0)
        scatter_ylim = (global_y_min - y_pad, global_y_max + y_pad)
    else:
        scatter_ylim = None

    # TE y-limit
    if global_te_max > 0.0:
        te_ylim = (0.0, global_te_max * 1.1)
    else:
        te_ylim = None

    # ---------- FIGURE LAYOUT ----------
    fig = plt.figure(figsize=(7.2, 7.2))
    gs = gridspec.GridSpec(
        6,
        2,
        height_ratios=[0.03, 0.01, 0.24, 0.24, 0.24, 0.24],
        width_ratios=[0.40, 0.60],
        hspace=0.5,
        wspace=0.5,
        figure=fig,
    )

    # Title row
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        -0.1,
        0.0,
        f"Agent Influence on Others' Beliefs: {experiment_label}",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#333333",
    )

    # Column titles
    ax_scatter_title = fig.add_subplot(gs[1, 0])
    ax_scatter_title.axis("off")
    ax_scatter_title.text(
        0.5,
        0.0,
        "Avg. Plasticity vs. Influence",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#444444",
    )

    ax_transfer_title = fig.add_subplot(gs[1, 1])
    ax_transfer_title.axis("off")
    ax_transfer_title.text(
        0.5,
        0.0,
        "Transfer Entropy by Agent",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#444444",
    )

    # Panel letters (row-major for 4 rows x 2 cols)
    letters = {
        (statements_order[0], 0): "A",
        (statements_order[0], 1): "B",
        (statements_order[1], 0): "C",
        (statements_order[1], 1): "D",
        (statements_order[2], 0): "E",
        (statements_order[2], 1): "F",
        (statements_order[3], 0): "G",
        (statements_order[3], 1): "H",
    }

    shared_legend_handles: List[Line2D] = []
    last_scatter_ax = None

    bottom_row_index = 1 + len(statements_order)  # 1 + 4 = 5

    # ---------- PLOTTING ROWS ----------
    for idx_row, stmt_label in enumerate(statements_order, start=2):
        data = experiment_data.get(stmt_label, None)
        is_bottom_row = idx_row == bottom_row_index

        # Left: scatter
        ax_scatter = fig.add_subplot(gs[idx_row, 0])
        last_scatter_ax = ax_scatter
        letter_scatter = letters[(stmt_label, 0)]
        ax_scatter.text(
            -0.15,
            1.05,
            f"{letter_scatter}",
            transform=ax_scatter.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#444444",
        )

        # Row (statement) label on left side
        row_text = row_labels[stmt_label]
        ax_scatter.text(
            -0.38,
            0.5,
            row_text,
            transform=ax_scatter.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="#444444",
            rotation=90,
        )

        # Right: TE bar by role
        ax_bar = fig.add_subplot(gs[idx_row, 1])
        letter_bar = letters[(stmt_label, 1)]
        ax_bar.text(
            -0.10,
            1.05,
            f"{letter_bar}",
            transform=ax_bar.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#444444",
        )

        if data is None:
            # No data for this statement/experiment
            for ax in (ax_scatter, ax_bar):
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"No data for\n{stmt_label}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#777777",
                )
            continue

        df_summary = data.get("summary_df", pd.DataFrame())
        df_info = data.get("info_df", pd.DataFrame())

        # ---------- Scatter ----------
        handles = plot_single_scatter(ax_scatter, df_summary, doctor_role=doctor_role)

        if scatter_xlim is not None:
            ax_scatter.set_xlim(*scatter_xlim)
        if scatter_ylim is not None:
            ax_scatter.set_ylim(*scatter_ylim)

        # Hide x tick labels for non-bottom rows
        if not is_bottom_row:
            ax_scatter.set_xlabel("")  # remove label text
            ax_scatter.set_xticklabels([])

        if shared_legend_handles == [] and handles:
            shared_legend_handles = handles

        # ---------- TE bar ----------
        plot_info_leadership_bar(
            ax_bar,
            df_info,
            te_ylim=te_ylim,
            show_xticklabels=is_bottom_row,
        )

        if annotate_zero_te and stmt_label == "True Statement (Doctor is Incorrect)":
            ax_bar.text(
                0.5,
                0.5,
                (
                    "Beliefs do not change over time,\n"
                    "so Transfer Entropy is zero for\n"
                    "all agents."
                ),
                transform=ax_bar.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.85,
                ),
            )

    # ---------- Shared legend under the whole figure ----------
    if shared_legend_handles and last_scatter_ax is not None:
        labels = [h.get_label() for h in shared_legend_handles]
        last_scatter_ax.legend(
            shared_legend_handles,
            labels,
            loc="center",
            bbox_to_anchor=(0.5, -0.70),
            ncol=2,
            fontsize=7,
            frameon=False,
        )

    out_path = output_dir / f"{experiment_slug}_influence_by_statement_grid.pdf"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved influence grid for experiment '{experiment_label}' to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Avg. Plasticity vs. Influence and Transfer Entropy by agent, "
            "aggregated by experiment type across statements."
        )
    )
    parser.add_argument(
        "--runs-dir",
        default="outputs/runs",
        help="Path to runs directory (default: outputs/runs)",
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("outputs") / "plots",
        help=(
            "Root directory for summary CSVs (we append probe/experiment/prompt/statement). "
            "Default: outputs/plots"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "plots",
        help="Base directory to save plots. Default: outputs/plots",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=1,
        help="Minimum number of runs required for a config (default: 1)",
    )
    parser.add_argument(
        "--doctor-role",
        type=str,
        default="Clinical Physician",
        help="Role name to treat as the doctor (default: 'Clinical Physician').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Discover multi-run configs
    print("Discovering multi-run experiments...")
    multi_run_configs = find_multi_run_experiments(args.runs_dir)

    # Filter by min-runs
    multi_run_configs = {
        k: v for k, v in multi_run_configs.items() if len(v) >= args.min_runs
    }

    if not multi_run_configs:
        print(f"No multi-run experiments found with at least {args.min_runs} runs.")
        return

    # experiments[(probe, experiment_pretty)][statement_pretty] -> dict with dfs
    experiments: Dict[Tuple[str, str], Dict[str, Dict[str, pd.DataFrame]]] = (
        defaultdict(dict)
    )

    # Only include these experiment types
    allowed_experiments = {"experts", "only_llms", "random_roles", "random_experts"}

    for config_name, run_dirs in multi_run_configs.items():
        print(f"\n{'=' * 60}")
        print(f"Loading: {config_name}")
        print(f"{'=' * 60}")

        # path: probe/experiment/prompt/statement
        parts = config_name.split("/")
        if len(parts) != 4:
            print(f"Unexpected config format: {config_name}, skipping.")
            continue

        probe, experiment_name, _prompt, _statement_raw = parts

        if experiment_name not in allowed_experiments:
            print(f"Skipping experiment '{experiment_name}' (not in allowed set).")
            continue

        # Load summary CSV for scatter
        try:
            summary_df = load_summary_for_config(args.summary_root, config_name)
        except FileNotFoundError as e:
            print(f"ERROR loading summary CSV: {e}", file=sys.stderr)
            continue

        # Compute information_leadership per agent over runs
        info_df = analyze_influence_scores_across_runs(run_dirs)
        if info_df.empty:
            print(
                "Warning: no influence/leadership data found across runs; TE plots may be empty."
            )

        # Pretty labels
        experiment_pretty, statement_pretty = parse_experiment_and_statement(
            config_name
        )
        key = (probe, experiment_pretty)

        experiments[key][statement_pretty] = {
            "summary_df": summary_df,
            "info_df": info_df,
        }

    if not experiments:
        print("No experiment-level data available after loading; exiting.")
        return

    # Create one figure per (probe, experiment)
    for (probe, experiment_label), exp_data in experiments.items():
        print(
            f"\nCreating influence grid for probe={probe}, experiment='{experiment_label}'"
        )
        annotate_zero_te = experiment_label == "Generalists (Randomized Roles)"

        experiment_slug = slugify_experiment(experiment_label)
        out_dir_exp = args.output_dir / probe
        plot_experiment_influence_grid(
            experiment_label=experiment_label,
            experiment_slug=experiment_slug,
            experiment_data=exp_data,
            output_dir=out_dir_exp,
            doctor_role=args.doctor_role,
            annotate_zero_te=annotate_zero_te,
        )


if __name__ == "__main__":
    main()
