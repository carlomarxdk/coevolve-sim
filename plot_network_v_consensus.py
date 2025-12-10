import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.metrics.general import collective_accuracy_scores


def find_multi_run_experiments(runs_dir: str = "outputs/runs") -> Dict[str, List[Path]]:
    """
    Discover multi-run experiments organized by configuration.

    Experiments are grouped by: probe/catalog/prompt/statement

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

    multi_run_configs = defaultdict(list)

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
                        if d.is_dir() and (d / "config.json").exists()
                        # skip slurm "_xx" re-run dirs
                        and not (
                            len(d.name) >= 3
                            and d.name[-3] == "_"
                            and len(d.name[-2:]) == 2
                        )
                    ]

                    if len(run_dirs) > 1:  # Multi-run experiment
                        config_key = f"{probe_dir.name}/{catalog_dir.name}/{prompt_dir.name}/{statement_dir.name}"
                        multi_run_configs[config_key] = run_dirs

    return dict(multi_run_configs)


def analyze_network_structure_impact(run_dirs: List[Path]) -> pd.DataFrame:
    """
    Analyze how network structure properties affect belief diffusion.

    Args:
        run_dirs: List of run directories

    Returns:
        DataFrame with network properties and diffusion outcomes
    """
    data_records = []

    for run_dir in run_dirs:
        # Load config
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue

        with open(config_path, "r") as f:
            config = json.load(f)

        # Load network manifest
        network_path = run_dir / "results" / "network_manifest.json"
        if not network_path.exists():
            continue

        with open(network_path, "r") as f:
            network_manifest = json.load(f)

        # Load final metrics
        final_path = run_dir / "results" / "final_metrics.json"
        if not final_path.exists():
            continue

        # final_metrics not currently used
        # with open(final_path, "r") as f:
        #     final_metrics = json.load(f)

        # Load per-round metrics for convergence info
        metrics_path = run_dir / "results" / "per_round_metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path, "r") as f:
            per_round_metrics = json.load(f)

        # Extract network properties
        graph_info = network_manifest.get("graph", {})
        # Extract diffusion outcomes
        per_round = per_round_metrics.get("per_round", [])

        correct_label = int(config.get("statement", {}).get("label", {}).get("correct"))
        if per_round:
            final_round = per_round[-1]
            initial_round = per_round[0]

            final_labels = (
                final_round.get("belief", {}).get("labels", {}).get("values", [])
            )
            final_num_correct = ([val == correct_label for val in final_labels]).count(
                True
            )

            # Compute diffusion metrics
            initial_consensus = initial_round.get("consensus", {}).get(
                "consensus_fraction", 0
            )
            final_consensus = final_round.get("consensus", {}).get(
                "consensus_fraction", 0
            )
            consensus_change = np.abs(final_consensus - initial_consensus)

            initial_score = (
                initial_round.get("belief", {}).get("score", {}).get("mean", 0)
            )
            final_score = final_round.get("belief", {}).get("score", {}).get("mean", 0)
            score_change = final_score - initial_score

            init_scores = np.array(
                initial_round.get("belief", {}).get("score", {}).get("values", [])
            )
            final_scores = np.array(
                final_round.get("belief", {}).get("score", {}).get("values", [])
            )

            init_acc = collective_accuracy_scores(init_scores, correct_label).mean()
            final_acc = collective_accuracy_scores(final_scores, correct_label).mean()
            lift = final_acc - init_acc

            initial_entropy = initial_round.get("entropy", 0)
            final_entropy = final_round.get("entropy", 0)
            entropy_change = np.abs(final_entropy - initial_entropy)

            record = {
                "run_id": run_dir.name,
                "seed": config.get("seed"),
                # Network structure
                "num_nodes": graph_info.get("num_nodes"),
                "num_edges": graph_info.get("num_edges"),
                "density": graph_info.get("density"),
                "avg_clustering": graph_info.get("avg_clustering"),
                "avg_degree": graph_info.get("degrees", {}).get("mean"),
                "diameter": graph_info.get("diameter"),
                "avg_path_length": graph_info.get("avg_path_length"),
                # Diffusion outcomes
                "initial_consensus": initial_consensus,
                "final_consensus": final_consensus,
                "consensus_change": consensus_change,
                "initial_score": initial_score,
                "final_score": final_score,
                "score_change": score_change,
                "initial_entropy": initial_entropy,
                "final_entropy": final_entropy,
                "entropy_change": entropy_change,
                "rounds_completed": len(per_round),
                "final_magnetization": final_round.get("consensus", {}).get(
                    "magnetization", 0
                ),
                "final_num_correct": final_num_correct,
                "initial_accuracy": init_acc,
                "final_accuracy": final_acc,
                "accuracy_lift": lift,
            }

            data_records.append(record)

    return pd.DataFrame(data_records)


def plot_network_impact_analysis(
    df: pd.DataFrame,
    experiment_name: str,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Plot the impact of network structure on belief diffusion for a single experiment type,
    aggregating across multiple statement types with different colors.

    Args:
        df: DataFrame with network properties, diffusion outcomes, and 'statement' column
        experiment_name: internal experiment key (e.g., 'experts', 'only_llms')
        output_dir: Directory to save plots. If None, saves to outputs/plots
    """
    if df.empty:
        print(f"No data to plot for experiment '{experiment_name}'")
        return

    # Pretty experiment labels
    experiment_labels = {
        "experts": "Experts",
        "random_experts": "Experts (Randomized)",
        "only_llms": "Base LLMs",
        "random_roles": "Base LLMs (Randomized)",
    }
    experiment_label = experiment_labels.get(experiment_name, experiment_name)

    # Statement labels & colors
    statement_labels = {
        "true_doc_1": "True (Doc. Correct)",
        "true_rest_1": "True (Doc. Incorrect)",
        "false_doc_1": "False (Doc. Correct)",
        "false_rest_1": "False (Doc. Incorrect)",
    }
    statement_colors = {
        "true_doc_1": "tab:blue",
        "true_rest_1": "tab:green",
        "false_doc_1": "tab:red",
        "false_rest_1": "tab:orange",
    }

    # Canonical order for 2x2 r-grid
    statement_order = ["true_doc_1", "true_rest_1", "false_doc_1", "false_rest_1"]

    # Only keep statement types we know how to label
    df = df[df["statement"].isin(statement_labels.keys())].copy()
    if df.empty:
        print(f"No recognized statement types for experiment '{experiment_name}'")
        return

    unique_statements = sorted(df["statement"].unique())

    # Create figure with subplots
    fig = plt.figure(figsize=(7.2, 4))
    gs = gridspec.GridSpec(
        7,
        5,
        height_ratios=[0.01, 0.01, 0.01, 0.46, 0.01, 0.46, 0.02],
        width_ratios=[0.04, 0.24, 0.24, 0.24, 0.24],
        hspace=0.7,
        wspace=0.6,
        figure=fig,
    )

    # Row 0 – Title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0,
        0,
        f"Network Impact on Belief Diffusion: {experiment_label}",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#444444",
    )

    # Network properties to analyze
    network_props = [
        ("density", "Density"),
        ("avg_clustering", "Avg. Clustering"),
        ("avg_degree", "Avg. Degree"),
        ("avg_path_length", "Avg. Path Length"),
    ]

    # Row 1 – Network property headers
    for idx, (_prop, label) in enumerate(network_props):
        ax_prop = fig.add_subplot(gs[1, idx + 1])
        ax_prop.axis("off")
        ax_prop.text(
            0.5,
            0,
            label,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="#444444",
        )

    # Row 2 – Panel letters for Consensus Change row
    letters_top = ["a", "b", "c", "d"]
    for idx in range(len(network_props)):
        ax_panel = fig.add_subplot(gs[2, idx + 1])
        ax_panel.axis("off")
        ax_panel.text(
            -0.2,
            -0.2,
            f"({letters_top[idx]})",
            transform=ax_panel.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color="#444444",
        )

    # -------- Collective Accuracy Change  --------
    ax_title_left = fig.add_subplot(gs[3, 0])
    ax_title_left.axis("off")
    ax_title_left.text(
        0.5,
        0.5,
        "Collective Accuracy Change",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#444444",
        rotation=90,
    )

    for idx, (prop, _label) in enumerate(network_props):
        ax_plot = fig.add_subplot(gs[3, idx + 1])
        if prop in df.columns and "accuracy_lift" in df.columns:
            # store correlations per statement
            corr_per_statement = {}

            # Scatter points for each statement
            for statement in unique_statements:
                sub = df[df["statement"] == statement]
                valid_mask = sub[prop].notna() & sub["accuracy_lift"].notna()
                x_data = sub.loc[valid_mask, prop]
                y_data = sub.loc[valid_mask, "accuracy_lift"]

                if len(x_data) == 0:
                    corr_per_statement[statement] = np.nan
                    continue

                color = statement_colors.get(statement, "gray")
                label_str = statement_labels.get(statement, statement)

                ax_plot.scatter(
                    x_data,
                    y_data,
                    alpha=0.6,
                    s=20,
                    color=color,
                    label=label_str,
                )

                # per-statement correlation
                if len(x_data) > 1 and x_data.std() > 1e-10 and y_data.std() > 1e-10:
                    corr_per_statement[statement] = x_data.corr(y_data)
                else:
                    corr_per_statement[statement] = np.nan

            # Dynamic ylim headroom
            ymin, ymax = ax_plot.get_ylim()
            if ymax > ymin:
                margin = 0.3 * (ymax - ymin)
                ax_plot.set_ylim(ymin, ymax + margin)

            # 2x2 r grid in axes coordinates
            pos_grid = {
                "true_doc_1": (0.04, 0.98),
                "true_rest_1": (0.04, 0.88),
                "false_doc_1": (0.45, 0.98),
                "false_rest_1": (0.45, 0.88),
            }

            for s in statement_order:
                if s not in unique_statements:
                    continue
                x_pos, y_pos = pos_grid[s]
                r_val = corr_per_statement.get(s, np.nan)
                if np.isnan(r_val):
                    text_str = "$r$ = 0.000"
                else:
                    text_str = rf"$r$ = {r_val:.3f}"

                # r text with light gray rounded box
                ax_plot.text(
                    x_pos,
                    y_pos,
                    text_str,
                    transform=ax_plot.transAxes,
                    fontsize=5,
                    va="top",
                    ha="left",
                    color="#333333",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="#f5f5f5",
                        edgecolor="none",
                        alpha=0.9,
                    ),
                )
                # colored square subscript
                ax_plot.text(
                    x_pos + 0.02,
                    y_pos - 0.06,
                    r"$\blacksquare$",
                    transform=ax_plot.transAxes,
                    fontsize=3,
                    va="top",
                    ha="left",
                    color=statement_colors.get(s, "gray"),
                )

            ax_plot.tick_params(axis="x", labelsize=7)
            ax_plot.tick_params(axis="y", labelsize=7)
            ax_plot.grid(True, alpha=0.3)
            ax_plot.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    # Row 4 – Panel letters for Score Change row
    letters_bottom = ["e", "f", "g", "h"]
    for idx in range(len(network_props)):
        ax_panel = fig.add_subplot(gs[4, idx + 1])
        ax_panel.axis("off")
        ax_panel.text(
            -0.2,
            -0.2,
            f"({letters_bottom[idx]})",
            transform=ax_panel.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color="#444444",
        )

    # -------- Score Change row --------
    ax_title_left2 = fig.add_subplot(gs[5, 0])
    ax_title_left2.axis("off")
    ax_title_left2.text(
        0.5,
        0.5,
        "Score Change",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#444444",
        rotation=90,
    )

    for idx, (prop, _label) in enumerate(network_props):
        ax_plot = fig.add_subplot(gs[5, idx + 1])
        if prop in df.columns and "score_change" in df.columns:
            corr_per_statement = {}

            for statement in unique_statements:
                sub = df[df["statement"] == statement]
                valid_mask = sub[prop].notna() & sub["score_change"].notna()
                x_data = sub.loc[valid_mask, prop]
                y_data = sub.loc[valid_mask, "score_change"]

                if len(x_data) == 0:
                    corr_per_statement[statement] = np.nan
                    continue

                color = statement_colors.get(statement, "gray")
                label_str = statement_labels.get(statement, statement)

                ax_plot.scatter(
                    x_data,
                    y_data,
                    alpha=0.6,
                    s=20,
                    color=color,
                    label=label_str,
                )

                if len(x_data) > 1 and x_data.std() > 1e-10 and y_data.std() > 1e-10:
                    corr_per_statement[statement] = x_data.corr(y_data)
                else:
                    corr_per_statement[statement] = np.nan

            # Dynamic ylim headroom
            ymin, ymax = ax_plot.get_ylim()
            if ymax > ymin:
                margin = 0.3 * (ymax - ymin)
                ax_plot.set_ylim(ymin, ymax + margin)

            pos_grid = {
                "true_doc_1": (0.04, 0.98),
                "true_rest_1": (0.04, 0.88),
                "false_doc_1": (0.45, 0.98),
                "false_rest_1": (0.45, 0.88),
            }

            for s in statement_order:
                if s not in unique_statements:
                    continue
                x_pos, y_pos = pos_grid[s]
                r_val = corr_per_statement.get(s, np.nan)
                if np.isnan(r_val):
                    text_str = "$r$ = 0.000"
                else:
                    text_str = rf"$r$ = {r_val:.3f}"

                ax_plot.text(
                    x_pos,
                    y_pos,
                    text_str,
                    transform=ax_plot.transAxes,
                    fontsize=5,
                    va="top",
                    ha="left",
                    color="#333333",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="#f5f5f5",
                        edgecolor="none",
                        alpha=0.9,
                    ),
                )
                ax_plot.text(
                    x_pos + 0.02,
                    y_pos - 0.06,
                    r"$\blacksquare$",
                    transform=ax_plot.transAxes,
                    fontsize=3,
                    va="top",
                    ha="left",
                    color=statement_colors.get(s, "gray"),
                )

        ax_plot.tick_params(axis="x", labelsize=7)
        ax_plot.tick_params(axis="y", labelsize=7)
        ax_plot.grid(True, alpha=0.3)
        ax_plot.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    # Row 6 – Legend
    ax_leg = fig.add_subplot(gs[6, 2:4])
    ax_leg.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=7,
            markerfacecolor="tab:blue",
            markeredgecolor="tab:blue",
            label="True (Doc. Correct)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=7,
            markerfacecolor="tab:green",
            markeredgecolor="tab:green",
            label="True (Doc. Incorrect)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=7,
            markerfacecolor="tab:red",
            markeredgecolor="tab:red",
            label="False (Doc. Correct)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=7,
            markerfacecolor="tab:orange",
            markeredgecolor="tab:orange",
            label="False (Doc. Incorrect)",
        ),
    ]
    ax_leg.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        ncol=4,
        fontsize=7,
        frameon=False,
    )

    if output_dir is None:
        output_dir = Path("outputs/plots")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{experiment_name}_network_impact.pdf"
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"Saved aggregated network impact plot for '{experiment_name}' to {out_path}")

    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate network impact across statement types, per experiment type."
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
        "--experiment",
        help="Optional experiment filter (e.g., experts, only_llms, random_experts, random_roles)",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=1,
        help="Minimum number of runs required for a config (default: 1)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    # Discover multi-run experiments (per probe/experiment/prompt/statement)
    print("Discovering multi-run experiments...")
    multi_run_configs = find_multi_run_experiments(args.runs_dir)
    print(multi_run_configs)

    # Filter by minimum runs (per config)
    multi_run_configs = {
        k: v for k, v in multi_run_configs.items() if len(v) >= args.min_runs
    }

    if not multi_run_configs:
        print(f"No multi-run experiments found with at least {args.min_runs} runs.")
        return

    print(f"\nFound {len(multi_run_configs)} multi-run configurations:")
    for config_name, run_dirs in multi_run_configs.items():
        print(f"  {config_name}: {len(run_dirs)} runs")

    # Reorganize configs by (probe, experiment) and statement
    # config_name = "probe/experiment/prompt/statement"
    probe_experiment_to_statements: Dict[tuple, Dict[str, List[Path]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for config_name, run_dirs in multi_run_configs.items():
        parts = config_name.split("/")
        if len(parts) != 4:
            continue
        probe, experiment, _prompt, statement = parts
        probe_experiment_to_statements[(probe, experiment)][statement].extend(run_dirs)

    # Optional experiment filter (still ignores probe; we just keep them separate)
    if args.experiment:
        probe_experiment_to_statements = {
            (probe, exp): stmts
            for (probe, exp), stmts in probe_experiment_to_statements.items()
            if exp == args.experiment
        }
        if not probe_experiment_to_statements:
            print(f"No experiments matched filter '{args.experiment}'.")
            return

    # For each (probe, experiment), aggregate across statement types and plot
    for (probe, experiment), stmts_dict in probe_experiment_to_statements.items():
        print(f"\n{'='*60}")
        print(f"Aggregating probe={probe}, experiment={experiment}")
        print(f"{'='*60}")

        dfs = []
        for statement, run_dirs in stmts_dict.items():
            print(f"  Statement: {statement} ({len(run_dirs)} runs)")
            net_df = analyze_network_structure_impact(run_dirs)
            if net_df.empty:
                print("    -> No network data, skipping.")
                continue
            net_df["statement"] = statement
            dfs.append(net_df)

        if not dfs:
            print(
                f"No usable network data for probe={probe}, experiment='{experiment}'. Skipping."
            )
            continue

        df_exp = pd.concat(dfs, ignore_index=True)

        # Save under outputs/plots/<probe>/<experiment>/
        out_dir_exp = output_dir / probe / experiment
        plot_network_impact_analysis(df_exp, experiment, out_dir_exp)


if __name__ == "__main__":
    main()
