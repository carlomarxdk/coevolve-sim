
import argparse
from curses import OK
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontManager
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Style / fonts (matches your other scripts)
# -------------------------
def _configure_fonts() -> None:
    desired_fonts = {
        "serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "monospace": ["Courier New", "Courier", "DejaVu Sans Mono"],
    }

    mpl.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.serif": desired_fonts["serif"][0],
            "font.sans-serif": desired_fonts["sans-serif"][0],
            "font.monospace": desired_fonts["monospace"][0],
            "axes.titlesize": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,       # TrueType, editable text in PDF
            "ps.fonttype": 42,        # TrueType, editable text in PS
            "svg.fonttype": "none",   # Keep text as <text> in SVG
        }
    )

_configure_fonts()

# Colorblind-safe palette (Okabe-Ito)
OKABE_ITO = {
    "yellow":    "#F0E442",
    "sky":       "#56B4E9",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "green":     "#009E73",
    "sky":       "#56B4E9",
    "orange":    "#E69F00",
    "purple":    "#CC79A7",
    "black":     "#000000",
}


SETTING_MAP = {
    'base_llms': 'I. Base LLMs',
    'random_roles': 'II. Random roles',
    'random_experts': 'III. Random experts',
    'experts': 'IV. Expert (matching)',
}

SETTING_COLOR = {
        'base_llms':     OKABE_ITO["orange"],
        'random_roles':   OKABE_ITO["vermilion"],
        'random_experts': OKABE_ITO["blue"],
        'experts':        OKABE_ITO["green"],
    }

SETTING_MARKER = {
        'base_llms':      "o",
        'random_roles':   "s",
        'random_experts': "D",
        'experts':        "H",
    }

# COL_PRETTY_NAMES = {
#     "influence_in": "Incoming Influence (Magnitude)",
#     "influence_out": "Outgoing Influence (Magnitude)",
#     "influence_in_dir": "Incoming Influence (Direction)",
#     "influence_out_dir": "Outgoing Influence (Direction)",
#     "plasticity_tv": "Plasticity",
#     "plasticity_tv_1": "Plasticity (without baseline)",
#     "plasticity_false": "Plasticity (False)",
#     "plasticity_true": "Plasticity (True)",
#     "plasticity_neither": "Plasticity (Uncertain)",
#     "monotonicity": "Directionality",
#     'net_change': 'Net Drift',
#     'net_gain': 'Net Gain to Truth',
#     'progress_efficiency': 'Progress Efficiency',
#     # 'net_drift': 'Net Drift towards Correct',
#     'agent_accuracy': 'Agent Accuracy',
#     'agent_error': 'Agent Error',
#     # 'progress_ratio': 'Progress Ratio',
#     'progress_forward': 'Drift to Truth',
#     'progress_backward': 'Drift away from Truth',
# }

# COL_RUN_PRETTY_NAMES = {
#     'modal_consensus_T': 'Consensus',
#     # 'accuracy_0': 'Initial Accuracy',
#     'accuracy_T': 'Final Accuracy',
#     # 'diff_accuracy': 'Accuracy Change',
#     'modal_consensus_mean': 'Consensus (mean)',
#     'modal_consensus_change': 'Consensus Change',
#     'polarization_T': 'Polarization',
#     'polarization_change': 'Polarization Change',
#     'entropy_T': 'Entropy',
#     'entropy_change': 'Entropy Change',
# }


GRAPH_MAP = {
    'erdos-renyi': 'Erdős-Rényi',
    'watts-strogatz': 'Watts-Strogatz',
}

GRAPH_COLOR ={
    'erdos-renyi': OKABE_ITO["purple"],
    'watts-strogatz': OKABE_ITO["green"]
}


def plot_emmeans_grouped_bar(
    emm_df,
    setting_reference=None,
    out_fig=None,
    title="Estimated Marginal Means by Setting and Graph Type",
    y_label = "Estimated marginal mean",
    figsize=(4.5, 4),
    show=True,
):
    """Plot grouped emmeans bars with asymmetric CIs for two graph types.

    Args:
        emm_df: DataFrame with columns including setting, graph_type, emmean, and CI bounds.
        setting_reference: Optional DataFrame (e.g., dft) used only to preserve setting order.
        out_fig: Optional file path to save the figure.
        title: Plot title.
        figsize: Matplotlib figure size tuple.
        show: Whether to call plt.show().

    Returns:
        Tuple (fig, ax, emm_plot) where emm_plot is the normalized plotting DataFrame.
    """
    # Ensure shared style is applied (safe to call repeatedly).
    _configure_fonts()

    emm_plot = emm_df.copy()

    # Normalize CI column names from emmeans output.
    rename_map = {}
    if "LCL" in emm_plot.columns and "lower.CL" not in emm_plot.columns:
        rename_map["LCL"] = "lower.CL"
    if "asymp.LCL" in emm_plot.columns and "lower.CL" not in emm_plot.columns:
        rename_map["asymp.LCL"] = "lower.CL"
    if "UCL" in emm_plot.columns and "upper.CL" not in emm_plot.columns:
        rename_map["UCL"] = "upper.CL"
    if "asymp.UCL" in emm_plot.columns and "upper.CL" not in emm_plot.columns:
        rename_map["asymp.UCL"] = "upper.CL"
    if rename_map:
        emm_plot = emm_plot.rename(columns=rename_map)

    required_cols = {"setting", "graph_type", "emmean", "lower.CL", "upper.CL"}
    missing = required_cols.difference(set(emm_plot.columns))
    if missing:
        raise ValueError(f"Missing required columns in emm_df: {sorted(missing)}")

    emm_plot["setting"] = emm_plot["setting"].astype(str)
    emm_plot["graph_type"] = emm_plot["graph_type"].astype(str)

    # Keep setting order consistent with reference table when provided.
    if setting_reference is not None and "setting" in setting_reference.columns:
        setting_order = [
            s for s in setting_reference["setting"].astype(str).dropna().unique()
            if s in set(emm_plot["setting"])
        ]
    else:
        setting_order = sorted(emm_plot["setting"].unique())

    graph_order = sorted(emm_plot["graph_type"].unique())
    if len(graph_order) != 2:
        raise ValueError(
            f"Expected exactly 2 graph types, got {len(graph_order)}: {graph_order}"
        )

    pivot_mean = (
        emm_plot.pivot(index="setting", columns="graph_type", values="emmean")
        .reindex(index=setting_order, columns=graph_order)
        .astype(float)
    )
    pivot_lcl = (
        emm_plot.pivot(index="setting", columns="graph_type", values="lower.CL")
        .reindex(index=setting_order, columns=graph_order)
        .astype(float)
    )
    pivot_ucl = (
        emm_plot.pivot(index="setting", columns="graph_type", values="upper.CL")
        .reindex(index=setting_order, columns=graph_order)
        .astype(float)
    )

    x = np.arange(len(setting_order))
    width = 0.4

    # Use project palette for graph groups.
    graph_colors = [GRAPH_COLOR['erdos-renyi'], GRAPH_COLOR['watts-strogatz']]
    ci_color = OKABE_ITO["black"]

    fig, ax = plt.subplots(figsize=figsize)
    for j, g in enumerate(graph_order):
        y = pivot_mean[g].to_numpy()

        # Asymmetric 95% CI around emmean.
        yerr_lower = y - pivot_lcl[g].to_numpy()
        yerr_upper = pivot_ucl[g].to_numpy() - y

        xpos = x + (j - 0.5) * width
        ax.bar(
            xpos,
            y,
            width=width - 0.05,
            label=GRAPH_MAP.get(g, g),
            color=graph_colors[j % len(graph_colors)],
            edgecolor=ci_color,
            linewidth=1.5,
            yerr=np.vstack([yerr_lower, yerr_upper]),
            capsize=4,
            error_kw={"elinewidth": 1.3, "ecolor": ci_color},
            alpha=1,
        )


    setting_order = [SETTING_MAP.get(s, s) for s in setting_order]
    ax.set_xticks(x)
    ax.set_xticklabels(setting_order, rotation=15, ha="center")
    ax.set_xlabel("Setting")
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_title(title)
    ax.legend(title="Graph type", frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=1)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.3, length=5)


    plt.tight_layout()

    if out_fig is not None:
        fig.savefig(out_fig,  bbox_inches="tight", format="pdf")
        print(f"Saved figure to: {out_fig}")

    if show:
        plt.show()

    return fig, ax, emm_plot


# def plot_emmeans_dual_grouped_bar(
#     emm_df_primary,
#     emm_df_secondary,
#     metric_names=("Primary", "Secondary"),
#     setting_reference=None,
#     out_fig=None,
#     title="Estimated Marginal Means Comparison",
#     y_label="Estimated marginal mean",
#     figsize=(8, 5),
#     show=True,
# ):
#     """Plot dual-metric grouped bars with asymmetric CIs for two graph types.

#     Plots two related metrics side-by-side for each setting-graph combination.
#     Primary metric uses full opacity with solid fill. Secondary metric uses
#     hatching patterns and reduced opacity (alpha=0.7) for visual distinction.

#     Args:
#         emm_df_primary: DataFrame with columns setting, graph_type, emmean, lower.CL, upper.CL.
#         emm_df_secondary: DataFrame with same structure as emm_df_primary.
#         metric_names: Tuple of (primary_label, secondary_label) for legend.
#         setting_reference: Optional DataFrame to preserve setting order.
#         out_fig: Optional file path to save the figure.
#         title: Plot title.
#         y_label: Y-axis label.
#         figsize: Matplotlib figure size tuple.
#         show: Whether to call plt.show().

#     Returns:
#         Tuple (fig, ax) where fig and ax are matplotlib figure and axes objects.
#     """
#     _configure_fonts()

#     # Process primary dataframe
#     emm_primary = emm_df_primary.copy()
#     rename_map = {}
#     if "LCL" in emm_primary.columns and "lower.CL" not in emm_primary.columns:
#         rename_map["LCL"] = "lower.CL"
#     if "asymp.LCL" in emm_primary.columns and "lower.CL" not in emm_primary.columns:
#         rename_map["asymp.LCL"] = "lower.CL"
#     if "UCL" in emm_primary.columns and "upper.CL" not in emm_primary.columns:
#         rename_map["UCL"] = "upper.CL"
#     if "asymp.UCL" in emm_primary.columns and "upper.CL" not in emm_primary.columns:
#         rename_map["asymp.UCL"] = "upper.CL"
#     if rename_map:
#         emm_primary = emm_primary.rename(columns=rename_map)

#     # Process secondary dataframe
#     emm_secondary = emm_df_secondary.copy()
#     rename_map = {}
#     if "LCL" in emm_secondary.columns and "lower.CL" not in emm_secondary.columns:
#         rename_map["LCL"] = "lower.CL"
#     if "asymp.LCL" in emm_secondary.columns and "lower.CL" not in emm_secondary.columns:
#         rename_map["asymp.LCL"] = "lower.CL"
#     if "UCL" in emm_secondary.columns and "upper.CL" not in emm_secondary.columns:
#         rename_map["UCL"] = "upper.CL"
#     if "asymp.UCL" in emm_secondary.columns and "upper.CL" not in emm_secondary.columns:
#         rename_map["asymp.UCL"] = "upper.CL"
#     if rename_map:
#         emm_secondary = emm_secondary.rename(columns=rename_map)

#     # Validate required columns
#     required_cols = {"setting", "graph_type", "emmean", "lower.CL", "upper.CL"}
#     for name, df in [("primary", emm_primary), ("secondary", emm_secondary)]:
#         missing = required_cols.difference(set(df.columns))
#         if missing:
#             raise ValueError(f"Missing columns in {name} df: {sorted(missing)}")

#     emm_primary["setting"] = emm_primary["setting"].astype(str)
#     emm_primary["graph_type"] = emm_primary["graph_type"].astype(str)
#     emm_secondary["setting"] = emm_secondary["setting"].astype(str)
#     emm_secondary["graph_type"] = emm_secondary["graph_type"].astype(str)

#     # Determine setting order
#     if setting_reference is not None and "setting" in setting_reference.columns:
#         setting_order = [
#             s for s in setting_reference["setting"].astype(str).dropna().unique()
#             if s in set(emm_primary["setting"])
#         ]
#     else:
#         setting_order = sorted(emm_primary["setting"].unique())

#     graph_order = sorted(emm_primary["graph_type"].unique())
#     if len(graph_order) != 2:
#         raise ValueError(
#             f"Expected exactly 2 graph types, got {len(graph_order)}: {graph_order}"
#         )

#     # Pivot both datasets
#     def pivot_df(df, order_settings, order_graphs):
#         mean_pivot = (
#             df.pivot(index="setting", columns="graph_type", values="emmean")
#             .reindex(index=order_settings, columns=order_graphs)
#             .astype(float)
#         )
#         lcl_pivot = (
#             df.pivot(index="setting", columns="graph_type", values="lower.CL")
#             .reindex(index=order_settings, columns=order_graphs)
#             .astype(float)
#         )
#         ucl_pivot = (
#             df.pivot(index="setting", columns="graph_type", values="upper.CL")
#             .reindex(index=order_settings, columns=order_graphs)
#             .astype(float)
#         )
#         return mean_pivot, lcl_pivot, ucl_pivot

#     mean_primary, lcl_primary, ucl_primary = pivot_df(
#         emm_primary, setting_order, graph_order
#     )
#     mean_secondary, lcl_secondary, ucl_secondary = pivot_df(
#         emm_secondary, setting_order, graph_order
#     )

#     x = np.arange(len(setting_order))
#     width = 0.4
#     bar_offset = 0.1

#     # Use project palette for graph groups
#     graph_colors = [OKABE_ITO["sky"], OKABE_ITO["green"]]
#     ci_color = OKABE_ITO["black"]

#     # Hatching patterns by graph type
#     hatch_patterns = ["///", "\\\\\\"]

#     fig, ax = plt.subplots(figsize=figsize)

#     # Plot primary (solid) and secondary (hatched) bars for each graph type
#     for j, g in enumerate(graph_order):
#         y_primary = mean_primary[g].to_numpy()
#         yerr_lower_primary = y_primary - lcl_primary[g].to_numpy()
#         yerr_upper_primary = ucl_primary[g].to_numpy() - y_primary

#         y_secondary = mean_secondary[g].to_numpy()
#         yerr_lower_secondary = y_secondary - lcl_secondary[g].to_numpy()
#         yerr_upper_secondary = ucl_secondary[g].to_numpy() - y_secondary

#         # Positions for primary and secondary bars within each graph group
#         xpos_primary = x + (j - 0.5) * width - bar_offset
#         xpos_secondary = x + (j - 0.5) * width + bar_offset

#         bar_width = 0.18

#         # Plot primary bars (solid, full opacity)
#         ax.bar(
#             xpos_primary,
#             y_primary,
#             width=bar_width,
#             label=f"{metric_names[0]} ({GRAPH_MAP.get(g, g)})",
#             color=graph_colors[j],
#             edgecolor=ci_color,
#             linewidth=1.5,
#             yerr=np.vstack([yerr_lower_primary, yerr_upper_primary]),
#             capsize=4,
#             error_kw={"elinewidth": 1.3, "ecolor": ci_color},
#             alpha=0.92,
#         )

#         # Plot secondary bars (hatched, reduced opacity)
#         ax.bar(
#             xpos_secondary,
#             y_secondary,
#             width=bar_width,
#             label=f"{metric_names[1]} ({GRAPH_MAP.get(g, g)})",
#             color=graph_colors[j],
#             hatch=hatch_patterns[j],
#             edgecolor=ci_color,
#             linewidth=1.5,
#             yerr=np.vstack([yerr_lower_secondary, yerr_upper_secondary]),
#             capsize=4,
#             error_kw={"elinewidth": 1.3, "ecolor": ci_color},
#             alpha=0.7,
#         )

#         # Overlay scatter points for both metrics
#         for _, r in emm_primary.iterrows():
#             i = setting_order.index(str(r["setting"]))
#             j_check = graph_order.index(str(r["graph_type"]))
#             if j_check == j:
#                 xpos = x[i] + (j - 0.5) * width - bar_offset
#                 ax.scatter(xpos, float(r["emmean"]), color=ci_color, s=12, zorder=4)

#         for _, r in emm_secondary.iterrows():
#             i = setting_order.index(str(r["setting"]))
#             j_check = graph_order.index(str(r["graph_type"]))
#             if j_check == j:
#                 xpos = x[i] + (j - 0.5) * width + bar_offset
#                 ax.scatter(xpos, float(r["emmean"]), color=ci_color, s=12, zorder=4)

#     # Format axes
#     setting_order_mapped = [SETTING_MAP.get(s, s) for s in setting_order]
#     ax.set_xticks(x)
#     ax.set_xticklabels(setting_order_mapped, rotation=15, ha="center")
#     ax.set_xlabel("Setting")
#     ax.set_ylabel(y_label)
#     ax.set_title(title)
#     ax.legend(loc="upper left", fontsize=10)
#     ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=1)
#     ax.set_axisbelow(True)

#     plt.tight_layout()

#     if out_fig is not None:
#         fig.savefig(out_fig, bbox_inches="tight", format="pdf")
#         print(f"Saved figure to: {out_fig}")

#     if show:
#         plt.show()

#     return fig, ax
