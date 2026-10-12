
import argparse
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


# Style / fonts
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


SCENARIO_MAP = {
    'base_llms': 'I. Base LLMs',
    'random_roles': 'II. Random roles',
    'random_experts': 'III. Spcs (random)',
    'experts': 'IV. Spcs (matched)',
}

SCENARIO_EXTENDED_MAP = {
    'base_llms': 'I. Base LLMs',
    'random_roles': 'II. Random roles',
    'random_experts': 'III. Random specialists',
    'experts': 'IV. Specialists (matched)',
}

SCENARIO_COLOR = {
        'base_llms':     OKABE_ITO["orange"],
        'random_roles':   OKABE_ITO["vermilion"],    
        'random_experts': OKABE_ITO["blue"],
        'experts':        OKABE_ITO["green"],
    }

SCENARIO_MARKER = {
        'base_llms':      "o",
        'random_roles':   "s",
        'random_experts': "D",
        'experts':        "H",
    }

SCENARIO_ORDER = list(SCENARIO_MAP)

GRAPH_MAP = {
    'erdos-renyi': 'Erdős-Rényi',
    'watts-strogatz': 'Watts-Strogatz',
}

GRAPH_COLOR ={
    'erdos-renyi': OKABE_ITO["purple"], 
    'watts-strogatz': OKABE_ITO["green"]
}

GRAPH_SUBTYPE_MAP = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}

GRAPH_SUBTYPE_COLOR = {
    "low": OKABE_ITO["blue"],
    "medium": OKABE_ITO["orange"],
    "high": OKABE_ITO["green"],
}

# Subtypes are ordinal: alphabetical sorting would render them low/medium/high.
GRAPH_SUBTYPE_ORDER = ["low", "medium", "high"]

GRAPH_SUBTYPE_MAP = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}


def _make_room_for_legend(ax, step=0.08, max_steps=15, pad_px=3):
    """Raise the y-axis top until the legend overlaps no bar or error bar.

    The legend keeps `loc="best"`; extra headroom just gives it a free spot above
    the bars. Does nothing when there is no legend or no overlap.
    """
    legend = ax.get_legend()
    if legend is None:
        return
    renderer = ax.figure.canvas.get_renderer()

    def _overlaps(box):
        # Each bar, each error-bar segment and each cap point is tested on its own:
        # a whole errorbar series' bounding box would span every bar.
        if any(box.overlaps(p.get_window_extent(renderer)) for p in ax.patches):
            return True
        for coll in ax.collections:
            tr = coll.get_transform()
            if any(box.overlaps(path.get_extents(tr)) for path in coll.get_paths()):
                return True
        for line in ax.lines:
            pts = line.get_transform().transform(line.get_xydata())
            if any(box.contains(x, y) for x, y in pts):
                return True
        return False

    for _ in range(max_steps):
        ax.figure.canvas.draw()  # legend "best" position is resolved at draw time
        if not _overlaps(legend.get_window_extent(renderer).padded(pad_px)):
            return
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + step * (hi - lo))


def plot_emmeans_grouped_bar(
    emm_df,
    group_col="graph_subtype",
    out_fig=None,
    title="",
    y_label="Estimated marginal mean",
    group_order=None,
    group_colors=None,
    group_labels=None,
    legend_title=None,
    figsize=None,
    show=None,
    ax=None,
    y_format="%.2f",
):
    """Plot grouped emmeans bars with asymmetric CIs, for any number of groups.

    Bar width and offsets are derived from the number of groups, so the same call
    works for the two-level `graph_family` split and for the three-level density
    bins.

    Args:
        emm_df: DataFrame with `run_scenario`, `group_col`, `emmean` and CI bounds.
            CI columns may be named `lower.CL`/`upper.CL`, `LCL`/`UCL`, or the
            `asymp.` variants; all are normalized.
        group_col: Column holding the grouping variable. Defaults to `graph_subtype`.
        out_fig: Optional path to save the figure as PDF.
        title: Plot title.
        y_label: Y-axis label.
        group_order: Explicit left-to-right group order. Defaults to the categorical
            order of the column if it has one, then to `GRAPH_SUBTYPE_ORDER`, then to
            sorted order. Alphabetical sorting is never used for the subtypes,
            which are ordinal.
        group_colors: Mapping group -> colour. Defaults by `group_col`, falling back
            to the Okabe-Ito cycle for unrecognized groups.
        group_labels: Mapping group -> legend label. Defaults by `group_col`.
        legend_title: Legend title. Defaults by `group_col`.
        figsize: Matplotlib figure size. Defaults to a width that scales with the
            number of scenarios and groups.
        show: Whether to call `plt.show()`. Defaults to True only when the function
            creates its own figure, so grid panels are not shown one at a time.
        ax: Optional existing axis to draw on (e.g. one panel of a grid). When given,
            `figsize` is ignored and layout is left to the caller.
        y_format: Y tick format, as a `%`-style string or a matplotlib `Formatter`.

    Returns:
        Tuple (fig, ax, emm_plot) where emm_plot is the normalized plotting DataFrame.
    """
    # Ensure shared style is applied (safe to call repeatedly).
    _configure_fonts()

    emm_plot = emm_df.copy()

    # Normalize CI column names from emmeans output.
    rename_map = {}
    for src_col, dst_col in [("LCL", "lower.CL"), ("asymp.LCL", "lower.CL"),
                             ("UCL", "upper.CL"), ("asymp.UCL", "upper.CL")]:
        if src_col in emm_plot.columns and dst_col not in emm_plot.columns:
            rename_map[src_col] = dst_col
    if rename_map:
        emm_plot = emm_plot.rename(columns=rename_map)

    required_cols = {"run_scenario", group_col, "emmean", "lower.CL", "upper.CL"}
    missing = required_cols.difference(set(emm_plot.columns))
    if missing:
        raise ValueError(f"Missing required columns in emm_df: {sorted(missing)}")

    # Capture any categorical order before the astype(str) below discards it.
    declared_order = None
    if isinstance(emm_plot[group_col].dtype, pd.CategoricalDtype):
        declared_order = [str(c) for c in emm_plot[group_col].cat.categories]

    emm_plot["run_scenario"] = emm_plot["run_scenario"].astype(str)
    emm_plot[group_col] = emm_plot[group_col].astype(str)

    present_scenarios = set(emm_plot["run_scenario"])
    scenario_order = [s for s in SCENARIO_ORDER if s in present_scenarios]
    scenario_order += sorted(present_scenarios.difference(scenario_order))

    present = set(emm_plot[group_col].unique())
    if group_order is None:
        if declared_order is not None:
            group_order = [g for g in declared_order if g in present]
        else:
            known = [g for g in GRAPH_SUBTYPE_ORDER if g in present]
            group_order = known + sorted(present.difference(known))
    else:
        group_order = [str(g) for g in group_order]
        unknown = present.difference(group_order)
        if unknown:
            raise ValueError(f"group_order omits groups present in emm_df: {sorted(unknown)}")
    if not group_order:
        raise ValueError(f"No groups found in column {group_col!r}")

    # Defaults keyed on what is being grouped.
    if group_colors is None:
        group_colors = GRAPH_SUBTYPE_COLOR if group_col == "graph_subtype" else GRAPH_COLOR
    if group_labels is None:
        group_labels = GRAPH_SUBTYPE_MAP if group_col == "graph_subtype" else GRAPH_MAP
    if legend_title is None:
        legend_title = "Network density" if group_col == "graph_subtype" else "Graph type"

    # Any group without an assigned colour falls back to the shared palette.
    fallback = [OKABE_ITO[k] for k in ("blue", "orange", "vermilion", "green", "purple", "sky")]
    colors = [group_colors.get(g, fallback[j % len(fallback)]) for j, g in enumerate(group_order)]

    def _pivot(value_col):
        return (emm_plot.pivot(index="run_scenario", columns=group_col, values=value_col)
                        .reindex(index=scenario_order, columns=group_order)
                        .astype(float))

    pivot_mean, pivot_lcl, pivot_ucl = _pivot("emmean"), _pivot("lower.CL"), _pivot("upper.CL")

    n_groups = len(group_order)
    x = np.arange(len(scenario_order))

    # Split a fixed slot among the groups instead of assuming two bars.
    total_width = 0.8
    width = total_width / n_groups
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2) * width

    ci_color = OKABE_ITO["black"]
    own_fig = ax is None
    if own_fig:
        if figsize is None:
            figsize = (max(4.5, 2.0 + 0.45 * len(scenario_order) * n_groups), 4.0)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for j, g in enumerate(group_order):
        y = pivot_mean[g].to_numpy()

        # Asymmetric 95% CI around emmean.
        yerr = np.vstack([y - pivot_lcl[g].to_numpy(), pivot_ucl[g].to_numpy() - y])

        ax.bar(
            x + offsets[j],
            y,
            width=width * 0.9,
            label=group_labels.get(g, g),
            color=colors[j],
            edgecolor=ci_color,
            linewidth=1.5,
            yerr=yerr,
            capsize=4,
            error_kw={"elinewidth": 1.3, "ecolor": ci_color},
            alpha=1,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_MAP.get(s, s) for s in scenario_order], rotation=15, ha="center")
    ax.set_xlabel("Scenario")
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(
        FormatStrFormatter(y_format) if isinstance(y_format, str) else y_format
    )
    ax.set_title(title)
    ax.legend(title=legend_title, frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=1)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.3, length=5)

    if own_fig:
        plt.tight_layout()
    _make_room_for_legend(ax)

    if out_fig is not None:
        fig.savefig(out_fig, bbox_inches="tight", format="pdf")
        print(f"Saved figure to: {out_fig}")

    if show is None:
        show = own_fig
    if show:
        plt.show()

    return fig, ax, emm_plot

