"""Shared plotting style and helpers for the manuscript figures.

Colorblind-safe palette (Okabe & Ito, 2008) and a consistent font/spine style
are applied everywhere so figures produced across notebooks look like one
system. `plot_emmeans_grouped_bar` renders the grouped estimated-marginal-mean
bar charts (one bar pair per scenario, split by network type) used in
X2_agent_analysis.ipynb and X3_run_analysis.ipynb.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


def _configure_fonts() -> None:
    """Apply the manuscript's matplotlib style. Safe to call repeatedly."""
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
    "orange":    "#E69F00",
    "purple":    "#CC79A7",
    "black":     "#000000",
}

# Scenario labels: short form (figure axes) and extended form (tables).
SETTING_MAP = {
    'base_llms': 'I. Base LLMs',
    'random_roles': 'II. Random roles',
    'random_experts': 'III. Spcs (random)',
    'experts': 'IV. Spcs (matched)',
}

SETTING_EXTENDED_MAP = {
    'base_llms': 'I. Base LLMs',
    'random_roles': 'II. Random roles',
    'random_experts': 'III. Random Specialists',
    'experts': 'IV. Matched Specialists',
}

GRAPH_MAP = {
    'erdos-renyi': 'Erdős-Rényi',
    'watts-strogatz': 'Watts-Strogatz',
}

GRAPH_COLOR = {
    'erdos-renyi': OKABE_ITO["purple"],
    'watts-strogatz': OKABE_ITO["green"],
}


def plot_emmeans_grouped_bar(
    emm_df,
    setting_reference=None,
    out_fig=None,
    title="Estimated Marginal Means by Setting and Graph Type",
    y_label="Estimated marginal mean",
    figsize=(4.5, 4),
    show=True,
):
    """Plot grouped emmeans bars (one pair of bars per scenario) with asymmetric CIs.

    Args:
        emm_df: DataFrame with columns setting, graph_type, emmean, and CI bounds
            (as returned by `r_utils.fit_lmer_full`'s `emm` output).
        setting_reference: Optional DataFrame (e.g. the full data) used only to
            preserve the scenario display order (I -> IV).
        out_fig: Optional file path to save the figure as a PDF.
        title: Plot title.
        y_label: Y-axis label.
        figsize: Matplotlib figure size tuple.
        show: Whether to call plt.show().

    Returns:
        Tuple (fig, ax, emm_plot) where emm_plot is the normalized plotting DataFrame.
    """
    _configure_fonts()

    emm_plot = emm_df.copy()

    # Normalize CI column names from emmeans output (Satterthwaite vs. asymptotic).
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

    graph_colors = [GRAPH_COLOR['erdos-renyi'], GRAPH_COLOR['watts-strogatz']]
    ci_color = OKABE_ITO["black"]

    fig, ax = plt.subplots(figsize=figsize)
    for j, g in enumerate(graph_order):
        y = pivot_mean[g].to_numpy()

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

    setting_labels = [SETTING_MAP.get(s, s) for s in setting_order]
    ax.set_xticks(x)
    ax.set_xticklabels(setting_labels, rotation=15, ha="center")
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
        fig.savefig(out_fig, bbox_inches="tight", format="pdf")
        print(f"Saved figure to: {out_fig}")

    if show:
        plt.show()

    return fig, ax, emm_plot
