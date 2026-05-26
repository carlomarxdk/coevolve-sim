"""YAML configuration file generation for selected statements."""

from pathlib import Path
from textwrap import dedent

import polars as pl
import yaml


# Custom representer to force literal block scalar for multiline strings
def _str_representer(dumper, data):
    """Force literal block scalar style for multiline strings in YAML."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, _str_representer)


STAT_FIELDS = [
    "label_ground_truth",
    "doc_predicted_label",
    "doc_prob",
    "doc_accuracy",
    "other_frac_predicted_label",
    "other_accuracy",
    "other_median_prob",
    "consensus_score",
]


def write_statement_configs(
    df: pl.DataFrame,
    selected_ids: list[int],
    out_dir: Path = Path("configs/statement"),
    stat_fields: list[str] | None = None,
) -> list[Path]:
    """Write YAML configs for selected statements with sequential numbering per label.

    Generates Hydra-compatible YAML configuration files for each selected statement.
    Files are named {true|false}_{0-14}.yaml with independent sequential numbering
    per label group.

    Args:
        df: DataFrame with selected statements (must include "selection_order" column).
        selected_ids: List of statement indices to write configs for.
        out_dir: Output directory for YAML files.
        stat_fields: Fields to include in stats section. Uses default if None.

    Returns:
        List of paths to created YAML files.

    Note:
        The DataFrame must have a "selection_order" column to ensure proper
        sequential numbering. Use the output from select_diverse_statements().

    Example:
        >>> paths = write_statement_configs(df_selected, selected_ids)
        >>> len(paths)
        30
        >>> paths[0].name
        'false_0.yaml'
    """
    if stat_fields is None:
        stat_fields = STAT_FIELDS

    out_dir.mkdir(parents=True, exist_ok=True)

    desc = dedent(
        """\
    Auto-generated from maximin selection pipeline.

    Explaining fields in stats (doctor LLM is m42-health/Llama3-Med42-8B):
    - label_ground_truth: Statement label (binary, 1=True, 0=False)
    - doc_predicted_label: Predicted label by doctor LLM (binary, 1=True, 0=False)
    - doc_prob: Probability assigned by doctor LLM to statement being true
    - doc_accuracy: Accuracy of doctor LLM prediction vs ground truth
    - other_frac_predicted_label: Fraction of other LLMs predicting statement as true
    - other_accuracy: Accuracy of other LLM predictions vs ground truth
    - other_median_prob: Median probability from other LLMs that statement is true
    - consensus_score: Agreement between doctor LLM and other LLM probabilities (not accuracy!)

    To ensure diversity, statements were selected using a maximin criterion on the following features:
    - label_ground_truth
    - doc_predicted_label
    - doc_accuracy
    - other_accuracy
    - consensus_score

    """
    ).strip()

    # Filter and sort by selection_order
    df_filtered = df.filter(pl.col("idx").is_in(selected_ids)).sort("selection_order")

    # Track sequential IDs per label
    label_counters = {"false": 0, "true": 0}
    paths: list[Path] = []

    for row in df_filtered.to_dicts():
        label_key = "true" if row["label_ground_truth"] else "false"
        label_id = label_counters[label_key]
        label_counters[label_key] += 1

        cfg_id = f"{label_key}_{label_id}"
        cfg = {
            "defaults": ["base", "_self_"],
            "id": cfg_id,
            "statement": row["statement"],
            "label": {
                "correct": bool(row["label_ground_truth"]),
                "neither": False,
                "negated": False,
            },
            "description": desc,
            "stats": {
                k: (
                    round(float(row[k]), 5)
                    if isinstance(row[k], (int, float))
                    else row[k]
                )
                for k in stat_fields
            },
        }
        path = out_dir / f"{cfg_id}.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(
                cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False
            )
        paths.append(path)

    return paths
