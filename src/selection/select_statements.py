#!/usr/bin/env python3
"""Script for selecting diverse medical statements using maximin criterion.

This script loads medical statement predictions from multiple LLMs, computes
diversity features, and selects K statements with balanced true/false labels
while maximizing feature space diversity.
"""

import argparse
from pathlib import Path

from config_writer import write_statement_configs

from diversity import diversity_report
from statement_selection import load_statement_data, select_diverse_statements



def main():
    """Select diverse medical statements and generate config files."""
    parser = argparse.ArgumentParser(
        description="Select diverse medical statements for experiments"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("resources/data/predictions"),
        help="Directory containing prediction CSV files",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=30,
        help="Number of statements to select (must be even)",
    )
    parser.add_argument(
        "--doctor-model",
        type=str,
        default="llama-doc",
        help="Name of the doctor/expert model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/statement"),
        help="Output directory for YAML config files",
    )
    parser.add_argument(
        "--init-statements",
        nargs="+",
        help="Optional initial statements to include",
    )
    parser.add_argument(
        "--show-diversity",
        action="store_true",
        help="Print diversity diagnostics",
    )

    args = parser.parse_args()

    if args.k % 2 != 0:
        parser.error(f"K must be even (got {args.k})")

    # Load data
    print("Loading statement predictions...")
    df_preds, valid_idx = load_statement_data(
        args.data_dir,
        doctor_model=args.doctor_model,
    )
    print(f"Loaded {len(df_preds)} statements with predictions from 16 models")

    # Define features for diversity
    feature_cols = [
        "label_ground_truth",
        "doc_predicted_label",
        "doc_accuracy",
        "other_accuracy",
        "consensus_score",
    ]

    # Select diverse statements
    print(f"\nSelecting {args.k} diverse statements using balanced maximin...")
    selected_ids, df_selected, Xz = select_diverse_statements(
        df_preds,
        K=args.k,
        feature_cols=feature_cols,
        init_statements=args.init_statements,
    )

    # Count labels
    n_true = sum(
        df_selected.filter(df_selected["label_ground_truth"] == 1).height for _ in [0]
    )
    n_false = len(selected_ids) - n_true
    print(f"Selected {n_true} true statements and {n_false} false statements")

    # Diversity diagnostics
    if args.show_diversity:
        print("\nDiversity diagnostics:")
        report = diversity_report(df_preds, selected_ids, feature_cols)
        print(f"  Min pairwise distance: {report['min_pairwise_dist']:.3f}")
        print(f"  Mean pairwise distance: {report['mean_pairwise_dist']:.3f}")
        print(f"  Median pairwise distance: {report['median_pairwise_dist']:.3f}")
        print("\n  Feature variance:")
        for feat, var in report["feature_variance"].items():
            print(f"    {feat}: {var:.4f}")
        print("\n  Feature range:")
        for feat, rng in report["feature_range"].items():
            print(f"    {feat}: {rng:.4f}")
        print("\n  PCA explained variance ratios:")
        for i, ratio in enumerate(report["pca_explained_ratio"], 1):
            print(f"    PC{i}: {ratio:.1%}")

    # Write config files
    print(f"\nWriting YAML configs to {args.output_dir}...")
    out_paths = write_statement_configs(
        df=df_selected,
        selected_ids=selected_ids,
        out_dir=args.output_dir,
    )
    print(f"Created {len(out_paths)} config files")

    # Count files by label
    true_files = [p for p in out_paths if p.stem.startswith("true_")]
    false_files = [p for p in out_paths if p.stem.startswith("false_")]
    print(
        f"  - {len(true_files)} true statement configs (true_0.yaml to true_{len(true_files) - 1}.yaml)"
    )
    print(
        f"  - {len(false_files)} false statement configs (false_0.yaml to false_{len(false_files) - 1}.yaml)"
    )


if __name__ == "__main__":
    main()
