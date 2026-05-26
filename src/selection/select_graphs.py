#!/usr/bin/env python3
"""Script for selecting diverse network graphs using maximin criterion.

This script generates a pool of candidate graphs and selects K diverse graphs
based on structural properties: mean shortest path, clustering coefficient,
degree statistics, etc.
"""

import argparse
from pathlib import Path

import numpy as np

from graph_selection import generate_graph_pool, select_diverse_graphs


def main():
    """Select diverse network graphs and save seeds."""
    parser = argparse.ArgumentParser(
        description="Select structurally diverse network graphs"
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="ER",
        choices=["ER", "WS"],
        help="Graph generator type (ER=Erdős-Rényi, WS=Watts-Strogatz)",
    )
    parser.add_argument("--n", type=int, default=48, help="Number of nodes per graph")
    parser.add_argument(
        "--pool-size",
        type=int,
        default=1000,
        help="Number of candidate graphs to generate",
    )
    parser.add_argument(
        "--k", type=int, default=8, help="Number of diverse graphs to select"
    )
    parser.add_argument(
        "--master-seed", type=int, default=814183, help="Master random seed"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resources/seeds/graph_seeds.txt"),
        help="Output file for selected seeds",
    )

    # Generator-specific parameters
    parser.add_argument(
        "--er-p", type=float, default=0.3, help="Edge probability for ER graphs"
    )
    parser.add_argument(
        "--ws-k", type=int, default=8, help="Number of nearest neighbors for WS graphs"
    )
    parser.add_argument(
        "--ws-beta",
        type=float,
        default=0.1,
        help="Rewiring probability for WS graphs",
    )

    args = parser.parse_args()

    # Generate master seeds
    print(f"Generating {args.pool_size} candidate graphs...")
    master_rng = np.random.Generator(np.random.PCG64(args.master_seed))
    seeds = master_rng.integers(low=0, high=2**32, size=args.pool_size, dtype=np.uint32)

    # Build configuration template
    if args.generator == "ER":
        cfg_template = {
            "seed": None,
            "network": {"generator": "ER", "params": {"n": args.n, "p": args.er_p}},
        }
    else:  # WS
        cfg_template = {
            "seed": None,
            "network": {
                "generator": "WS",
                "params": {"n": args.n, "k": args.ws_k, "beta": args.ws_beta},
            },
        }

    # Generate pool
    X, component_counts = generate_graph_pool(cfg_template, seeds, N=args.n)
    print(f"Generated {len(X)} graphs")

    # Select diverse graphs
    print(f"\nSelecting {args.k} diverse graphs using maximin criterion...")
    selected_indices, Xz = select_diverse_graphs(
        X, component_counts, K=args.k, require_connected=True
    )

    # Check connectivity
    all_connected = all(component_counts[i] == 1 for i in selected_indices)
    print(f"All selected graphs connected: {all_connected}")

    # Print statistics
    print("\nSelected graph statistics:")
    feature_names = [
        "mean_sp",
        "global_clust",
        "mean_deg",
        "std_deg",
        "node0_deg",
    ]
    for i in selected_indices:
        row = X[i]
        seed_val = seeds[i]
        stats = ", ".join(
            f"{name}={val:.3f}" for name, val in zip(feature_names, row, strict=True)
        )
        print(f"  Seed {seed_val:>11}: {stats}")

    # Save seeds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected_seeds = seeds[selected_indices].tolist()
    with args.output.open("w") as f:
        f.write("\n".join(map(str, selected_seeds)))
    print(f"\nSaved {len(selected_seeds)} seeds to {args.output}")


if __name__ == "__main__":
    main()
