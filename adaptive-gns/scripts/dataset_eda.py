"""Compute graph statistics at a chosen rollout step on a GNS .npz dataset.

Usage:
    python adaptive-gns/scripts/dataset_eda.py \
        --data_path WaterDrop/dataset/test.npz \
        --step 100 --radius 0.015 --n_trajectories 30

Reports nodes, edges, density, average degree, and average clustering
coefficient (mean ± std across the chosen number of trajectories).
"""

import argparse
import json
import sys

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def load_trajectories(npz_path):
    """Return a list of (positions, particle_type[, material]) tuples.

    positions has shape (sequence_length, n_particles, dimension).
    Supports both gns_data layout and per-key layouts.
    """
    with np.load(npz_path, allow_pickle=True) as f:
        if "gns_data" in f:
            data = f["gns_data"]
        else:
            data = [item for _, item in f.items()]
    return list(data)


def graph_stats(positions_TND, step, radius):
    """Build the radius graph at the given step and return (n, e, density, deg, clust)."""
    pos = positions_TND[step]
    n = int(pos.shape[0])
    if n < 2:
        return n, 0, 0.0, 0.0, 0.0
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=radius)
    e = len(pairs)
    if e == 0:
        return n, 0, 0.0, 0.0, 0.0
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(pairs)
    density = (2.0 * e) / (n * (n - 1))
    avg_degree = (2.0 * e) / n
    avg_clustering = nx.average_clustering(G)
    return n, e, density, avg_degree, avg_clustering


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="path to .npz dataset")
    ap.add_argument("--step", type=int, default=100, help="rollout step at which to compute stats")
    ap.add_argument("--radius", type=float, default=0.015, help="connectivity radius")
    ap.add_argument("--n_trajectories", type=int, default=30, help="number of trajectories to process")
    ap.add_argument("--output_json", type=str, default=None, help="optional path to write JSON summary")
    args = ap.parse_args()

    print(f"Loading {args.data_path}...")
    data = load_trajectories(args.data_path)
    total = len(data)
    n_use = min(args.n_trajectories, total)
    print(f"  total trajectories: {total}; using first {n_use} for EDA")
    print(f"  step: {args.step}, radius: {args.radius}")

    nodes, edges, density, degree, clustering = [], [], [], [], []
    for i in range(n_use):
        item = data[i]
        positions = item[0]
        if positions.shape[0] <= args.step:
            print(f"  trajectory {i}: only {positions.shape[0]} steps, skipping")
            continue
        n, e, dens, deg, clust = graph_stats(positions, args.step, args.radius)
        nodes.append(n)
        edges.append(e)
        density.append(dens)
        degree.append(deg)
        clustering.append(clust)
        if (i + 1) % 10 == 0 or i == n_use - 1:
            print(f"  processed {i + 1}/{n_use} trajectories")

    def stat(name, arr, fmt="{:.4g}"):
        a = np.asarray(arr, dtype=float)
        return {
            "name": name,
            "mean": float(a.mean()),
            "std": float(a.std()),
            "pretty": f"{fmt.format(a.mean())} ± {fmt.format(a.std())}",
        }

    summary = {
        "data_path": args.data_path,
        "step": args.step,
        "radius": args.radius,
        "n_trajectories_used": len(nodes),
        "n_trajectories_total": total,
        "stats": {
            "nodes": stat("nodes", nodes, "{:.0f}"),
            "edges": stat("edges", edges, "{:.0f}"),
            "density": stat("density", density, "{:.3e}"),
            "avg_degree": stat("avg_degree", degree, "{:.3f}"),
            "avg_clustering": stat("avg_clustering", clustering, "{:.3f}"),
        },
    }

    print()
    print("=" * 64)
    print(f"EDA: {args.data_path}, step {args.step}, r = {args.radius}, "
          f"{len(nodes)} trajectories")
    print("-" * 64)
    for k in ("nodes", "edges", "density", "avg_degree", "avg_clustering"):
        print(f"  {k:<16}{summary['stats'][k]['pretty']}")
    print("=" * 64)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary to {args.output_json}")


if __name__ == "__main__":
    main()
