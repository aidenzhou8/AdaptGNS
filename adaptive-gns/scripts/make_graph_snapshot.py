#!/usr/bin/env python3
"""Render side-by-side interaction graph snapshots for the paper.

Produces ``figures/fig_graph_snapshot.pdf`` (Figure 1 in
``CPSC_4830_Final_Report.tex``): the radius graph at step 100 of a
WaterDrop trajectory next to the radius graph at step 100 of a Sand
trajectory.

Usage::

    python adaptive-gns/scripts/make_graph_snapshot.py \\
        --waterdrop_path WaterDrop/dataset/test.npz \\
        --sand_path      Sand/dataset/test.npz \\
        --step 100 --radius 0.015 --traj 0 \\
        --output figures/fig_graph_snapshot.pdf
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree


def load_trajectory(npz_path, traj_idx):
    """Return the ``(T, N, D)`` position array for trajectory ``traj_idx``."""
    with np.load(npz_path, allow_pickle=True) as f:
        keys = list(f.keys())
        if "gns_data" in f:
            entry = f["gns_data"][traj_idx]
        else:
            entry = f[keys[traj_idx]]
    positions = entry[0] if isinstance(entry, (list, tuple, np.ndarray)) else entry
    return np.asarray(positions)


def draw_panel(ax, positions, radius, title):
    tree = cKDTree(positions)
    pairs = list(tree.query_pairs(r=radius))

    if pairs:
        segs = np.stack(
            [np.stack([positions[i], positions[j]], axis=0) for (i, j) in pairs],
            axis=0,
        )
        ax.add_collection(
            LineCollection(segs, linewidths=0.4, alpha=0.35, colors="#1f77b4")
        )

    ax.scatter(
        positions[:, 0], positions[:, 1], s=4, c="#1f77b4", edgecolors="none"
    )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title}  (n={len(positions)}, |E|={len(pairs)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waterdrop_path", required=True)
    ap.add_argument("--sand_path", required=True)
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--radius", type=float, default=0.015)
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--output", default="figures/fig_graph_snapshot.pdf")
    args = ap.parse_args()

    wd = load_trajectory(args.waterdrop_path, args.traj)
    sd = load_trajectory(args.sand_path, args.traj)

    if wd.shape[0] <= args.step or sd.shape[0] <= args.step:
        raise SystemExit(
            f"Trajectory shorter than step={args.step}: "
            f"WaterDrop has {wd.shape[0]} steps, Sand has {sd.shape[0]} steps."
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    draw_panel(axes[0], wd[args.step], args.radius, "WaterDrop")
    draw_panel(axes[1], sd[args.step], args.radius, "Sand")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
