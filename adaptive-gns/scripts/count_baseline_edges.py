#!/usr/bin/env python3
"""
Model-independent edge-count sanity check for fixed-radius baselines.

Iterates over every frame of each test trajectory's *ground-truth* positions
and reports the average edge count of the radius graph at:
  - the base radius (k=5 baseline)
  - radius * (k10_radius / base_radius) (k=10 baseline)

This number is the GT-trajectory-averaged equivalent of what the trained
baselines would consume if their rollouts perfectly tracked GT. The trained-
model's true rollout-averaged edge count is now emitted by
`evaluate_rollout_mse.py` as `mean_edges_per_step` -- prefer that for
fair comparison against `evaluate_adaptive_rollout.py`'s adaptive number,
since the adaptive number is also rollout-averaged on model predictions.

NOTE: An earlier version of this script sampled a single frame per
trajectory and called the result "avg edges/step", justified by an
incorrect assumption that "the graph is rebuilt identically at every
step". The radius is fixed but particles MOVE during simulation (sand
piles up), so edge density grows over time. Using a single-frame snapshot
underestimates the true rollout average, which makes the snapshot value
look smaller than the adaptive model's mean_edges_per_step and produces
an apparent "adaptive > k=10" inversion that is purely an artifact of
comparing different aggregations.

Usage (run from Misha after activating the gns conda env):
  python count_baseline_edges.py --data_path /path/to/Sand/dataset/

Outputs:
  Model            radius   GT-avg edges/step
  k=5 (base)       0.015               13xxx
  k=10             0.019               21xxx
"""
import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gns import data_loader, reading_utils

INPUT_SEQUENCE_LENGTH = 6
KINEMATIC_PARTICLE_ID = 3


def count_edges_at(positions_t, connectivity_radius, device):
    """Count edges of the radius graph for a single (n_particles, dim) snapshot."""
    from torch_geometric.nn import radius_graph
    n = positions_t.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=device)
    edge_index = radius_graph(positions_t.to(device), r=connectivity_radius, batch=batch,
                              loop=True, max_num_neighbors=128)
    return edge_index.shape[1]


def trajectory_avg_edges(positions, connectivity_radius, device, stride=1):
    """Average edges per frame across an entire (n_particles, T, dim) trajectory."""
    T = positions.shape[1]
    counts = []
    # Skip the (T, T-1, ...) first INPUT_SEQUENCE_LENGTH-1 frames? No: include all
    # frames; the radius graph is well-defined at every frame and the rollout
    # would visit each of them.
    for t in range(0, T, stride):
        counts.append(count_edges_at(positions[:, t], connectivity_radius, device))
    return sum(counts) / len(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--max_trajectories', type=int, default=10,
                        help='Number of test trajectories to sample (default: 10)')
    parser.add_argument('--frame_stride', type=int, default=1,
                        help='Sample every Nth frame within a trajectory (default: 1 = all frames)')
    args = parser.parse_args()

    data_path = args.data_path.rstrip('/') + '/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metadata = reading_utils.read_metadata(data_path, 'rollout')
    base_radius = metadata['default_connectivity_radius']   # 0.015 for Sand
    k10_radius = base_radius * (0.019 / 0.015)              # matches train_baseline_k10.sh

    ds = data_loader.get_data_loader_by_trajectories(path=data_path + 'test.npz')

    base_avgs, k10_avgs = [], []

    with torch.no_grad():
        for i, features in enumerate(ds):
            if i >= args.max_trajectories:
                break
            positions = features[0]  # (n_particles, T, dim)
            base_avgs.append(trajectory_avg_edges(
                positions, base_radius, device, stride=args.frame_stride))
            k10_avgs.append(trajectory_avg_edges(
                positions, k10_radius, device, stride=args.frame_stride))

    avg_base = sum(base_avgs) / len(base_avgs)
    avg_k10 = sum(k10_avgs) / len(k10_avgs)

    print(f"\n{'Model':<20} {'radius':>8}   {'GT-avg edges/step':>20}   {'ratio vs k=5':>14}")
    print("-" * 72)
    print(f"{'k=5 (base)':<20} {base_radius:>8.4f}   {avg_base:>20.1f}   {'1.00x':>14}")
    print(f"{'k=10':<20} {k10_radius:>8.4f}   {avg_k10:>20.1f}   {avg_k10/avg_base:>13.2f}x")
    print()
    print("This is a GT-based, model-independent sanity check.")
    print("For the comparison-table number, run evaluate_rollout_mse.py /")
    print("evaluate_adaptive_rollout.py and read 'mean_edges_per_step' --")
    print("those are averaged over the trained model's actual rollout, which")
    print("is the apples-to-apples value to compare across rows.")


if __name__ == '__main__':
    main()
