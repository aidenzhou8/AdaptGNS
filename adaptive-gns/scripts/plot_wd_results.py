#!/usr/bin/env python3
"""
Plot edge count and MSE over rollout time for WaterDrop models.

Usage (from repo root):
  python adaptive-gns/scripts/plot_wd_results.py \
      --models_dir WaterDrop/models \
      --output_dir wd_results_plots
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# (label, subdir, npz_stem, color, linestyle)
KEY_MODELS = [
    ('k=5 (fixed)',
     'baseline_k5', 'rollout_mse_model-500000',
     '#2196F3', '-'),

    ('k=10 (fixed)',
     'baseline_k10', 'rollout_mse_model-400000',
     '#4CAF50', '-'),

    ('AdaptGNS fixed-graph',
     'adaptive_gns', 'rollout_mse_model-500000',
     '#FF9800', '--'),

    ('AdaptGNS (ours)',
     'adaptive_gns', 'adaptive_rollout_mse_model-500000_p70_r1.267',
     '#E91E63', '-'),
]


def load(models_dir, subdir, stem):
    path = os.path.join(models_dir, subdir, stem + '.npz')
    if not os.path.exists(path):
        print(f'  [skip] {path} not found')
        return None
    return np.load(path)


def smooth(arr, window=15):
    if len(arr) <= window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def plot_over_time(models_dir, output_dir, smooth_window=15, max_step=994):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_e, ax_m = axes

    for label, subdir, stem, color, ls in KEY_MODELS:
        d = load(models_dir, subdir, stem)
        if d is None:
            continue
        steps = d['steps'][:max_step]

        if 'mean_edges_by_step' in d:
            ec = d['mean_edges_by_step'][:max_step].astype(float)
            ec_s = smooth(ec, smooth_window)
            off = len(steps) - len(ec_s)
            ax_e.plot(steps[off:], ec_s, label=label, color=color,
                      linestyle=ls, linewidth=2)

        if 'mse_mean' in d:
            mse = d['mse_mean'][:max_step]
            ax_m.plot(steps, mse, label=label, color=color,
                      linestyle=ls, linewidth=2)

    ax_e.set_ylabel('Mean edges / step', fontsize=12)
    ax_e.set_title('WaterDrop — Edge count over rollout', fontsize=13)
    ax_e.legend(fontsize=10)
    ax_e.grid(True, alpha=0.3)

    ax_m.set_xlabel('Rollout step', fontsize=12)
    ax_m.set_ylabel('MSE', fontsize=12)
    ax_m.set_title('WaterDrop — Rollout MSE over time', fontsize=13)
    ax_m.set_yscale('log')
    ax_m.legend(fontsize=10)
    ax_m.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    out = os.path.join(output_dir, 'wd_edge_count_and_mse_over_time.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved → {out}')


def plot_efficiency_scatter(models_dir, output_dir, mse_step=200):
    fig, ax = plt.subplots(figsize=(7, 5))

    offsets = {
        'k=5 (fixed)':          (-10, 8),
        'k=10 (fixed)':         (6,  -14),
        'AdaptGNS fixed-graph': (6,   5),
        'AdaptGNS (ours)':      (6,   5),
    }

    for label, subdir, stem, color, _ in KEY_MODELS:
        d = load(models_dir, subdir, stem)
        if d is None or 'mean_edges_per_step' not in d or 'mse_mean' not in d:
            continue
        edges = float(d['mean_edges_per_step'])
        idx = min(mse_step - 1, len(d['mse_mean']) - 1)
        mse = float(d['mse_mean'][idx])
        ax.scatter(edges, mse, color=color, marker='o', s=110, zorder=4)
        xy_off = offsets.get(label, (6, 5))
        ax.annotate(label, (edges, mse), textcoords='offset points',
                    xytext=xy_off, fontsize=9)

    ax.set_xlabel('Mean edges / step', fontsize=12)
    ax.set_ylabel(f'MSE @ step {mse_step}', fontsize=12)
    ax.set_title('WaterDrop — Efficiency–Accuracy tradeoff', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, 'wd_efficiency_accuracy_scatter.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved → {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--smooth_window', type=int, default=15)
    parser.add_argument('--max_step', type=int, default=994)
    parser.add_argument('--mse_step', type=int, default=200)
    args = parser.parse_args()

    models_dir = os.path.expanduser(args.models_dir.rstrip('/'))
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(models_dir), 'results_plots')
    os.makedirs(output_dir, exist_ok=True)

    plot_over_time(models_dir, output_dir, args.smooth_window, args.max_step)
    plot_efficiency_scatter(models_dir, output_dir, args.mse_step)
    print('Done.')


if __name__ == '__main__':
    main()
