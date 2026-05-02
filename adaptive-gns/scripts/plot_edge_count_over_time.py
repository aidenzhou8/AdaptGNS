#!/usr/bin/env python3
"""
Plot edge count and MSE over rollout time for key models only.
Full hyperparameter sweep data is kept in the .npz files for the appendix.

Usage (after rsyncing results from Misha):
  python plot_edge_count_over_time.py
      --models_dir "~/Desktop/S2026 Courses/CPSC 483/AdaptiveGNS/Sand/models"
      --output_dir "~/Desktop/S2026 Courses/CPSC 483/AdaptiveGNS/results_plots"
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Models for the over-time line plots (kept minimal) ──────────────────────
# (label, subdir, npz_stem, color, linestyle)
KEY_MODELS = [
    ('k=5 (fixed)',
     'baseline_k5', 'rollout_mse_model-500000',
     '#2196F3', '-'),

    ('k=10 (fixed)',
     'baseline_k10', 'rollout_mse_model-500000',
     '#4CAF50', '-'),

    ('AdaptGNS fixed-graph',
     'adaptive_gns', 'rollout_mse_model-500000',
     '#FF9800', '--'),

    ('AdaptGNS (ours)',
     'adaptive_gns', 'adaptive_rollout_mse_model-500000_p70_r1.267',
     '#E91E63', '-'),
]

# ── Extra sweep points shown only in the efficiency scatter ──────────────────
# Chosen to trace the Pareto frontier; kept out of the line plots for clarity.
SCATTER_EXTRA = [
    ('pct=70, r=1.5',
     'adaptive_gns', 'adaptive_rollout_mse_model-500000_p70_r1.5',
     '#C2185B'),

    ('pct=80, r=1.267',
     'adaptive_gns', 'adaptive_rollout_mse_model-500000_p80_r1.267',
     '#7B1FA2'),

    ('AdaptGNS NLL fixed r=0.0176',
     'adaptive_gns', 'rollout_mse_model-500000_r0.0176',
     '#FF9800'),
]


def load(models_dir, subdir, stem):
    path = os.path.join(models_dir, subdir, stem + '.npz')
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    return np.load(path)


def smooth(arr, window=5):
    if len(arr) <= window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def plot_over_time(models_dir, output_dir, smooth_window=5, max_step=314):
    """Two-panel: edge count (top) and MSE (bottom) vs rollout step."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
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
    ax_e.set_title('Edge count over rollout', fontsize=13)
    ax_e.legend(fontsize=10)
    ax_e.grid(True, alpha=0.3)

    ax_m.set_xlabel('Rollout step', fontsize=12)
    ax_m.set_ylabel('MSE', fontsize=12)
    ax_m.set_title('Rollout MSE over time', fontsize=13)
    ax_m.set_yscale('log')
    ax_m.legend(fontsize=10)
    ax_m.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    out = os.path.join(output_dir, 'edge_count_and_mse_over_time.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


def plot_efficiency_scatter(models_dir, output_dir):
    """Scatter: mean edges/step (x) vs MSE@step200 (y).

    Shows KEY_MODELS plus extra sweep points to trace the Pareto frontier.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Annotation offsets per label to avoid overlap
    offsets = {
        'k=5 (fixed)':        (-10, 8),
        'k=10 (fixed)':       (6,  -14),
        'AdaptGNS fixed-graph': (6, 5),
        'AdaptGNS (ours)':    (6,  5),
        'pct=70, r=1.5':      (6,   5),
        'pct=80, r=1.267':    (6,   5),
        'AdaptGNS NLL fixed r=0.0176': (6, -14),
    }

    def _plot_point(label, subdir, stem, color, marker='o'):
        d = load(models_dir, subdir, stem)
        if d is None or 'mean_edges_per_step' not in d or 'mse_mean' not in d:
            return
        edges = float(d['mean_edges_per_step'])
        idx = min(199, len(d['mse_mean']) - 1)
        mse = float(d['mse_mean'][idx])
        ax.scatter(edges, mse, color=color, marker=marker, s=110, zorder=4)
        xy_off = offsets.get(label, (6, 5))
        ax.annotate(label, (edges, mse), textcoords='offset points',
                    xytext=xy_off, fontsize=9)

    for label, subdir, stem, color, _ in KEY_MODELS:
        _plot_point(label, subdir, stem, color, marker='o')

    for label, subdir, stem, color in SCATTER_EXTRA:
        _plot_point(label, subdir, stem, color, marker='^')

    ax.set_xlabel('Mean edges / step', fontsize=12)
    ax.set_ylabel('MSE @ step 200', fontsize=12)
    ax.set_title('Efficiency – Accuracy tradeoff', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, 'efficiency_accuracy_scatter.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--smooth_window', type=int, default=5)
    parser.add_argument('--max_step', type=int, default=314)
    args = parser.parse_args()

    models_dir = os.path.expanduser(args.models_dir.rstrip('/'))
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(models_dir), 'results_plots')
    os.makedirs(output_dir, exist_ok=True)

    plot_over_time(models_dir, output_dir, args.smooth_window, args.max_step)
    plot_efficiency_scatter(models_dir, output_dir)
    print("Done.")


if __name__ == '__main__':
    main()
