#!/usr/bin/env python3
"""
Combined Sand + WaterDrop rollout plots.

Default output is a 2x2 ``square'' figure with:
  - columns = datasets (Sand left, WaterDrop right)
  - rows    = metrics  (mean edges/step top, MSE bottom; MSE is log-scaled)
  - one figure-level legend below the four panels

Reads the same .npz files produced by evaluate_rollout_mse.py and
evaluate_adaptive_rollout.py (keys: ``steps``, ``mse_mean``,
``mean_edges_by_step``).

Usage (repo root, after rsyncing ``Sand/models`` and ``WaterDrop/models``):
  python adaptive-gns/scripts/plot_mse_vs_steps_sand_waterdrop.py \\
      --sand_models_dir Sand/models \\
      --wd_models_dir WaterDrop/models \\
      --output_dir combined_results_plots
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Must match filenames on disk -- adjust stems if checkpoints differ.

SAND_CURVES = [
    ("k=5 (fixed)", "baseline_k5", "rollout_mse_model-500000", "#2196F3", "-"),
    ("k=10 (fixed)", "baseline_k10", "rollout_mse_model-500000", "#4CAF50", "-"),
    (
        "AdaptGNS fixed-graph",
        "adaptive_gns",
        "rollout_mse_model-500000",
        "#FF9800",
        "--",
    ),
    (
        "AdaptGNS (ours)",
        "adaptive_gns",
        "adaptive_rollout_mse_model-500000_p70_r1.267",
        "#E91E63",
        "-",
    ),
]

WD_CURVES = [
    ("k=5 (fixed)", "baseline_k5", "rollout_mse_model-500000", "#2196F3", "-"),
    (
        # Some runs keep latest as 400k; fallback tried in load_rollout_npz.
        "k=10 (fixed)",
        "baseline_k10",
        "rollout_mse_model-500000",
        "#4CAF50",
        "-",
    ),
    (
        "AdaptGNS fixed-graph",
        "adaptive_gns",
        "rollout_mse_model-500000",
        "#FF9800",
        "--",
    ),
    (
        "AdaptGNS (ours)",
        "adaptive_gns",
        "adaptive_rollout_mse_model-500000_p70_r1.267",
        "#E91E63",
        "-",
    ),
]

# plot_wd_results.py historically used ``model-400000`` for k=10 WD.
WD_K10_FALLBACK_STEMS = ["rollout_mse_model-400000"]


def load_rollout_npz(
    models_dir: str,
    subdir: str,
    stem: str,
    *,
    fallback_stems: list[str] | None = None,
):
    """Load ``.npz`` for a model; try optional fallback stems."""
    candidates = [stem]
    if fallback_stems:
        candidates.extend(s for s in fallback_stems if s not in candidates)
    for s in candidates:
        path = os.path.join(models_dir, subdir, s + ".npz")
        if os.path.isfile(path):
            data = np.load(path)
            if s != stem:
                print(f"  [note] used {path} instead of '{stem}'")
            return data
    print(f"  [skip] no file for {subdir}/{stem}*.npz (tried fallbacks)")
    return None


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(arr) <= window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def _plot_mse(ax, curves, models_dir, max_step, *, wd_k10_fallback=False):
    for label, subdir, stem, color, ls in curves:
        fbs = (
            WD_K10_FALLBACK_STEMS
            if (wd_k10_fallback and subdir == "baseline_k10")
            else None
        )
        d = load_rollout_npz(models_dir, subdir, stem, fallback_stems=fbs)
        if d is None or "mse_mean" not in d or "steps" not in d:
            continue
        steps = d["steps"][:max_step].astype(float)
        mse = d["mse_mean"][:max_step].astype(float)
        nmin = min(len(steps), len(mse))
        ax.plot(
            steps[:nmin],
            mse[:nmin],
            label=label,
            color=color,
            linestyle=ls,
            linewidth=2,
        )


def _plot_edges(
    ax, curves, models_dir, max_step, *, smooth_window=5, wd_k10_fallback=False
):
    for label, subdir, stem, color, ls in curves:
        fbs = (
            WD_K10_FALLBACK_STEMS
            if (wd_k10_fallback and subdir == "baseline_k10")
            else None
        )
        d = load_rollout_npz(models_dir, subdir, stem, fallback_stems=fbs)
        if d is None or "mean_edges_by_step" not in d or "steps" not in d:
            continue
        steps = d["steps"][:max_step].astype(float)
        ec = d["mean_edges_by_step"][:max_step].astype(float)
        ec_s = _smooth(ec, smooth_window)
        offset = len(steps) - len(ec_s)
        ax.plot(
            steps[offset:],
            ec_s,
            label=label,
            color=color,
            linestyle=ls,
            linewidth=2,
        )


def plot_combined_square(
    sand_dir: str,
    wd_dir: str,
    out_path: str,
    *,
    sand_max_step: int,
    wd_max_step: int,
    smooth_window: int = 5,
):
    """2x2 grid: rows = (edges, MSE), cols = (Sand, WaterDrop)."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
    (ax_edges_sand, ax_edges_wd), (ax_mse_sand, ax_mse_wd) = axes

    # Top row: edge counts (linear)
    _plot_edges(
        ax_edges_sand, SAND_CURVES, sand_dir, sand_max_step,
        smooth_window=smooth_window, wd_k10_fallback=False,
    )
    _plot_edges(
        ax_edges_wd, WD_CURVES, wd_dir, wd_max_step,
        smooth_window=smooth_window, wd_k10_fallback=True,
    )

    # Bottom row: MSE (log)
    _plot_mse(
        ax_mse_sand, SAND_CURVES, sand_dir, sand_max_step,
        wd_k10_fallback=False,
    )
    _plot_mse(
        ax_mse_wd, WD_CURVES, wd_dir, wd_max_step, wd_k10_fallback=True,
    )

    # Column titles
    ax_edges_sand.set_title("Sand", fontsize=14)
    ax_edges_wd.set_title("WaterDrop", fontsize=14)

    # Row labels via y-axis on the leftmost column only
    ax_edges_sand.set_ylabel("Mean edges / step", fontsize=12)
    ax_mse_sand.set_ylabel("MSE", fontsize=12)

    # X labels only on the bottom row
    for ax in (ax_mse_sand, ax_mse_wd):
        ax.set_xlabel("Rollout step", fontsize=12)

    # Log scale + grid for MSE; light grid for edges
    for ax in (ax_mse_sand, ax_mse_wd):
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
    for ax in (ax_edges_sand, ax_edges_wd):
        ax.grid(True, alpha=0.3)

    # Reserve space at the bottom for a single shared legend, then place it
    # below the bottom-row x-axis labels (no overlap).
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    handles, labels = ax_mse_sand.get_legend_handles_labels()
    if not handles:
        handles, labels = ax_edges_sand.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels) if labels else 1,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved \u2192 {out_path}")


def plot_combined_row(
    sand_dir: str,
    wd_dir: str,
    out_path: str,
    *,
    sand_max_step: int,
    wd_max_step: int,
    smooth_window: int = 5,
):
    """1x4 row: [Sand edges | Sand MSE | WD edges | WD MSE] with a divider."""
    fig, axes = plt.subplots(
        1, 5, figsize=(16, 3.6),
        gridspec_kw={"width_ratios": [1, 1, 0.04, 1, 1]},
    )
    ax_edges_sand, ax_mse_sand, ax_div, ax_edges_wd, ax_mse_wd = axes

    ax_div.set_visible(False)

    _plot_edges(
        ax_edges_sand, SAND_CURVES, sand_dir, sand_max_step,
        smooth_window=smooth_window, wd_k10_fallback=False,
    )
    _plot_mse(
        ax_mse_sand, SAND_CURVES, sand_dir, sand_max_step, wd_k10_fallback=False,
    )
    _plot_edges(
        ax_edges_wd, WD_CURVES, wd_dir, wd_max_step,
        smooth_window=smooth_window, wd_k10_fallback=True,
    )
    _plot_mse(
        ax_mse_wd, WD_CURVES, wd_dir, wd_max_step, wd_k10_fallback=True,
    )

    ax_edges_sand.set_title("Sand \u2014 Edges/step", fontsize=11)
    ax_mse_sand.set_title("Sand \u2014 MSE", fontsize=11)
    ax_edges_wd.set_title("WaterDrop \u2014 Edges/step", fontsize=11)
    ax_mse_wd.set_title("WaterDrop \u2014 MSE", fontsize=11)

    for ax in (ax_edges_sand, ax_edges_wd):
        ax.set_ylabel("Mean edges / step", fontsize=10)
        ax.grid(True, alpha=0.3)
    for ax in (ax_mse_sand, ax_mse_wd):
        ax.set_ylabel("MSE", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
    for ax in (ax_edges_sand, ax_mse_sand, ax_edges_wd, ax_mse_wd):
        ax.set_xlabel("Rollout step", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)

    # Reserve space at the bottom for a single shared legend.
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    # Draw a vertical divider line through the (invisible) spacer column.
    div_bbox = ax_div.get_position()
    x_div = 0.5 * (div_bbox.x0 + div_bbox.x1)
    fig.add_artist(Line2D(
        [x_div, x_div], [0.12, 0.95],
        transform=fig.transFigure,
        color="0.55", linewidth=1.0, linestyle="-",
    ))

    handles, labels = ax_mse_sand.get_legend_handles_labels()
    if not handles:
        handles, labels = ax_edges_sand.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(labels) if labels else 1,
        fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved \u2192 {out_path}")


def plot_mse_only_side_by_side(
    sand_dir: str,
    wd_dir: str,
    out_path: str,
    *,
    sand_max_step: int,
    wd_max_step: int,
    log_y: bool,
):
    """Original 1x2 MSE-only side-by-side figure (kept for compatibility)."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), constrained_layout=True)
    for ax, curves, models_dir, max_step, title, wd_fb in (
        (axes[0], SAND_CURVES, sand_dir, sand_max_step,
         "Sand \u2014 MSE vs rollout step", False),
        (axes[1], WD_CURVES, wd_dir, wd_max_step,
         "WaterDrop \u2014 MSE vs rollout step", True),
    ):
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Rollout step", fontsize=12)
        ax.set_ylabel("MSE (mean over test trajectories)", fontsize=11)
        if log_y:
            ax.set_yscale("log")
        _plot_mse(ax, curves, models_dir, max_step, wd_k10_fallback=wd_fb)
        ax.grid(True, alpha=0.3, which=("both" if log_y else "major"))
        ax.legend(fontsize=9, loc="upper left")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved \u2192 {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sand_models_dir",
        default="Sand/models",
        help="Directory containing Sand baseline_k5, baseline_k10, adaptive_gns, ...",
    )
    p.add_argument(
        "--wd_models_dir",
        default="WaterDrop/models",
        help="Directory containing WaterDrop model subdirectories",
    )
    p.add_argument(
        "--output_dir",
        default="combined_results_plots",
        help="Where to write PNG outputs",
    )
    p.add_argument("--sand_max_step", type=int, default=314)
    p.add_argument("--wd_max_step", type=int, default=994)
    p.add_argument("--smooth_window", type=int, default=5)
    p.add_argument(
        "--mse_only",
        action="store_true",
        help="Only emit the original 1x2 MSE side-by-side figure (no edges row)",
    )
    p.add_argument(
        "--square",
        action="store_true",
        help="Use the older 2x2 square layout instead of the default 1x4 row.",
    )
    p.add_argument(
        "--linear_y",
        action="store_true",
        help="Use linear Y axis for the MSE-only figure (default: log scale)",
    )
    p.add_argument(
        "--also_split",
        action="store_true",
        help="Also write per-dataset MSE-only PNGs (mse_vs_steps_{sand,waterdrop}.png)",
    )
    args = p.parse_args()

    sand_dir = os.path.expanduser(args.sand_models_dir.rstrip(os.sep))
    wd_dir = os.path.expanduser(args.wd_models_dir.rstrip(os.sep))
    out_dir = os.path.expanduser(args.output_dir.rstrip(os.sep))
    os.makedirs(out_dir, exist_ok=True)

    if not args.mse_only:
        out_combined = os.path.join(out_dir, "edges_and_mse_sand_waterdrop.png")
        if args.square:
            plot_combined_square(
                sand_dir,
                wd_dir,
                out_combined,
                sand_max_step=args.sand_max_step,
                wd_max_step=args.wd_max_step,
                smooth_window=args.smooth_window,
            )
        else:
            plot_combined_row(
                sand_dir,
                wd_dir,
                out_combined,
                sand_max_step=args.sand_max_step,
                wd_max_step=args.wd_max_step,
                smooth_window=args.smooth_window,
            )

    if args.mse_only or args.also_split:
        plot_mse_only_side_by_side(
            sand_dir,
            wd_dir,
            os.path.join(out_dir, "mse_vs_rollout_steps_sand_waterdrop.png"),
            sand_max_step=args.sand_max_step,
            wd_max_step=args.wd_max_step,
            log_y=not args.linear_y,
        )

    if args.also_split:
        log_y = not args.linear_y
        for fname, curves, md, mx, ttl, wd_fb in (
            ("mse_vs_steps_sand.png", SAND_CURVES, sand_dir,
             args.sand_max_step, "Sand", False),
            ("mse_vs_steps_waterdrop.png", WD_CURVES, wd_dir,
             args.wd_max_step, "WaterDrop", True),
        ):
            fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
            ax.set_title(f"{ttl} \u2014 MSE vs rollout step", fontsize=13)
            ax.set_xlabel("Rollout step", fontsize=12)
            ax.set_ylabel("MSE (mean over test trajectories)", fontsize=11)
            if log_y:
                ax.set_yscale("log")
            _plot_mse(ax, curves, md, mx, wd_k10_fallback=wd_fb)
            ax.grid(True, alpha=0.3, which=("both" if log_y else "major"))
            ax.legend(fontsize=9, loc="upper left")
            sp = os.path.join(out_dir, fname)
            fig.savefig(sp, dpi=150)
            plt.close(fig)
            print(f"Saved \u2192 {sp}")
    print("Done.")


if __name__ == "__main__":
    main()
