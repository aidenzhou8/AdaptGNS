#!/usr/bin/env python3
"""
Restyle sigma_traj* calibration plots for the paper.

The source PNGs are produced by `verify_calibration.py` at fig=(6,5), dpi=150
giving a fixed 900×750 layout:

    rows 0–41    : title
    rows 42–47   : whitespace above axes
    row  48      : axes top border
    rows 49–669  : main axes interior
    row  670     : axes bottom border
    rows 671–676 : x-tick MARKS extending below border
    rows 684–699 : x-tick NUMBERS  (we KEEP these)
    rows 711+    : x-axis "x" label  (we CUT this)

    cols 27–42   : y-axis "y" label  (we CUT this)
    cols 53–82   : y-tick NUMBERS    (we KEEP these)
    col  98      : axes left border
    cols 98–757  : main axes interior
    cols 770–791 : colorbar strip
    cols 807–836 : colorbar tick NUMBERS  (we KEEP these)
    cols 848–862 : "σ_i" axis label   (we REPLACE with "Variance (σ²)")

Outputs PDFs with title/x/y labels removed and the colorbar relabelled.

Usage:
    python scripts/make_sigma_strip.py --traj 0 --steps 1 10 50
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def detect_layout(arr: np.ndarray):
    """
    Return a dict with key pixel coordinates of axes / colorbar / tick text.
    All coords are in the original image's pixel space.
    """
    rgb = arr[:, :, :3]
    H, W = arr.shape[:2]

    def rd_full(r): return int((rgb[r].min(axis=1) < 240).sum())
    def cd_full(c): return int((rgb[:, c].min(axis=1) < 240).sum())

    # ── 1. axes box: dense horizontal/vertical lines (density > 600) ───────
    axes_top = next(r for r in range(20, 200) if rd_full(r) > 600)
    axes_bot = next(r for r in range(H - 30, axes_top + 100, -1) if rd_full(r) > 600)
    axes_lt  = next(c for c in range(20, 200) if cd_full(c) > 600)
    # axes right border (between main axes and the colorbar gap)
    axes_rt = axes_lt + next(
        i for i in range(50, W - axes_lt)
        if int((rgb[axes_top:axes_bot, axes_lt + i].min(axis=1) < 240).sum()) > 600
    )

    # Restrict column scans to axes-interior rows; row scans to axes-interior cols.
    def cd(c): return int((rgb[axes_top:axes_bot, c].min(axis=1) < 240).sum())
    def rd(r): return int((rgb[r, axes_lt:axes_rt].min(axis=1) < 240).sum())

    # ── 1b. title bottom row ────────────────────────────────────────────────
    # The title sits ABOVE the axes box (rows < axes_top) and spans the axes-
    # interior columns. Scan upward from axes_top - 1; last row with density>5
    # in axes-interior cols is the title bottom.
    title_bot = -1   # -1 means no title found
    for r in range(axes_top - 1, -1, -1):
        if rd(r) > 5:
            title_bot = r
            break

    # ── 2. end-of-xtick-numbers row ─────────────────────────────────────────
    # Below axes_bot, restricted to axes-interior cols (so colorbar tick MARKS
    # don't pollute density). Tick MARKS in axes interior have density ~15-35;
    # tick NUMBERS have density > 50.
    state = 'below_axes'
    xtick_end = axes_bot + 30  # safe fallback
    for r in range(axes_bot + 5, min(H, axes_bot + 60)):
        d = rd(r)
        if state == 'below_axes' and d > 50:
            state = 'in_ticktext'
        elif state == 'in_ticktext' and d == 0:
            xtick_end = r - 1
            break

    # ── 3. start-of-ytick-numbers col ───────────────────────────────────────
    # Going LEFT from axes_lt, skip tick MARKS (low density), enter tick
    # NUMBER region. Tick number TEXT has small internal gaps between digits
    # ("0", ".", "X") so we must require a SUSTAINED gap (5+ consecutive zero
    # cols) before declaring the boundary between tick numbers and y-label.
    ytick_start = axes_lt - 50  # fallback
    GAP_NEEDED = 5
    state = 'tick_marks'        # immediately L of axes border = tick marks
    zero_run = 0
    for c in range(axes_lt - 1, 10, -1):
        d = cd(c)
        if state == 'tick_marks':
            if d > 60:           # only true tick NUMBER cols hit this
                state = 'in_ticknums'
                zero_run = 0
        elif state == 'in_ticknums':
            if d == 0:
                zero_run += 1
                if zero_run >= GAP_NEEDED:
                    # the run of zeros begins at c+1, ends at c+GAP_NEEDED
                    ytick_start = c + 1
                    break
            else:
                zero_run = 0

    # ── 4. colorbar label region (rightmost sparse content; σ_i text) ───────
    # Scan R→L from W-1: skip right margin (zeros), enter label region, exit
    # at first gap. The label lives at cols ~848-862 in the reference layout.
    cb_label_lt = cb_label_rt = None
    state = 'right_margin'
    for c in range(W - 1, max(0, W - 80), -1):
        d = cd(c)
        if state == 'right_margin':
            if d > 0:
                state = 'in_label'
                cb_label_rt = c
                cb_label_lt = c
        elif state == 'in_label':
            if d == 0:
                break
            cb_label_lt = c

    return {
        'axes_top':     axes_top,
        'axes_bot':     axes_bot,
        'axes_lt':      axes_lt,
        'axes_rt':      axes_rt,
        'title_bot':    title_bot,
        'xtick_end':    xtick_end,
        'ytick_start':  ytick_start,
        'cb_label_lt':  cb_label_lt,
        'cb_label_rt':  cb_label_rt,
        'H': H, 'W': W,
    }


def get_font(size: int):
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def make_rotated_label(text: str, font_size: int = 18) -> Image.Image:
    """Return a transparent RGBA PIL image of `text` rotated 90° (reads bottom→top)."""
    font = get_font(font_size)

    # measure
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = int(font_size * len(text) * 0.55), font_size + 4

    pad_x, pad_y = 4, 3
    canvas = Image.new("RGBA", (tw + 2 * pad_x, th + 2 * pad_y), (255, 255, 255, 0))
    ImageDraw.Draw(canvas).text((pad_x, pad_y), text, fill=(0, 0, 0, 255), font=font)
    return canvas.rotate(90, expand=True, resample=Image.BICUBIC)


def restyle_png_to_array(png_path: str, cb_label: str = "Variance (\u03c3\u00b2)"):
    """
    Load a sigma plot PNG, strip the title and x/y axis labels, white-out the
    old colorbar label, draw a new colorbar label, return RGBA numpy array.

    The keep-region (after cropping) is:
        rows  : axes_top - 6  ..  xtick_end + 2
        cols  : ytick_start   ..  W
    """
    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img).copy()
    L = detect_layout(arr)

    # ── erase the TITLE in-place (rows above the axes box, axes-interior cols) ──
    # The title text vertically overlaps with the topmost y-tick label "0.9",
    # so we cannot just crop above it without also clipping the tick. Instead,
    # white-out the title region (cols axes_lt .. axes_rt, rows 0 .. title_bot)
    # before cropping. The y-tick text lives in cols < axes_lt, so it survives.
    if L['title_bot'] >= 0:
        arr[: L['title_bot'] + 1, L['axes_lt']: L['axes_rt'] + 1] = [255, 255, 255, 255]

    # Padding choices (measured against actual tick-text extents):
    #   y-tick text "0.9" extends ≈ axes_top − 7 to axes_top + 8
    #   y-tick text "0.1" extends ≈ axes_bot − 8 to axes_bot + 7
    #   x-tick numbers end at L['xtick_end']; xlabel begins ≈ +12 rows after
    # We use:
    #   crop_top    = axes_top − 14   (≈ 7 px whitespace above topmost label)
    #   crop_bottom = xtick_end + 10  (≈ 10 px whitespace below x-tick numbers,
    #                                  still well clear of the xlabel)
    crop_top    = max(0, L['axes_top'] - 14)
    crop_bottom = min(L['H'], L['xtick_end'] + 10)
    crop_left   = max(0, L['ytick_start'])
    crop_right  = L['W']

    cropped = arr[crop_top:crop_bottom, crop_left:crop_right].copy()

    # ── white-out old colorbar label only ───────────────────────────────────
    if L['cb_label_lt'] is not None and L['cb_label_rt'] is not None:
        c0 = max(0, L['cb_label_lt'] - crop_left - 4)
        c1 = min(cropped.shape[1], L['cb_label_rt'] - crop_left + 5)
        cropped[:, c0:c1] = [255, 255, 255, 255]
        cb_zone_center = (c0 + c1) // 2
    else:
        cb_zone_center = cropped.shape[1] - 12

    # ── render new label ────────────────────────────────────────────────────
    pil_img = Image.fromarray(cropped, mode="RGBA")
    rotated = make_rotated_label(cb_label, font_size=18)
    paste_col = cb_zone_center - rotated.width // 2
    paste_row = max(0, (pil_img.height - rotated.height) // 2)
    pil_img.alpha_composite(rotated, dest=(paste_col, paste_row))

    return np.array(pil_img)


def save_pdf(arr: np.ndarray, out_path: str, dpi: int = 150):
    h, w = arr.shape[:2]
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(arr, interpolation='lanczos')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, format='pdf', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_strip(arrays, out_path: str, dpi: int = 150, gap_px: int = 20):
    """Concatenate equal-height RGBA arrays into one horizontal PDF strip."""
    target_h = max(a.shape[0] for a in arrays)
    norm = []
    for a in arrays:
        if a.shape[0] != target_h:
            new_w = int(a.shape[1] * target_h / a.shape[0])
            pil = Image.fromarray(a, "RGBA").resize((new_w, target_h), Image.LANCZOS)
            norm.append(np.array(pil))
        else:
            norm.append(a)
    gap = np.full((target_h, gap_px, 4), 255, dtype=np.uint8)
    parts = []
    for i, r in enumerate(norm):
        parts.append(r)
        if i < len(norm) - 1:
            parts.append(gap)
    save_pdf(np.concatenate(parts, axis=1), out_path, dpi=dpi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir',  default='../calibration_plots/calibration')
    p.add_argument('--output_dir', default='../calibration_plots/paper')
    p.add_argument('--traj',  type=int, default=0)
    p.add_argument('--steps', nargs='+', type=int, default=[1, 10, 50])
    p.add_argument('--dpi',   type=int, default=150)
    p.add_argument('--cb_label', default='Variance (\u03c3\u00b2)')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    arrays = []
    for s in args.steps:
        src = os.path.join(args.input_dir, f'sigma_traj{args.traj}_step{s}.png')
        if not os.path.isfile(src):
            print(f"WARNING: missing {src}")
            continue
        arr = restyle_png_to_array(src, cb_label=args.cb_label)
        save_pdf(arr, os.path.join(args.output_dir, f'sigma_traj{args.traj}_step{s}.pdf'),
                 dpi=args.dpi)
        arrays.append(arr)

    if len(arrays) >= 2:
        steps_str = '_'.join(str(s) for s in args.steps)
        make_strip(arrays,
                   os.path.join(args.output_dir,
                                f'sigma_traj{args.traj}_strip_{steps_str}.pdf'),
                   dpi=args.dpi)


if __name__ == '__main__':
    main()
