#!/usr/bin/env python3
"""
plot_timings.py  --  Visualise per-frame pipeline timing from timing_log.csv.

Usage:
  python3 scripts/plot_timings.py --csv timing_log.csv --output timing_plots.png
"""
# Reads a timing_log.csv (one row per processed frame, columns = pipeline
# stage durations in ms) and renders a multi-panel PNG: stacked per-stage
# timing, callback-vs-end-to-end latency, box plots, and a summary stats
# table. Run standalone via `python3 scripts/plot_timings.py --csv ... --output ...`.
import argparse
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


STAGES = ['pre_ms', 'seg_ms', 'depth_ms', 'post_ms', 'pub_ms']  # csv column names for each pipeline stage's duration (ms)
STAGE_LABELS = ['Preprocess', 'SceneSeg', 'Scene3D depth', 'Post/CSV', 'Publish']  # human-readable names matching STAGES order
STAGE_COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']  # plot colors matching STAGES order


def load(csv_path):
    rows = {k: [] for k in ['frame'] + STAGES + ['total_ms', 'e2e_ms']}  # total_ms = full callback wall time, e2e_ms = capture-to-publish latency
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            for k in rows:
                rows[k].append(float(row[k]))
    return {k: np.array(v) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',    default='timing_log.csv')  # path to the input timing log
    ap.add_argument('--output', default='timing_plots.png')  # path to write the combined figure
    args = ap.parse_args()

    d = load(args.csv)
    frames = d['frame']
    n = len(frames)  # total frame count, used in titles/labels

    fig = plt.figure(figsize=(14, 16))
    fig.suptitle(f'Pipeline timing  ({n} frames)', fontsize=13, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Stacked area: per-stage contribution over time ──────────────────
    ax1 = fig.add_subplot(gs[0, :])
    bottom = np.zeros(n)  # running stack height for the stacked area fill
    for key, label, color in zip(STAGES, STAGE_LABELS, STAGE_COLORS):
        vals = d[key]
        ax1.fill_between(frames, bottom, bottom + vals, label=label,
                         color=color, alpha=0.85, step='mid')
        bottom += vals
    ax1.plot(frames, d['total_ms'], color='black', linewidth=0.8,
             linestyle='--', label='Total (wall)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_xlabel('Frame')
    ax1.set_title('Per-stage breakdown over time (stacked)')
    ax1.legend(loc='upper right', fontsize=8, ncol=3)
    ax1.set_xlim(frames[0], frames[-1])

    # ── 2. Total callback vs true end-to-end latency ───────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(frames, d['total_ms'], color='steelblue', linewidth=0.9,
             label='total (callback wall time)')
    ax2.plot(frames, d['e2e_ms'],   color='tomato',    linewidth=0.9,
             label='e2e (capture → publish)')
    ax2.axhline(np.mean(d['total_ms']), color='steelblue', linewidth=0.7,
                linestyle=':', alpha=0.7)
    ax2.axhline(np.mean(d['e2e_ms']),   color='tomato',    linewidth=0.7,
                linestyle=':', alpha=0.7)
    ax2.set_ylabel('Time (ms)')
    ax2.set_xlabel('Frame')
    ax2.set_title('Total callback time vs true end-to-end latency')
    ax2.legend(fontsize=9)
    ax2.set_xlim(frames[0], frames[-1])

    # ── 3. Box plots of each stage ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    box_data = [d[k] for k in STAGES]
    bp = ax3.boxplot(box_data, patch_artist=True, widths=0.5,
                     medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], STAGE_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax3.set_xticklabels(STAGE_LABELS, fontsize=8, rotation=15, ha='right')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Stage distribution (box plot)')

    # ── 4. Summary stats table ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')

    all_keys = STAGES + ['total_ms', 'e2e_ms']  # data columns to summarize in the stats table
    all_labels = STAGE_LABELS + ['Total', 'End-to-end']  # matching row labels for the table
    col_labels = ['Stage', 'Mean', 'p50', 'p95', 'Max']  # stats table header
    rows_data = []
    for key, label in zip(all_keys, all_labels):
        v = d[key]
        rows_data.append([
            label,
            f'{np.mean(v):.1f}',
            f'{np.percentile(v, 50):.1f}',
            f'{np.percentile(v, 95):.1f}',
            f'{np.max(v):.1f}',
        ])

    tbl = ax4.table(cellText=rows_data, colLabels=col_labels,
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 1.55)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2d2d2d')
            cell.set_text_props(color='white', fontweight='bold')
        elif r <= len(STAGES):
            cell.set_facecolor(STAGE_COLORS[r - 1] + '33')
        elif r == len(STAGES) + 1:
            cell.set_facecolor('#dddddd')
        else:
            cell.set_facecolor('#f5ddd5')

    ax4.set_title('Summary statistics (ms)', pad=12)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved → {args.output}  ({n} frames)')
    print(f'  total:  mean={np.mean(d["total_ms"]):.1f}  '
          f'p95={np.percentile(d["total_ms"], 95):.1f}  max={np.max(d["total_ms"]):.1f} ms')
    print(f'  e2e:    mean={np.mean(d["e2e_ms"]):.1f}  '
          f'p95={np.percentile(d["e2e_ms"], 95):.1f}  max={np.max(d["e2e_ms"]):.1f} ms')
    for key, label in zip(STAGES, STAGE_LABELS):
        print(f'  {label:<20} mean={np.mean(d[key]):.1f}  p95={np.percentile(d[key], 95):.1f} ms')


if __name__ == '__main__':
    main()
