#!/usr/bin/env python3
"""
plot_pipeline_log.py
Plots the per-frame pipeline log (bin_number, linear_velocity, angular_velocity, command)
written by webcam_navigator_node.py / CommandLogger.

Usage:
  /data/archit0030/miniforge3/envs/vp_gpu/bin/python3 scripts/plot_pipeline_log.py \
      --csv logs/pipeline_log.csv --output logs/pipeline_log_plots.png
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_log(csv_path):
    bins, linear, angular, command = [], [], [], []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            bins.append(int(row['bin_number']))
            linear.append(float(row['linear_velocity']))
            angular.append(float(row['angular_velocity']))
            command.append(row['command'])
    return (np.array(bins), np.array(linear), np.array(angular), np.array(command))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='logs/pipeline_log.csv')
    ap.add_argument('--output', default='logs/pipeline_log_plots.png')
    args = ap.parse_args()

    bins, linear, angular, command = load_log(args.csv)
    frame = np.arange(len(bins))
    brake = command == 'BRAKE'

    fig = plt.figure(figsize=(12, 17))
    axes = [fig.add_subplot(5, 1, i + 1) for i in range(3)]

    # 1. Linear velocity over time, BRAKE frames shaded.
    ax = axes[0]
    ax.plot(frame, linear, color='tab:blue', linewidth=0.8)
    ax.fill_between(frame, 0, linear.max() if linear.max() > 0 else 1.0,
                     where=brake, color='red', alpha=0.12, step='mid')
    ax.set_ylabel('linear velocity (m/s)')
    ax.set_title('Linear velocity over time (red = BRAKE)')

    # 2. Angular velocity over time, BRAKE frames shaded.
    ax = axes[1]
    ax.plot(frame, angular, color='tab:orange', linewidth=0.8)
    ylim = max(abs(angular.min()), abs(angular.max()), 1e-3)
    ax.fill_between(frame, -ylim, ylim, where=brake, color='red', alpha=0.12, step='mid')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel('angular velocity (rad/s)')
    ax.set_title('Angular velocity over time (red = BRAKE)')

    # 3. Chosen bin number over time.
    ax = axes[2]
    ax.step(frame, bins, color='tab:green', linewidth=0.8, where='mid')
    ax.set_ylabel('bin number')
    ax.set_xlabel('frame')
    ax.set_title('Chosen steering bin over time')

    # 4. Command distribution + bin histogram side by side.
    ax_cmd = fig.add_subplot(5, 2, 7)
    counts = [int(np.sum(command == 'CLEAR')), int(np.sum(brake))]
    ax_cmd.bar(['CLEAR', 'BRAKE'], counts, color=['tab:green', 'tab:red'])
    for i, c in enumerate(counts):
        ax_cmd.text(i, c, str(c), ha='center', va='bottom')
    ax_cmd.set_title('Command distribution')

    ax_bin = fig.add_subplot(5, 2, 8)
    ax_bin.hist(bins, bins=np.arange(bins.min(), bins.max() + 2) - 0.5, color='tab:green', rwidth=0.8)
    ax_bin.set_title('Chosen bin distribution')
    ax_bin.set_xlabel('bin number')

    # 5. Bin vs linear velocity, and bin vs angular velocity (scatter + per-bin mean).
    bin_vals = np.unique(bins)
    rng = np.random.default_rng(0)
    jitter = (rng.random(len(bins)) - 0.5) * 0.3

    ax_lin = fig.add_subplot(5, 2, 9)
    ax_lin.scatter(bins + jitter, linear, s=8, alpha=0.25, color='tab:blue')
    means_lin = [linear[bins == b].mean() for b in bin_vals]
    ax_lin.plot(bin_vals, means_lin, color='black', marker='o', linewidth=1.5, label='mean')
    ax_lin.set_xlabel('bin number')
    ax_lin.set_ylabel('linear velocity (m/s)')
    ax_lin.set_title('Bin vs linear velocity')
    ax_lin.legend()

    ax_ang = fig.add_subplot(5, 2, 10)
    ax_ang.scatter(bins + jitter, angular, s=8, alpha=0.25, color='tab:orange')
    means_ang = [angular[bins == b].mean() for b in bin_vals]
    ax_ang.plot(bin_vals, means_ang, color='black', marker='o', linewidth=1.5, label='mean')
    ax_ang.axhline(0, color='gray', linewidth=0.5)
    ax_ang.set_xlabel('bin number')
    ax_ang.set_ylabel('angular velocity (rad/s)')
    ax_ang.set_title('Bin vs angular velocity')
    ax_ang.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f'Saved plot to {args.output}  ({len(bins)} frames, {int(np.sum(brake))} BRAKE / {len(bins) - int(np.sum(brake))} CLEAR)')


if __name__ == '__main__':
    main()
