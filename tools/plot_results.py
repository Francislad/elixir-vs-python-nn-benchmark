#!/usr/bin/env python3
"""Draws the report PNGs. Usage: plot_results.py [-o dir] [--baremetal dir|none] [batch_dir ...]"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGS = {"Elixir": "#6e4a9e", "Python": "#e0a526"}
DATASETS = [("train_mnist", "MNIST"), ("train_cifar10", "CIFAR-10")]
TIME_RE = re.compile(r"Training time[^:]*:\s*([\d.]+)\s*seconds")
BAR_W = 0.36


def collect(base, lang, script):
    found = []
    pattern = os.path.join(base, lang.lower(), script, "run_*_output.log")
    for log in sorted(glob.glob(pattern)):
        with open(log, errors="ignore") as fh:
            m = TIME_RE.search(fh.read())
        stats = log.replace("_output.log", "_stats.csv")
        if m and os.path.exists(stats):
            found.append((float(m.group(1)), stats))
    return found


def pooled(dirs, lang, script):
    runs = []
    for d in dirs:
        runs += collect(d, lang, script)
    return runs


def bar_pair(ax, i, j):
    return i + (j - 0.5) * BAR_W


def main():
    ap = argparse.ArgumentParser(description="Draw benchmark graphs from run logs and stats CSVs.")
    ap.add_argument("dirs", nargs="*", help="Batch folders holding elixir/ and python/ (default: results/docker/batch*)")
    ap.add_argument("-o", "--output", default=os.path.join(PROJECT, "results", "plots"), help="Output folder for the PNGs (default: results/plots)")
    ap.add_argument("--baremetal", default=os.path.join(PROJECT, "results", "baremetal"), help="Bare metal folder with elixir/ and python/, or 'none' (default: results/baremetal)")
    args = ap.parse_args()

    dirs = args.dirs or sorted(glob.glob(os.path.join(PROJECT, "results", "docker", "batch*")))
    if not dirs:
        sys.exit("No input directories found.")
    bare = None if args.baremetal == "none" or not os.path.isdir(args.baremetal) else args.baremetal
    os.makedirs(args.output, exist_ok=True)

    runs = {(l, s): pooled(dirs, l, s) for l in LANGS for s, _ in DATASETS}
    empty = [k for k, v in runs.items() if not v]
    if empty:
        sys.exit(f"No runs found for {empty}; check the input directories.")
    times = {k: np.array([t for t, _ in v]) for k, v in runs.items()}
    stats = {k: [pd.read_csv(p) for _, p in v] for k, v in runs.items()}
    peaks = {k: np.array([df.ram_used_mb.max() / 1024 for df in v]) for k, v in stats.items()}
    host_gib = max(df.ram_total_mb.iloc[0] for v in stats.values() for df in v) / 1024
    bm = {}
    if bare:
        for k in runs:
            r = collect(bare, *k)
            if r:
                bm[k] = np.mean([t for t, _ in r])

    for k, v in times.items():
        print(f"{k[0]:7s} {k[1]:14s} n={len(v):2d} mean={v.mean():6.2f} sd={v.std(ddof=1):5.2f} min={v.min():5.1f} max={v.max():5.1f}")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    rng = np.random.default_rng(1)
    n_runs = len(next(iter(times.values())))

    def out(name):
        return os.path.join(args.output, name)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (sc, name) in enumerate(DATASETS):
        for j, lang in enumerate(LANGS):
            v = times[(lang, sc)]
            x = bar_pair(ax, i, j)
            ax.bar(x, v.mean(), BAR_W * 0.92, yerr=v.std(ddof=1), capsize=4, color=LANGS[lang], alpha=0.75)
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=14, color="k", alpha=0.55, zorder=3)
            if (lang, sc) in bm:
                ax.scatter(x, bm[(lang, sc)], marker="D", s=40, color="white", edgecolor="k", zorder=4)
    ax.set_xticks(range(2))
    ax.set_xticklabels([n for _, n in DATASETS])
    ax.set_ylabel("Training time (s)")
    ax.set_title(f"Training time, {n_runs} runs per combination")
    handles = [Patch(color=c, alpha=0.75, label=f"{l}: bar = mean") for l, c in LANGS.items()]
    handles += [
        Line2D([], [], color="k", marker="_", ls="", markersize=10, label="whisker = 1 standard deviation"),
        Line2D([], [], color="k", marker="o", ls="", alpha=0.55, markersize=4, label="dot = one run"),
    ]
    if bm:
        handles.append(Line2D([], [], color="k", marker="D", mfc="white", ls="", markersize=6, label="diamond = bare metal mean"))
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out("training-time.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (sc, name) in enumerate(DATASETS):
        for j, lang in enumerate(LANGS):
            v = peaks[(lang, sc)]
            x = bar_pair(ax, i, j)
            ax.bar(x, v.max(), BAR_W * 0.92, color=LANGS[lang], alpha=0.75)
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=14, color="k", alpha=0.55, zorder=3)
    ax.axhline(host_gib, color="gray", ls=":", lw=1)
    ax.set_xticks(range(2))
    ax.set_xticklabels([n for _, n in DATASETS])
    ax.set_ylabel("Peak RAM, whole host (GiB)")
    ax.set_title("Peak RAM per run")
    ax.set_ylim(0, host_gib * 1.17)
    handles = [Patch(color=c, alpha=0.75, label=f"{l}: bar = highest run") for l, c in LANGS.items()]
    handles += [
        Line2D([], [], color="k", marker="o", ls="", alpha=0.55, markersize=4, label="dot = peak of one run"),
        Line2D([], [], color="gray", ls=":", label="dotted line = host RAM"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.84), fontsize=9)
    fig.tight_layout()
    fig.savefig(out("ram-peak.png"), dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(9, 7.5), sharex="col")
    cols = [("cpu_percent", "CPU (%)", 1), ("gpu_util_percent", "GPU (%)", 1), ("vram_used_mb", "VRAM (GiB)", 1024)]
    for c, (sc, name) in enumerate(DATASETS):
        for lang in LANGS:
            v = times[(lang, sc)]
            df = stats[(lang, sc)][int(np.argsort(v)[len(v) // 2])]
            t = df.timestamp - df.timestamp.iloc[0]
            for r, (col, lab, div) in enumerate(cols):
                axes[r, c].plot(t, df[col] / div, color=LANGS[lang], lw=1.1, label=lang if r == 0 and c == 0 else None)
                if c == 0:
                    axes[r, c].set_ylabel(lab)
        axes[0, c].set_title(f"{name} (median run)")
        axes[2, c].set_xlabel("Time (s)")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out("resource-curves.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    v = np.sort(times[("Elixir", "train_cifar10")])
    py = times[("Python", "train_cifar10")].mean()
    ax.bar(range(1, len(v) + 1), v, color=LANGS["Elixir"], alpha=0.8)
    ax.axhline(py, color=LANGS["Python"], lw=2, label=f"Python mean ({py:.1f} s)")
    ax.set_xlabel("Run (sorted by time)")
    ax.set_ylabel("Training time (s)")
    ax.set_title(f"Elixir CIFAR-10, {len(v)} runs sorted")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out("elixir-cifar10-runs.png"), dpi=160)
    plt.close(fig)

    print(f"Wrote 4 graphs to {args.output}")


if __name__ == "__main__":
    main()
