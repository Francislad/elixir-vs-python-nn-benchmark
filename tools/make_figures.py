#!/usr/bin/env python3
"""Draws the article figures as vector PDFs. Usage: make_figures.py [--results DIR] [--datasets DIR] [--baremetal DIR|none] [-o DIR] [--preview DIR] [batch_dir ...]"""

import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

COLORS = {"Elixir": "#6e4a9e", "Python": "#c98a10"}
LANGS = list(COLORS)
DATASETS = [("train_mnist", "MNIST"), ("train_cifar10", "CIFAR-10")]
DOCKER_BATCHES = ["batch1-5runs", "batch2-10runs"]
TIME_RE = re.compile(r"Training time[^:]*:\s*([\d.]+)\s*seconds")
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
FULL = 6.3
MIB = 1024
BAR_W = 0.36


def collect(base, lang, script):
    runs = []
    for log in sorted(glob.glob(os.path.join(base, lang.lower(), script, "run_*_output.log"))):
        with open(log, errors="ignore") as fh:
            m = TIME_RE.search(fh.read())
        stats = log.replace("_output.log", "_stats.csv")
        if m and os.path.exists(stats):
            df = pd.read_csv(stats).iloc[1:]  # the first sample has a sampler artifact in the GPU reading
            runs.append((float(m.group(1)), df))
    return runs


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "font.family": "DejaVu Sans",
    })


def save(fig, name, outdir, preview):
    fig.savefig(os.path.join(outdir, name + ".pdf"))
    if preview:
        fig.savefig(os.path.join(preview, name + ".png"), dpi=110)
    plt.close(fig)


def fig_datasets(datasets, outdir, preview):
    mnist = os.path.join(datasets, "keras", "datasets", "mnist.npz")
    base = os.path.join(datasets, "keras", "datasets", "cifar-10-batches-py-target", "cifar-10-batches-py")
    if not (os.path.exists(mnist) and os.path.isdir(base)):
        print(f"Dataset caches not found in {datasets}; skipping the sample figures")
        return
    with np.load(mnist) as d:
        x, y = d["x_train"], d["y_train"]
    fig, axes = plt.subplots(2, 5, figsize=(FULL * 0.8, 2.8))
    for digit, ax in enumerate(axes.flat):
        ax.imshow(x[np.argmax(y == digit)], cmap="gray")
        ax.set_title(f"label {digit}")
        ax.axis("off")
    save(fig, "fig-mnist-samples", outdir, preview)

    with open(os.path.join(base, "data_batch_1"), "rb") as fh:
        batch = pickle.load(fh, encoding="bytes")
    data, labels = batch[b"data"], np.array(batch[b"labels"])
    fig, axes = plt.subplots(2, 5, figsize=(FULL * 0.8, 3.0))
    for cls, ax in enumerate(axes.flat):
        img = data[np.argmax(labels == cls)].reshape(3, 32, 32).transpose(1, 2, 0)
        ax.imshow(img)
        ax.set_title(CIFAR_CLASSES[cls])
        ax.axis("off")
    save(fig, "fig-cifar10-samples", outdir, preview)


def box(ax, x, y, w, h, text, fc, ec="#444444", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#444444", lw=1.0))


def fig_overview(outdir, preview):
    fig, ax = plt.subplots(figsize=(FULL, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.01, 0.58, 0.27, 0.33, "Elixir container\nNx, Axon, EXLA, Scidata\ntrain_mnist.exs\ntrain_cifar10.exs", "#e6def2")
    box(ax, 0.01, 0.10, 0.27, 0.33, "Python container\nKeras 3, TensorFlow\ntrain_mnist.py\ntrain_cifar10.py", "#f7e8c4")
    box(ax, 0.345, 0.30, 0.31, 0.40, "Benchmark wrapper (bash)\nruns the script N times and\nsamples CPU, RAM, GPU\nand VRAM every 0.1 s", "#dbe8f5", fs=7.5)
    box(ax, 0.73, 0.58, 0.26, 0.33, "Per run files\noutput log with the\ntraining time, stats CSV\nwith the samples", "#e3efe0")
    box(ax, 0.73, 0.10, 0.26, 0.33, "Analysis\nstatistics, figures,\nxlsx report", "#e3efe0")
    arrow(ax, 0.28, 0.74, 0.345, 0.60)
    arrow(ax, 0.28, 0.26, 0.345, 0.40)
    arrow(ax, 0.655, 0.60, 0.73, 0.72)
    arrow(ax, 0.86, 0.58, 0.86, 0.43)
    ax.text(0.5, 0.02, "One container at a time on the NVIDIA GPU", ha="center", fontsize=8, style="italic", color="#555555")
    save(fig, "fig-overview", outdir, preview)


def bar_x(i, j):
    return i + (j - 0.5) * BAR_W


def fig_training_time(docker, bare, outdir, preview):
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(FULL * 0.8, 3.2))
    for i, (sc, _) in enumerate(DATASETS):
        for j, lang in enumerate(LANGS):
            v = np.array([t for t, _ in docker[(lang, sc)]])
            x = bar_x(i, j)
            ax.bar(x, v.mean(), BAR_W * 0.92, yerr=v.std(ddof=1), capsize=3, color=COLORS[lang], alpha=0.75)
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=9, color="k", alpha=0.5, zorder=3, linewidths=0)
            b = [t for t, _ in bare[(lang, sc)]]
            if b:
                ax.scatter(x, np.mean(b), marker="D", s=28, color="white", edgecolor="k", zorder=4, linewidths=0.8)
    ax.set_xticks(range(2))
    ax.set_xticklabels([n for _, n in DATASETS])
    ax.set_ylabel("Training time (s)")
    ax.set_ylim(0, 1.25 * max(t for v in docker.values() for t, _ in v))
    handles = [Patch(color=c, alpha=0.75, label=f"{l}, mean") for l, c in COLORS.items()]
    handles += [
        Line2D([], [], color="k", marker="_", ls="", markersize=8, label="whisker: 1 SD"),
        Line2D([], [], color="k", marker="o", ls="", alpha=0.5, markersize=3, label="one Docker run"),
    ]
    if any(bare.values()):
        handles.append(Line2D([], [], color="k", marker="D", mfc="white", ls="", markersize=5, label="bare metal mean"))
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=2)
    save(fig, "fig-training-time", outdir, preview)


def fig_ram(docker, outdir, preview):
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=(FULL * 0.8, 3.0))
    host = max(df.ram_total_mb.iloc[0] for v in docker.values() for _, df in v) / MIB
    for i, (sc, _) in enumerate(DATASETS):
        for j, lang in enumerate(LANGS):
            v = np.array([df.ram_used_mb.max() / MIB for _, df in docker[(lang, sc)]])
            x = bar_x(i, j)
            ax.bar(x, v.mean(), BAR_W * 0.92, color=COLORS[lang], alpha=0.75)
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=9, color="k", alpha=0.5, zorder=3, linewidths=0)
    ax.axhline(host, color="gray", ls=":", lw=1)
    ax.set_xticks(range(2))
    ax.set_xticklabels([n for _, n in DATASETS])
    ax.set_ylabel("Peak RAM, whole host (GiB)")
    ax.set_ylim(0, host * 1.2)
    handles = [Patch(color=c, alpha=0.75, label=f"{l}, mean of peaks") for l, c in COLORS.items()]
    handles += [Line2D([], [], color="k", marker="o", ls="", alpha=0.5, markersize=3, label="peak of one run"),
                Line2D([], [], color="gray", ls=":", label="host RAM")]
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=2)
    save(fig, "fig-ram-peak", outdir, preview)


def fig_curves(docker, outdir, preview):
    rows = [("cpu_percent", "CPU (%)", 1.0), ("ram_used_mb", "RAM (GiB)", 1024.0), ("gpu_util_percent", "GPU (%)", 1.0), ("vram_used_mb", "VRAM (GiB)", 1024.0)]
    fig, axes = plt.subplots(4, 2, figsize=(FULL, 6.6), sharex="col")
    for c, (sc, name) in enumerate(DATASETS):
        for lang in LANGS:
            runs = docker[(lang, sc)]
            order = np.argsort([t for t, _ in runs])
            for r, (col, lab, div) in enumerate(rows):
                ax = axes[r, c]
                for _, df in runs:
                    t = df.timestamp - df.timestamp.iloc[0]
                    ax.plot(t, df[col] / div, color=COLORS[lang], lw=0.4, alpha=0.25)
                _, med = runs[order[len(runs) // 2]]
                t = med.timestamp - med.timestamp.iloc[0]
                ax.plot(t, med[col] / div, color=COLORS[lang], lw=1.3, label=lang if r == 0 and c == 0 else None)
                if c == 0:
                    ax.set_ylabel(lab)
        axes[0, c].set_title(name)
        axes[3, c].set_xlabel("Time since start of the run (s)")
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig-resource-curves", outdir, preview)


def fig_elixir_cifar(docker, outdir, preview):
    v = np.sort([t for t, _ in docker[("Elixir", "train_cifar10")]])
    py = np.mean([t for t, _ in docker[("Python", "train_cifar10")]])
    fig, ax = plt.subplots(figsize=(FULL * 0.8, 2.8))
    ax.bar(range(1, len(v) + 1), v, color=COLORS["Elixir"], alpha=0.8)
    ax.axhline(py, color=COLORS["Python"], lw=1.8, label=f"Python mean ({py:.2f} s)")
    ax.set_xlabel("Elixir CIFAR-10 run, sorted by training time")
    ax.set_ylabel("Training time (s)")
    ax.set_xticks(range(1, len(v) + 1))
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig-elixir-cifar10-runs", outdir, preview)


def main():
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="Docker batch folders to pool (default: batch1-5runs and batch2-10runs under results/docker)")
    ap.add_argument("--results", default=os.path.join(project, "results"))
    ap.add_argument("--baremetal", help="Bare metal folder with elixir/ and python/, or 'none' (default: baremetal under --results)")
    ap.add_argument("--datasets", default=os.path.join(project, "datasets"))
    ap.add_argument("-o", "--output", default=os.path.join(project, "results", "article", "figures"))
    ap.add_argument("--preview", default=None)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    if args.preview:
        os.makedirs(args.preview, exist_ok=True)

    batches = args.dirs or [os.path.join(args.results, "docker", b) for b in DOCKER_BATCHES]
    bare_dir = args.baremetal or os.path.join(args.results, "baremetal")
    if bare_dir == "none" or not os.path.isdir(bare_dir):
        bare_dir = None
    docker, bare = {}, {}
    for lang in LANGS:
        for sc, _ in DATASETS:
            docker[(lang, sc)] = sum((collect(d, lang, sc) for d in batches), [])
            bare[(lang, sc)] = collect(bare_dir, lang, sc) if bare_dir else []
    empty = [k for k, v in docker.items() if not v]
    if empty:
        sys.exit(f"No Docker runs found for {empty}; check the batch folders.")

    style()
    fig_datasets(args.datasets, args.output, args.preview)
    fig_overview(args.output, args.preview)
    fig_training_time(docker, bare, args.output, args.preview)
    fig_ram(docker, args.output, args.preview)
    fig_curves(docker, args.output, args.preview)
    fig_elixir_cifar(docker, args.output, args.preview)
    print(f"Wrote figures to {args.output}")


if __name__ == "__main__":
    main()
