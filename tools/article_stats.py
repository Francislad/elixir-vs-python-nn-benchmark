#!/usr/bin/env python3
"""Computes every number of the article from the raw runs. Usage: article_stats.py [--results DIR] [--baremetal DIR|none] [-o numbers.md] [batch_dir ...]"""

import argparse
import glob
import math
import os
import re
import sys
from itertools import combinations

import numpy as np
import pandas as pd

LANGS = ["Elixir", "Python"]
DATASETS = [("train_mnist", "MNIST"), ("train_cifar10", "CIFAR-10")]
DOCKER_BATCHES = ["batch1-5runs", "batch2-10runs"]
TIME_RE = re.compile(r"Training time[^:]*:\s*([\d.]+)\s*seconds")
FAST_LIMIT = 22.5  # seconds; splits the two modes of the Elixir CIFAR-10 runs
SEED = 20261102
MIB = 1024

# Reference values from the first analysis (time mean, sd, min, max; Docker 15 runs), used for the consistency checks
WRITEUP_DOCKER = {
    ("Elixir", "MNIST"): (7.04, 0.10, 6.89, 7.21),
    ("Python", "MNIST"): (11.25, 0.10, 11.14, 11.52),
    ("Elixir", "CIFAR-10"): (24.26, 2.73, 18.92, 27.01),
    ("Python", "CIFAR-10"): (28.25, 0.26, 27.83, 28.76),
}
WRITEUP_BARE = {
    ("Elixir", "MNIST"): (7.07, 0.08),
    ("Python", "MNIST"): (11.32, 0.08),
    ("Elixir", "CIFAR-10"): (26.53, 0.95),
    ("Python", "CIFAR-10"): (28.47, 0.26),
}
# (highest run peak RAM GiB, mean CPU %, mean GPU %, highest run peak VRAM GiB), as rounded in the write-up
WRITEUP_RESOURCES = {
    ("Elixir", "MNIST"): (2.6, 42, 9, 7.1),
    ("Python", "MNIST"): (3.0, 39, 3, 6.2),
    ("Elixir", "CIFAR-10"): (3.6, 45, 15, 7.1),
    ("Python", "CIFAR-10"): (3.9, 38, 12, 6.2),
}


def betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def t_critical(df, conf=0.95):
    alpha = 1.0 - conf
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def welch(x, y):
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    se2 = vx / nx + vy / ny
    t = (np.mean(x) - np.mean(y)) / math.sqrt(se2)
    df = se2**2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return t, df, t_two_sided_p(t, df)


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    sp = math.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / sp


def mann_whitney_p(x, y):
    """Two-sided p. Exact by enumeration when small, normal approximation otherwise."""
    nx, ny = len(x), len(y)
    allv = np.concatenate([x, y])
    ranks = pd.Series(allv).rank().to_numpy()
    u = ranks[:nx].sum() - nx * (nx + 1) / 2
    n = nx + ny
    if math.comb(n, nx) <= 300000:
        idx = range(n)
        hits = 0
        total = 0
        center = nx * ny / 2
        for pick in combinations(idx, nx):
            ur = ranks[list(pick)].sum() - nx * (nx + 1) / 2
            total += 1
            if abs(ur - center) >= abs(u - center) - 1e-9:
                hits += 1
        return u, hits / total, "exact"
    _, counts = np.unique(allv, return_counts=True)
    tie = (counts**3 - counts).sum() / (n * (n - 1))
    sigma = math.sqrt(nx * ny / 12.0 * ((n + 1) - tie))
    z = (abs(u - nx * ny / 2) - 0.5) / sigma
    return u, math.erfc(z / math.sqrt(2.0)), "normal approximation"


def mean_ci(x, conf=0.95):
    n = len(x)
    half = t_critical(n - 1, conf) * np.std(x, ddof=1) / math.sqrt(n)
    return np.mean(x) - half, np.mean(x) + half


def bootstrap_pct_diff(x, y, reps=10000):
    rng = np.random.default_rng(SEED)
    xs = rng.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
    ys = rng.choice(y, size=(reps, len(y)), replace=True).mean(axis=1)
    diff = (xs / ys - 1.0) * 100.0
    return np.percentile(diff, 2.5), np.percentile(diff, 97.5)


def collect(base, lang, script):
    runs = []
    pattern = os.path.join(base, lang.lower(), script, "run_*_output.log")
    for log in sorted(glob.glob(pattern)):
        with open(log, errors="ignore") as fh:
            m = TIME_RE.search(fh.read())
        stats = log.replace("_output.log", "_stats.csv")
        if m and os.path.exists(stats):
            df = pd.read_csv(stats)
            host, vram_total = df.ram_total_mb.iloc[0] / MIB, df.vram_total_mb.iloc[0] / MIB
            df = df.iloc[1:]  # the first sample has a sampler artifact in the GPU reading
            runs.append({
                "time": float(m.group(1)),
                "ram_gib": df.ram_used_mb.max() / MIB,
                "cpu": df.cpu_percent.mean(),
                "gpu": df.gpu_util_percent.mean(),
                "vram_gib": df.vram_used_mb.max() / MIB,
                "host_gib": host,
                "vram_total_gib": vram_total,
            })
    return runs


def summary(values):
    v = np.asarray(values, dtype=float)
    lo, hi = mean_ci(v)
    return {
        "n": len(v), "mean": v.mean(), "sd": v.std(ddof=1), "median": np.median(v),
        "min": v.min(), "max": v.max(), "ci_lo": lo, "ci_hi": hi, "cv": v.std(ddof=1) / v.mean() * 100,
    }


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def fnum(x, d=2):
    return f"{x:.{d}f}"


def fp(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def same(values):
    return values[0] if len(set(values)) == 1 else None


def main():
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="Docker batch folders to pool (default: batch1-5runs and batch2-10runs under results/docker)")
    ap.add_argument("--results", default=os.path.join(project, "results"))
    ap.add_argument("--baremetal", help="Bare metal folder with elixir/ and python/, or 'none' (default: baremetal under --results)")
    ap.add_argument("-o", "--output", default=os.path.join(project, "results", "article", "numbers.md"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    custom = bool(args.dirs)
    batches = args.dirs or [os.path.join(args.results, "docker", b) for b in DOCKER_BATCHES]
    bare_dir = args.baremetal or os.path.join(args.results, "baremetal")
    if bare_dir == "none" or not os.path.isdir(bare_dir):
        bare_dir = None

    docker = {}
    bare = {}
    for lang in LANGS:
        for script, ds in DATASETS:
            docker[(lang, ds)] = sum((collect(d, lang, script) for d in batches), [])
            bare[(lang, ds)] = collect(bare_dir, lang, script) if bare_dir else []
    short = [k for k, v in docker.items() if len(v) < 2]
    if short:
        sys.exit(f"Need at least 2 Docker runs for each of {short}; check the batch folders.")
    has_bare = all(len(v) >= 2 for v in bare.values())
    sets = [("Docker", docker)] + ([("Bare metal", bare)] if has_bare else [])

    quoted = [f"`{os.path.basename(os.path.normpath(d))}`" for d in batches]
    names = ", ".join(quoted[:-1]) + " and " + quoted[-1] if len(quoted) > 1 else quoted[0]
    n_docker = same([len(v) for v in docker.values()])
    n_bare = same([len(v) for v in bare.values()]) if has_bare else None
    docker_part = f"Docker means the pooled runs of {names}" + (f" ({n_docker} per combination)" if n_docker else "")
    if not has_bare:
        bare_part = "no bare metal runs were used"
    else:
        bare_part = "bare metal is the " + (f"{n_bare} run set" if n_bare else "set found in the bare metal folder")

    checks = []
    out = ["# Article numbers",
           f"Generated by `tools/article_stats.py` from the raw run files; do not edit by hand. {docker_part}; {bare_part}. Times are the script's own `Training time` line in seconds, sample standard deviation (n minus 1), 95% confidence intervals use the t distribution.",
           ""]

    out.append("## Training time")
    for label, data in sets:
        out.append(f"### {label}")
        rows = []
        for ds in [d for _, d in DATASETS]:
            for lang in LANGS:
                s = summary([r["time"] for r in data[(lang, ds)]])
                rows.append([lang, ds, s["n"], fnum(s["mean"]), fnum(s["sd"]), fnum(s["median"]), fnum(s["min"], 1), fnum(s["max"], 1),
                             f"{fnum(s['ci_lo'])} to {fnum(s['ci_hi'])}", fnum(s["cv"], 1)])
                if not custom:
                    exp = (WRITEUP_DOCKER if label == "Docker" else WRITEUP_BARE)[(lang, ds)]
                    ok = round(s["mean"], 2) == exp[0] and round(s["sd"], 2) == exp[1]
                    if label == "Docker":
                        ok = ok and round(s["min"], 2) == exp[2] and round(s["max"], 2) == exp[3]
                    checks.append((f"{label} time {lang} {ds}", ok))
        out += table(["Stack", "Dataset", "n", "Mean", "SD", "Median", "Min", "Max", "95% CI of mean", "CV %"], rows)
        out.append("")

    out.append("## Elixir against Python")
    out.append("Percent is Elixir mean over Python mean minus 1 (negative means Elixir is faster). The bootstrap interval resamples both groups 10000 times (seed " + str(SEED) + "). Welch is the two-sided unequal variance t test. Mann-Whitney is the two-sided rank test. Cohen's d uses the pooled standard deviation.")
    for label, data in sets:
        out.append(f"### {label}")
        rows = []
        for _, ds in DATASETS:
            x = np.array([r["time"] for r in data[("Elixir", ds)]])
            y = np.array([r["time"] for r in data[("Python", ds)]])
            pct = (x.mean() / y.mean() - 1) * 100
            blo, bhi = bootstrap_pct_diff(x, y)
            t, df, p = welch(x, y)
            u, pu, how = mann_whitney_p(x, y)
            rows.append([ds, f"{pct:.1f}", f"{blo:.1f} to {bhi:.1f}", fnum(x.mean() - y.mean()), f"{t:.2f}", f"{df:.1f}", fp(p),
                         f"{u:.0f}", fp(pu) + f" ({how})", f"{cohen_d(x, y):.2f}"])
        out += table(["Dataset", "Percent", "Bootstrap 95% CI", "Difference (s)", "Welch t", "df", "Welch p", "U", "Mann-Whitney p", "Cohen d"], rows)
        out.append("")

    out.append("## Elixir CIFAR-10 on Docker")
    times = np.array(sorted(r["time"] for r in docker[("Elixir", "CIFAR-10")]))
    fast, slow = times[times < FAST_LIMIT], times[times >= FAST_LIMIT]
    py = np.mean([r["time"] for r in docker[("Python", "CIFAR-10")]])
    gaps = np.diff(times)
    out.append("Sorted times (s): " + ", ".join(f"{t:.2f}" for t in times) + ".")
    if len(fast) and len(slow):
        out.append(f"Under {FAST_LIMIT} s: {len(fast)} of {len(times)} runs (mean {fast.mean():.2f}). Other runs: {len(slow)} (mean {slow.mean():.2f}, sd {slow.std(ddof=1):.2f}, range {slow.min():.2f} to {slow.max():.2f}).")
    else:
        out.append(f"Under {FAST_LIMIT} s: {len(fast)} of {len(times)} runs.")
    slow_part = f"Slow group mean against Python mean: {(slow.mean() / py - 1) * 100:.1f}%. " if len(slow) else ""
    out.append(f"{slow_part}Median of all {len(times)} against Python mean: {(np.median(times) / py - 1) * 100:.1f}%. Largest gap between neighbouring sorted runs: {gaps.max():.2f} s (between {times[gaps.argmax()]:.2f} and {times[gaps.argmax() + 1]:.2f}), to compare with a median gap of {np.median(gaps):.2f} s.")
    out.append("")
    if not custom:
        checks.append(("Elixir CIFAR-10 fast group size is 4", len(fast) == 4))
        checks.append(("Elixir CIFAR-10 slow mean is 25.7", round(slow.mean(), 1) == 25.7))

    out.append("## Resource use (Docker, per run, then mean over runs)")
    out.append("Peak RAM and peak VRAM are the highest sample of a run; CPU and GPU are the mean of the samples of a run. RAM and CPU are host wide. Cells are mean (sample standard deviation) across the runs; RAM and VRAM also show the highest run, which is the figure the write-up rounds.")
    rows = []
    for _, ds in DATASETS:
        for lang in LANGS:
            r = docker[(lang, ds)]
            cells = []
            for key, d in (("ram_gib", 2), ("cpu", 1), ("gpu", 1), ("vram_gib", 2)):
                v = np.array([x[key] for x in r])
                extra = f", max {v.max():.{d}f}" if key in ("ram_gib", "vram_gib") else ""
                cells.append(f"{v.mean():.{d}f} ({v.std(ddof=1):.{d}f}{extra})")
            rows.append([lang, ds] + cells)
            if not custom:
                exp = WRITEUP_RESOURCES[(lang, ds)]
                got = (max(x["ram_gib"] for x in r), np.mean([x["cpu"] for x in r]), np.mean([x["gpu"] for x in r]), max(x["vram_gib"] for x in r))
                ok = round(got[0], 1) == exp[0] and round(got[1]) == exp[1] and round(got[2]) == exp[2] and round(got[3], 1) == exp[3]
                detail = "" if ok else f" (write-up {exp}, recomputed RAM {got[0]:.3f}, CPU {got[1]:.2f}, GPU {got[2]:.2f}, VRAM {got[3]:.3f})"
                checks.append((f"Resources {lang} {ds}{detail}", ok))
    out += table(["Stack", "Dataset", "Peak RAM (GiB)", "Mean CPU (%)", "Mean GPU (%)", "Peak VRAM (GiB)"], rows)
    sample = next(iter(docker.values()))[0]
    out.append("")
    out.append(f"Host RAM total {sample['host_gib']:.1f} GiB, VRAM total {sample['vram_total_gib']:.1f} GiB (from the CSV headers of the samples).")
    out.append("")

    out.append("## Model sizes")
    mnist = (784 * 128 + 128) + (128 * 10 + 10)
    c1, bn1 = 3 * 3 * 3 * 32 + 32, 2 * 32
    c2, bn2 = 3 * 3 * 32 * 64 + 64, 2 * 64
    flat = 6 * 6 * 64
    d1, d2 = flat * 64 + 64, 64 * 10 + 10
    trainable = c1 + bn1 + c2 + bn2 + d1 + d2
    non_trainable = 2 * 32 + 2 * 64
    cifar = trainable + non_trainable
    out += table(["Model", "Total parameters", "Trainable", "Note"], [
        ["MNIST", f"{mnist:,}", f"{mnist:,}", "dense 128, dropout 0.5, dense 10"],
        ["CIFAR-10", f"{cifar:,}", f"{trainable:,}", "two conv blocks (32 and 64 filters, valid padding), batch norm, dense 64, dropout 0.5, dense 10; flatten size 2304"],
    ])
    out.append("")
    out.append("Computed by arithmetic from the layer shapes and confirmed with `model.count_params()` in the Python image (Keras 3.15.1): MNIST 101770, CIFAR-10 167946 total and 167754 trainable. The Elixir model has the same layers (see `elixir/scripts/train_cifar10.exs`).")
    checks.append(("Parameter counts match the Keras run", (mnist, cifar, trainable) == (101770, 167946, 167754)))
    out.append("")

    out.append("## Consistency checks against the write-up")
    if custom:
        out.append("Skipped for the time and resource figures: the write-up values describe the default batches.")
    out += [f"- {'OK' if ok else 'MISMATCH'}: {name}" for name, ok in checks]
    out.append("")

    with open(args.output, "w") as fh:
        fh.write("\n".join(out))
    bad = [n for n, ok in checks if not ok]
    print(f"Wrote {args.output}; {len(checks)} checks, {len(bad)} mismatches")
    for n in bad:
        print("MISMATCH:", n)


if __name__ == "__main__":
    main()
