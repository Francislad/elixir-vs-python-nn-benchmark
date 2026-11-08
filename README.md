# TCC: Elixir vs Python neural networks benchmark
>This was translated using AI.

Reproduction of the experiment from _Análise do potencial da linguagem Elixir no setor de redes neurais_; TAVANO, Lucas Campos; UTFPR; 2022. Elixir (Nx/Axon/EXLA) and Python (Keras/TensorFlow) train the same models (MNIST and CIFAR-10) on the same GPU while CPU, RAM, GPU and VRAM are sampled.

>New here? `RUNBOOK_EN.md` or `RUNBOOK_PT.md` can guide you to a first run.

## Requirements
- An NVIDIA GPU and a driver that reports CUDA 13.3 or newer (driver 610+).
- Docker with the NVIDIA Container Toolkit on Linux.
- Around 8GB of RAM.
- Disk space for the two images and the datasets.

>`tools/preflight.sh` exists to check all of this. Run that script.

## Layout
- `elixir/`: Mix project (`mix.exs`, `config/`, `scripts/train_mnist.exs`, `scripts/train_cifar10.exs`).
- `python/`: `train_mnist.py` and `train_cifar10.py`.
- `bench/`: `run_benchmark_elixir.sh` and `run_benchmark_python.sh`, wrappers that run a training script N times and sample the resources used into CSVs.
- `docker/`: `Dockerfile.elixir`, `Dockerfile.python`, `entrypoint-python.sh`. `docker-compose.yml` is at the project root.
- `tools/`: `preflight.sh`, `compile_results.py` (XLSX report), `plot_results.py` (PNG graphs), `fix_csv_locale.py`, and `save-images.sh` / `load-images.sh` (scripts to archive and restore the built images).
- `RUNBOOK_EN.md`, `RUNBOOK_PT.md`: step-by-step guide to running the project for the first time.
- `datasets/`: datasets cached by both stacks. `scidata/` is the Elixir cache; `keras/datasets/` is the Python cache.

## What a run does
Both stacks train the same models with the same settings:

| Dataset  | Optimizer | Loss                      | Epochs | Batch size |
| -------- | --------- | ------------------------- | ------ | ---------- |
| MNIST    | Adam      | categorical cross entropy | 5      | 128        |
| CIFAR-10 | Adam      | categorical cross entropy | 5      | 64         |

The wrapper repeats the script `RUNS` times and samples the machine every 0.1s. Under `results/docker/<BATCH>/<elixir or python>/<script>/` it saves, for each run `i`:
- `run_<i>_output.log`: the script output.
- `run_<i>_stats.csv`: one row per sample (CPU %, RAM, GPU %, VRAM).
- `summary.csv`: one per directory.

## Usage
Build one image at a time and archive them so this step does not have to be repeated:
```bash
COMPOSE_PARALLEL_LIMIT=1 docker compose build
tools/save-images.sh  # writes ~/docker-cache/nn-bench-images-<date>.tar.zst
# on a fresh OS, after driver + Docker + NVIDIA toolkit:
tools/load-images.sh ~/docker-cache/nn-bench-images-<date>.tar.zst
```
Rebuild only if `docker/` or `elixir/mix` change.

Run one stack at a time, choosing the parameters of the run:
```bash
BATCH=batch4 docker compose run --rm elixir-bench
BATCH=batch4 docker compose run --rm python-bench
BATCH=batch5 DATASET=cifar10 RUNS=10 docker compose run --rm elixir-bench
```
- `BATCH` (required): final directory for the results under `results/docker/`.
- `DATASET`: `mnist` (default) or `cifar10`.
- `RUNS`: number of runs (default 5).

>`BATCH` is read on every command, so it always has to be set.

Notes:
- Run one service at a time, over SSH and `tmux`, without the desktop (`sudo systemctl isolate multi-user.target`).
- The containers write as root; use `sudo chown -R $USER:$USER results datasets` to give the files back to your user.
- Network use is minimal, especially after caching.

## Report and graphs
They run on the host, in a virtual environment with the plotting packages:
```bash
python3 -m venv .venv && .venv/bin/pip install pandas openpyxl matplotlib
```

The xlsx report takes one `LABEL=folder` pair per results directory, so several results can be used:
```bash
.venv/bin/python tools/compile_results.py -o results/docker/report.xlsx \
  python=results/docker/batch1-5runs/python elixir=results/docker/batch1-5runs/elixir \
  python-extra=results/docker/batch2-10runs/python elixir-extra=results/docker/batch2-10runs/elixir
```

By default the graphs (PNG) use the results in `results/docker/batch*` and write to `results/plots/`:
```bash
.venv/bin/python tools/plot_results.py
```

For runs that do not use `batch*`, the directory path can be passed explicitly. It must contain `elixir/` and `python/`.
```bash
.venv/bin/python tools/plot_results.py results/docker/exp1 results/docker/exp2
```

Useful options:
- `-o <folder>`: output directory.

Because of the default pooling, give test or smoke runs a folder name that does not start with `batch` (the runbook uses `BATCH=mine`); they are then ignored unless you list them. Every combination (both stacks, both datasets) needs at least one run in the listed folders, or the script stops with "No runs found".