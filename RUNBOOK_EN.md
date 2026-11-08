# Running the Elixir vs Python benchmark
>This was translated using AI.

This project trains models on the MNIST and CIFAR-10 datasets using Elixir (Nx, Axon, EXLA) and Python (Keras, TensorFlow) in Docker, on an NVIDIA GPU, and samples CPU, RAM, GPU and VRAM.

This document shows how to get to a first successful run. It was tested on a machine with the following configuration:
- Intel i5-6400 processor (4 cores, no hyperthreading)
- 7.7GB of RAM; 2GB of swap
- RTX 3060 Ti graphics card (8GB of VRAM, compute capability 8.6)
- Ubuntu 24.04
- NVIDIA driver 610.57.04
- Docker 29.1.3
- Docker Compose 2.40.3
- NVIDIA Container Toolkit 1.20.1

## 1. Check
Run the check script from the project root. The script makes no change to the computer or the project. PASS, WARN or FAIL is printed for each check, together with the reason and a fix, and finally the values that need to be set (useful in [section 3](#3-adapt-the-files)). If there is any FAIL, the script returns 1. All FAILs must be fixed before continuing.

## 2. Host setup
1. **Driver:** Install an NVIDIA driver whose CUDA version is 13.3 or higher. Run `nvidia-smi` and look for `CUDA Version: 13.3` or higher in the output. Since this project uses containers, the host does not need the CUDA Toolkit, cuDNN, NCCL or NVSHMEM.
2. **Docker:** Install the Ubuntu packages `docker.io`, `docker-compose-v2` and `docker-buildx`. Add your user to the `docker` group (`sudo usermod -aG docker $USER`) and log in again.
3. **NVIDIA Container Toolkit:** Install it from the [NVIDIA apt repository][3], then run `sudo nvidia-ctk runtime configure --runtime=docker` and `sudo systemctl restart docker`.
4. **Check:** `tools/preflight.sh` should return no FAIL.

For other distributions: the drivers, Docker and the toolkit will differ depending on the package manager. NVIDIA has [documentation for the main distributions][3]. Follow the official documentation and run the check script.

## 3. Adapt the files
These depend on each machine's configuration, so they need editing if yours differs.

**Base image tag**, line 3 of `docker/Dockerfile.elixir` and line 3 of `docker/Dockerfile.python` (both must be identical):
```
FROM nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04
```
The image carries `NVIDIA_REQUIRE_CUDA=cuda>=13.3`, which forces the host to have CUDA 13.3 or higher. CUDA 13.2 (from driver 595.91.07) prevented the container from starting. The Elixir Dockerfile also has settings tied to CUDA 13.3: `libnvshmem3-cuda-13`, the `nvshmem/13` paths and `libnvrtc-builtins.so.13.3`.

**`XLA_TARGET`**, line 57 of `docker/Dockerfile.elixir`:
```
ENV XLA_TARGET=cuda13
```
Use `cuda13` for a CUDA 13 image, or `cuda12` for a CUDA 12.1 or higher image (expected to work according to the [documentation][4]).

**`mem_limit` and `memswap_limit`**, lines 6 and 7 of `docker-compose.yml`:
```
mem_limit: 6g
memswap_limit: 6g
```
Keeping them equal prevents swapping. This way an out-of-memory condition shows up as the process being killed instead of a slow benchmark.

**`MAKEFLAGS`**, line 58 of `docker/Dockerfile.elixir`:
```
ENV MAKEFLAGS=-j4
```
It only affects the native compilation of EXLA. Lower values are recommended for machines with less than 7GB of RAM.

**Display:** turn off the graphical interface with `sudo systemctl isolate multi-user.target`, run the benchmarks inside a `tmux` session, and turn it back on with `sudo systemctl isolate graphical.target`.

**Datasets:** `datasets/scidata/` (Elixir) and `datasets/keras/datasets/` (Python) are mounted in the container, and `TMPDIR` and `KERAS_HOME` point there. Create these directories if they do not exist (`mkdir -p datasets/scidata datasets/keras`).

## 4. First run
All commands must be run from the project root. Build one image at a time because the EXLA compilation uses a lot of RAM. Because of this, the Elixir image takes a while to build (about 30 minutes).
```bash
tools/preflight.sh
COMPOSE_PARALLEL_LIMIT=1 docker compose build
```
Do not run `docker builder prune` or `docker system prune -a` afterwards, because that would delete the compilation.

Compose reads the whole file on every command, so `BATCH` must be set even for the smoke tests below (any value works).

Smoke tests, without training:
```bash
export BATCH=mine
docker compose run --rm elixir-bench mix run -e 'IO.inspect(Nx.default_backend()); IO.inspect(EXLA.Client.get_supported_platforms())'
docker compose run --rm python-bench python3 -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```
Expected result: `{EXLA.Backend, [client: :cuda]}`; `%{host: 4, cuda: 1}` (host should match the number of CPU cores); `2.21.0`; `PhysicalDevice GPU:0`.

A short run per stack, with results saved to a separate directory:
```bash
BATCH=mine RUNS=1 docker compose run --rm elixir-bench
BATCH=mine RUNS=1 docker compose run --rm python-bench
```
`BATCH` is the results folder under `results/docker/` (required), `RUNS` the number of runs (default 5) and `DATASET` is `mnist` (default) or `cifar10`; the sampling interval stays 0.1 s. For example `BATCH=batch4 DATASET=cifar10 docker compose run --rm elixir-bench`. Run only one service at a time, never `docker compose up`.

### Optional
An optional archive of the images (`tools/save-images.sh`) that can easily be used on a fresh OS (`tools/load-images.sh <archive.tar.zst>`). Restoring the images took 6 minutes and 23 seconds.

### Expected results
Training time, mean of 15 Docker runs, standard deviation in parentheses:

|          | Elixir (s)                       | Python (s)   |
| -------- | -------------------------------- | ------------ |
| MNIST    | 7.04 (0.10)                      | 11.25 (0.10) |
| CIFAR-10 | 24.26 (2.73), range 18.9 to 27.0 | 28.25 (0.26) |


## Sources
[1]: https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/ (NVIDIA, "Navigating GPU architecture support")
[2]: https://docs.nvidia.com/cuda/archive/13.3.1/cuda-toolkit-release-notes/index.html (NVIDIA, CUDA Toolkit 13.3 Update 1 release notes)
[3]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html (NVIDIA Container Toolkit install guide: )
[4]: https://github.com/elixir-nx/xla (XLA for Elixir, README)
[5]: https://www.tensorflow.org/install/pip (TensorFlow, pip install)
