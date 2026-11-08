#!/usr/bin/env bash
# Read-only. Exits 1 if any check FAILs.
# Test hooks: NVIDIA_SMI=<command> replaces nvidia-smi, PREFLIGHT_MEMINFO=<file> replaces /proc/meminfo

set -u

NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"
MEMINFO="${PREFLIGHT_MEMINFO:-/proc/meminfo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -t 1 ]; then
  C_PASS=$'\033[32m'; C_WARN=$'\033[33m'; C_FAIL=$'\033[31m'; C_OFF=$'\033[0m'
else
  C_PASS=""; C_WARN=""; C_FAIL=""; C_OFF=""
fi

FAILS=0
WARNS=0

pass() { printf '%sPASS%s  %s\n' "$C_PASS" "$C_OFF" "$1"; }
warn() { printf '%sWARN%s  %s\n        fix: %s\n' "$C_WARN" "$C_OFF" "$1" "$2"; WARNS=$((WARNS + 1)); }
fail() { printf '%sFAIL%s  %s\n        fix: %s\n' "$C_FAIL" "$C_OFF" "$1" "$2"; FAILS=$((FAILS + 1)); }

# "13.3" -> 1303, so versions compare as integers
ver_num() {
  local major minor
  major="${1%%.*}"; minor="${1#*.}"; minor="${minor%%.*}"
  [ "$minor" = "$1" ] && minor=0
  echo $((major * 100 + minor))
}

# The expected CUDA version comes from the base image tag in the Dockerfile
REQUIRED_CUDA="13.3"
BASE_TAG=""
if [ -f "$PROJECT_DIR/docker/Dockerfile.elixir" ]; then
  BASE_TAG="$(sed -n 's/^FROM \(nvidia\/cuda:[^ ]*\).*/\1/p' "$PROJECT_DIR/docker/Dockerfile.elixir" | head -1)"
  tag_ver="$(echo "$BASE_TAG" | sed -n 's/^nvidia\/cuda:\([0-9]*\.[0-9]*\)\..*/\1/p')"
  [ -n "$tag_ver" ] && REQUIRED_CUDA="$tag_ver"
fi

echo "Preflight for the Elixir vs Python benchmark (read-only)"
echo "Image base from the Dockerfile: ${BASE_TAG:-not found, assuming CUDA $REQUIRED_CUDA}"
echo

arch="$(uname -m)"
if [ "$arch" = "x86_64" ]; then
  pass "CPU architecture x86_64 (the images are amd64)"
else
  fail "CPU architecture is $arch, the images are amd64 only" "use an x86_64 machine; arm64 was not tried"
fi
echo "      OS: $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || echo unknown) (verified here: Ubuntu 24.04)"

GPU_NAME=""; CC=""; VRAM_MIB=""; DRIVER=""; CUDA_REPORTED=""; DISPLAY_ACTIVE=""; VRAM_USED=""
SMI_TEXT=""
if ! command -v "${NVIDIA_SMI%% *}" >/dev/null 2>&1; then
  fail "nvidia-smi not found: no NVIDIA driver visible" "install the NVIDIA driver from your distribution, reboot, then rerun"
else
  SMI_TEXT="$($NVIDIA_SMI 2>&1)"
  q="$($NVIDIA_SMI --query-gpu=name,compute_cap,memory.total,driver_version,display_active,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
  if [ -z "$q" ]; then
    # Older drivers have no compute_cap field: retry without it
    q2="$($NVIDIA_SMI --query-gpu=name,memory.total,driver_version,display_active,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
    if [ -n "$q2" ]; then
      IFS=',' read -r GPU_NAME VRAM_MIB DRIVER DISPLAY_ACTIVE VRAM_USED <<<"$q2"
    fi
  else
    IFS=',' read -r GPU_NAME CC VRAM_MIB DRIVER DISPLAY_ACTIVE VRAM_USED <<<"$q"
  fi
  GPU_NAME="$(echo "$GPU_NAME" | sed 's/^ *//;s/ *$//')"; CC="$(echo "$CC" | tr -d ' ')"
  VRAM_MIB="$(echo "$VRAM_MIB" | tr -d ' ')"; DRIVER="$(echo "$DRIVER" | tr -d ' ')"
  DISPLAY_ACTIVE="$(echo "$DISPLAY_ACTIVE" | tr -d ' ')"; VRAM_USED="$(echo "$VRAM_USED" | tr -d ' ')"
  CUDA_REPORTED="$(echo "$SMI_TEXT" | sed -nE 's/.*CUDA( UMD)? Version: *([0-9]+\.[0-9]+).*/\2/p' | head -1)"

  if [ -z "$GPU_NAME" ]; then
    fail "nvidia-smi ran but listed no GPU" "check that the driver is loaded (nvidia-smi should show a table) and reboot if it was just installed"
  else
    echo "      GPU: $GPU_NAME, ${VRAM_MIB} MiB, driver ${DRIVER:-unknown}, CUDA reported ${CUDA_REPORTED:-unknown}, compute capability ${CC:-unknown}"
    if [ "$($NVIDIA_SMI -L 2>/dev/null | grep -c '^GPU ')" -gt 1 ]; then
      warn "more than one GPU is visible" "untested here; the compose file reserves all GPUs, so set CUDA_VISIBLE_DEVICES or use count: 1 for a clean single GPU run"
    fi

    if [ -z "$CC" ]; then
      warn "compute capability not reported (old driver)" "look up your GPU at https://developer.nvidia.com/cuda-gpus; CUDA 13 images need 7.5 or higher (Turing)"
    elif [ "$(ver_num "$CC")" -ge 705 ]; then
      pass "compute capability $CC (CUDA 13 needs 7.5 or higher; 8.6 verified here)"
    else
      fail "compute capability $CC is below 7.5: CUDA 13.0 removed Maxwell, Pascal and Volta" "use a CUDA 12 base image and XLA_TARGET=cuda12 in your copy of the Dockerfiles (expected to work, not tested; see the guide)"
    fi

    if [ "${VRAM_MIB:-0}" -ge 8000 ] 2>/dev/null; then
      pass "VRAM ${VRAM_MIB} MiB (8192 MiB verified here; XLA takes about 90 percent, TensorFlow about 6GB)"
    else
      warn "VRAM ${VRAM_MIB:-unknown} MiB is below the 8GB tested" "untested: XLA preallocates about 90 percent of the card, so it may run, but an out of memory error on CIFAR-10 is possible; close everything using the GPU"
    fi

    if [ -z "$CUDA_REPORTED" ]; then
      warn "could not read the CUDA version from nvidia-smi" "read it at the top of the nvidia-smi table (CUDA Version) and compare with the guide"
    elif [ "$(ver_num "$CUDA_REPORTED")" -ge "$(ver_num "$REQUIRED_CUDA")" ]; then
      pass "driver $DRIVER reports CUDA $CUDA_REPORTED, the image needs $REQUIRED_CUDA or newer"
    else
      fail "driver $DRIVER reports CUDA $CUDA_REPORTED, the image needs $REQUIRED_CUDA (NVIDIA_REQUIRE_CUDA): the container will refuse to start" "update the driver (610 or newer for CUDA 13.3), or lower the base image tag in both Dockerfiles to a CUDA version at or below $CUDA_REPORTED and rebuild (see the guide)"
    fi

    if [ "$DISPLAY_ACTIVE" = "Enabled" ] || { [ -n "$VRAM_USED" ] && [ "$VRAM_USED" -gt 500 ] 2>/dev/null; }; then
      warn "the GPU is in use by a display or other process (${VRAM_USED:-?} MiB used, display ${DISPLAY_ACTIVE:-unknown}); the desktop adds noise to timings and takes VRAM and RAM" "stop the desktop for measured runs: sudo systemctl isolate multi-user.target (work over SSH in tmux); restore with sudo systemctl isolate graphical.target"
    else
      pass "no display on the GPU, ${VRAM_USED:-0} MiB in use"
    fi
  fi
fi

MEM_KB="$(awk '/^MemTotal:/ {print $2}' "$MEMINFO" 2>/dev/null)"
AVAIL_KB="$(awk '/^MemAvailable:/ {print $2}' "$MEMINFO" 2>/dev/null)"
SWAP_KB="$(awk '/^SwapTotal:/ {print $2}' "$MEMINFO" 2>/dev/null)"
MEM_MB=$(( ${MEM_KB:-0} / 1024 )); AVAIL_MB=$(( ${AVAIL_KB:-0} / 1024 )); SWAP_MB=$(( ${SWAP_KB:-0} / 1024 ))
CORES="$(nproc 2>/dev/null || echo 1)"
MEM_LIMIT="6g"; MAKE_JOBS=4
if [ "$MEM_MB" -ge 7000 ]; then
  pass "RAM ${MEM_MB} MB, swap ${SWAP_MB} MB (7877 MB verified here with mem_limit 6g and -j4)"
elif [ "$MEM_MB" -ge 4500 ]; then
  MEM_LIMIT="$(( MEM_MB * 3 / 4 / 1024 ))g"; MAKE_JOBS=2
  warn "RAM ${MEM_MB} MB is below the 7.7GB tested" "set mem_limit and memswap_limit to $MEM_LIMIT in docker-compose.yml and MAKEFLAGS=-j2 in docker/Dockerfile.elixir (expected, not tested); the EXLA compile can be OOM-killed"
elif [ "$MEM_MB" -ge 3000 ]; then
  MEM_LIMIT="$(( MEM_MB * 3 / 4 / 1024 ))g"; MAKE_JOBS=1
  warn "RAM ${MEM_MB} MB is far below the 7.7GB tested; the CIFAR-10 runs peaked at 3.9GB of host RAM here, so they may be OOM-killed under a mem_limit of $MEM_LIMIT" "set mem_limit to $MEM_LIMIT and MAKEFLAGS=-j1 and expect a slow build; add swap only for the build, never for the runs (swap makes timings meaningless)"
else
  fail "RAM ${MEM_MB} MB is too small for the EXLA compile and the runs" "use a machine with at least 8GB; 7.7GB was the tested minimum"
fi
MAKE_JOBS=$(( MAKE_JOBS > CORES ? CORES : MAKE_JOBS ))
if [ "$AVAIL_MB" -lt 3000 ] && [ "$MEM_MB" -ge 4500 ]; then
  warn "only ${AVAIL_MB} MB of RAM is available right now (3GB or more needed before a run)" "close applications or stop the desktop and rerun this check"
fi

# Docker root if readable, else /
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
DISK_PATH="${DOCKER_ROOT:-/}"; [ -d "$DISK_PATH" ] || DISK_PATH="/"
DISK_GB="$(df -BG --output=avail "$DISK_PATH" 2>/dev/null | tail -1 | tr -dc '0-9')"
DISK_GB="${DISK_GB:-0}"
if [ "$DISK_GB" -ge 70 ]; then
  pass "${DISK_GB}GB free on $DISK_PATH (images 35GB + build cache 17.6GB + archive 8.3GB = about 61GB)"
elif [ "$DISK_GB" -ge 55 ]; then
  warn "${DISK_GB}GB free on $DISK_PATH; images and build cache need about 53GB, the optional archive 8.3GB more" "free some space or skip tools/save-images.sh"
else
  fail "${DISK_GB}GB free on $DISK_PATH; images (35GB) and build cache (17.6GB) need about 53GB" "free disk space or move Docker's data root to a bigger disk"
fi

if ! command -v docker >/dev/null 2>&1; then
  fail "docker not found" "install Docker (Ubuntu: docker.io, docker-compose-v2, docker-buildx; or Docker's own packages)"
else
  DOCKER_V="$(docker --version 2>/dev/null)"
  if docker info >/dev/null 2>&1; then
    pass "$DOCKER_V, daemon reachable (29.1.3 verified here)"
  else
    fail "$DOCKER_V, but the daemon is not reachable by this user" "start the service (sudo systemctl enable --now docker) and add yourself to the docker group (sudo usermod -aG docker \$USER, then log in again)"
  fi
  if COMPOSE_V="$(docker compose version 2>/dev/null)"; then
    pass "$COMPOSE_V (2.40.3 verified here; v2 syntax is required)"
  else
    fail "Docker Compose v2 plugin not found" "install docker-compose-v2 (Ubuntu) or docker-compose-plugin"
  fi
  if command -v nvidia-ctk >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then
    pass "NVIDIA Container Toolkit installed and registered with Docker ($(nvidia-ctk --version 2>/dev/null | head -1 | sed 's/^ *//'))"
  elif command -v nvidia-ctk >/dev/null 2>&1; then
    fail "NVIDIA Container Toolkit installed but Docker has no nvidia runtime" "sudo nvidia-ctk runtime configure --runtime=docker, then sudo systemctl restart docker"
  else
    fail "NVIDIA Container Toolkit not found" "install it from NVIDIA's apt repository: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
  fi
fi

echo
echo "Values to set in your own copy of the project"
if [ -n "$CUDA_REPORTED" ]; then
  cuda_num="$(ver_num "$CUDA_REPORTED")"
  if [ "$cuda_num" -ge 1300 ]; then XLA="cuda13"; elif [ "$cuda_num" -ge 1201 ]; then XLA="cuda12"; else XLA="none (CUDA $CUDA_REPORTED is older than XLA's cuda12 target needs, 12.1)"; fi
else
  XLA="unknown (driver CUDA version not read)"
fi
echo "  Base image (both Dockerfiles, FROM): ${BASE_TAG:-nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04} (unchanged works if the CUDA check passed)"
echo "  XLA_TARGET (docker/Dockerfile.elixir): $XLA"
echo "  mem_limit and memswap_limit (docker-compose.yml): $MEM_LIMIT"
echo "  MAKEFLAGS (docker/Dockerfile.elixir): -j$MAKE_JOBS"
echo
echo "Summary: $FAILS FAIL, $WARNS WARN"
[ "$FAILS" -eq 0 ]
