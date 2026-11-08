#!/bin/bash
# Usage: run_benchmark_elixir.sh <scripts/script.exs> [num_runs] [interval_s] [output_dir] [project_dir]
# Writes <output_dir>/<script>/run_<i>_stats.csv, run_<i>_output.log and summary.csv

set -u

# Under pt_BR awk prints decimals with a comma, which splits CSV fields
export LC_NUMERIC=C

SCRIPT="${1:?Usage: $0 <scripts/script.exs> [num_runs] [interval_seconds] [output_dir] [project_dir]}"
RUNS="${2:-1}"
INTERVAL="${3:-0.1}"
BASE_OUTDIR="${4:-./benchmark_results}"
PROJECT_DIR="${5:-.}"

if [ ! -f "$PROJECT_DIR/$SCRIPT" ]; then
  echo "Error: script '$SCRIPT' not found inside project dir '$PROJECT_DIR'." >&2
  exit 1
fi

if ! command -v mix >/dev/null 2>&1; then
  echo "Error: 'mix' not found in PATH. Make sure asdf/Elixir is set up (e.g. 'asdf shell' or a fresh shell after asdf install)." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Warning: nvidia-smi not found. GPU/VRAM columns will fail." >&2
fi

SCRIPT_NAME=$(basename "$SCRIPT")
SCRIPT_NAME="${SCRIPT_NAME%.*}"
OUTDIR="$BASE_OUTDIR/$SCRIPT_NAME"

mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.csv"
if [ ! -f "$SUMMARY" ]; then
  echo "script,run,elapsed_seconds,exit_code" > "$SUMMARY"
fi

# CPU% comes from the delta between two /proc/stat reads, not raw jiffies
monitor_loop() {
  local outfile="$1"
  local interval="$2"

  read -r _ u1 n1 s1 i1 io1 irq1 so1 st1 _ < /proc/stat

  while true; do
    sleep "$interval"

    read -r _ u2 n2 s2 i2 io2 irq2 so2 st2 _ < /proc/stat

    local total1=$((u1+n1+s1+i1+io1+irq1+so1+st1))
    local total2=$((u2+n2+s2+i2+io2+irq2+so2+st2))
    local idle1=$((i1+io1))
    local idle2=$((i2+io2))
    local dtotal=$((total2-total1))
    local didle=$((idle2-idle1))

    local cpu_pct
    if [ "$dtotal" -gt 0 ]; then
      cpu_pct=$(awk -v dt="$dtotal" -v di="$didle" 'BEGIN{printf "%.2f", ((dt-di)/dt)*100}')
    else
      cpu_pct="0.00"
    fi

    local ram
    ram=$(free -m | awk 'NR==2{print $3","$2}')

    local gpu
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
          --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
    if [ -z "$gpu" ]; then
      gpu=",,"
    fi

    local ts
    ts=$(date +%s.%N)

    echo "${ts},${cpu_pct},${ram},${gpu}" >> "$outfile"

    u1=$u2; n1=$n2; s1=$s2; i1=$i2; io1=$io2; irq1=$irq2; so1=$so2; st1=$st2
  done
}

for ((i=1; i<=RUNS; i++)); do
  STATS_FILE="$OUTDIR/run_${i}_stats.csv"
  LOG_FILE="$OUTDIR/run_${i}_output.log"

  echo "timestamp,cpu_percent,ram_used_mb,ram_total_mb,gpu_util_percent,vram_used_mb,vram_total_mb" > "$STATS_FILE"

  echo "[$SCRIPT_NAME][run $i/$RUNS] starting monitor + mix run $SCRIPT (in $PROJECT_DIR) ..."

  monitor_loop "$STATS_FILE" "$INTERVAL" &
  MONITOR_PID=$!

  START=$(date +%s.%N)
  (cd "$PROJECT_DIR" && mix run "$SCRIPT") > "$LOG_FILE" 2>&1
  EXIT_CODE=$?
  END=$(date +%s.%N)

  kill "$MONITOR_PID" 2>/dev/null
  wait "$MONITOR_PID" 2>/dev/null

  ELAPSED=$(awk -v a="$START" -v b="$END" 'BEGIN{printf "%.3f", b-a}')

  echo "${SCRIPT_NAME},${i},${ELAPSED},${EXIT_CODE}" >> "$SUMMARY"
  echo "[$SCRIPT_NAME][run $i/$RUNS] done. elapsed=${ELAPSED}s exit_code=${EXIT_CODE} -> $STATS_FILE"
done

echo "All runs complete. Results in: $OUTDIR"
