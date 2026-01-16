#!/bin/sh
# This script uses bash features; re-exec under bash if needed.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# Simple hyperparameter sweep (recommended first step): LR sweep
#
# Usage examples:
#   bash run_sweep_lr.sh
#   LRS="0.001 0.002 0.005" bash run_sweep_lr.sh
#   EPOCHS=800 LRS="0.001 0.002" bash run_sweep_lr.sh
#   DATASET_TAG=NDF OUTPUT_BASE=/home/nishioka/GNN/runs bash run_sweep_lr.sh
#
# Notes:
# - Each run is executed sequentially.
# - Per-run artifacts stay under OUTPUT_BASE (same as run_train_recommended.sh).
# - Sweep summary is appended to runs/_sweeps/*.csv.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SH="${TRAIN_SH:-${SCRIPT_DIR}/run_train_recommended.sh}"

[[ -f "$TRAIN_SH" ]] || { echo "[ERR] TRAIN_SH not found: $TRAIN_SH" >&2; exit 1; }

ts="$(date +'%Y%m%d_%H%M%S')"

OUTPUT_BASE="${OUTPUT_BASE:-/home/nishioka/GNN/runs}"
LOG_DIR="${LOG_DIR:-/home/nishioka/GNN/runs/_logs}"
SWEEP_DIR="${SWEEP_DIR:-/home/nishioka/GNN/runs/_sweeps}"
mkdir -p "$LOG_DIR" "$SWEEP_DIR"

SWEEP_LOG="${SWEEP_LOG:-${LOG_DIR}/sweep_lr_${ts}.log}"
SWEEP_CSV="${SWEEP_CSV:-${SWEEP_DIR}/sweep_lr_${ts}.csv}"

# What to sweep
LRS="${LRS:-0.001 0.002 0.005}"

echo "[INFO] Sweep LR: ${LRS}"
echo "[INFO] Sweep log: ${SWEEP_LOG}"
echo "[INFO] Sweep csv: ${SWEEP_CSV}"

if [[ ! -f "$SWEEP_CSV" ]]; then
  echo "ts,run_id,run_dir,lr,epochs,dataset_tag,best_macro_f1,exit_code" > "$SWEEP_CSV"
fi

for lr in $LRS; do
  echo ""
  echo "[INFO] ===== sweep lr=${lr} =====" | tee -a "$SWEEP_LOG"

  # Force auto RUN_ID to keep naming consistent, but record params in CSV.
  (
    export LR="$lr"
    export RUN_ID=""
    bash "$TRAIN_SH"
  ) 2>&1 | tee -a "$SWEEP_LOG"

  LAST_RUN_DIR="$(cat "${LOG_DIR}/last_run_dir.txt" 2>/dev/null || true)"
  LAST_RUN_ID="$(cat "${LOG_DIR}/last_run_id.txt" 2>/dev/null || true)"

  if [[ -z "$LAST_RUN_DIR" || -z "$LAST_RUN_ID" ]]; then
    echo "[ERR] Could not read last_run_dir/last_run_id from ${LOG_DIR}. Did training script finish?" | tee -a "$SWEEP_LOG"
    exit 1
  fi

  SUMMARY_JSON="${LAST_RUN_DIR}/meta/summary.json"
  if [[ ! -f "$SUMMARY_JSON" ]]; then
    echo "[ERR] summary.json not found: ${SUMMARY_JSON}" | tee -a "$SWEEP_LOG"
    exit 1
  fi

  # Extract best_macro_f1 and exit_code from summary.json (robust JSON parsing).
  read -r best_f1 exit_code < <(python - <<'PY'
import json, sys
import os
p = os.environ["SUMMARY_JSON"]
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)
best = d.get("best_macro_f1", None)
code = d.get("exit_code", None)
best_s = "" if best is None else f"{float(best):.6f}"
code_s = "" if code is None else str(int(code))
print(best_s, code_s)
PY
  )

  # Pick up common envs if they exist (they are typically set in run_train_recommended.sh).
  epochs="${EPOCHS:-}"
  dataset_tag="${DATASET_TAG:-}"

  echo "${ts},${LAST_RUN_ID},${LAST_RUN_DIR},${lr},${epochs},${dataset_tag},${best_f1},${exit_code}" >> "$SWEEP_CSV"
  echo "[INFO] Recorded: run_id=${LAST_RUN_ID} best_macro_f1=${best_f1}" | tee -a "$SWEEP_LOG"
done

echo ""
echo "[INFO] Sweep done."
echo "[INFO] CSV: ${SWEEP_CSV}"

