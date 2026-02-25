#!/bin/sh
# This script uses bash features; re-exec under bash if needed.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# Load .env file if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a
    source <(grep -v '^#' "${SCRIPT_DIR}/.env" | grep -v '^$' | sed 's/^export //')
    set +a
fi

# Simple hyperparameter sweep (recommended first step): LR sweep
#
# Usage examples:
#   bash run_sweep_lr.sh
#   LRS="0.001 0.002 0.005" bash run_sweep_lr.sh
#   EPOCHS=800 LRS="0.001 0.002" bash run_sweep_lr.sh
#   DATASET_TAG=NDF OUTPUT_BASE=/home/nishioka/GNN/runs bash run_sweep_lr.sh
#   MAX_RETRIES=3 bash run_sweep_lr.sh  # Enable automatic retry on failure
#
# Notes:
# - Each run is executed sequentially.
# - Per-run artifacts stay under OUTPUT_BASE (same as run_train_recommended.sh).
# - Sweep summary is appended to runs/_sweeps/*.csv.
# - Automatic retry on failure (controlled by MAX_RETRIES, default: 0 = disabled).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SH="${TRAIN_SH:-${SCRIPT_DIR}/run_train_recommended.sh}"
CONDA_PY="${CONDA_PY:-/home/nishioka/miniconda3/envs/gnn_final_env/bin/python}"

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

# Retry configuration
MAX_RETRIES="${MAX_RETRIES:-0}"  # 0 = disabled, >0 = number of retries
RETRY_DELAY="${RETRY_DELAY:-60}"  # seconds to wait before retry

echo "[INFO] Sweep LR: ${LRS}"
echo "[INFO] Sweep log: ${SWEEP_LOG}"
echo "[INFO] Sweep csv: ${SWEEP_CSV}"
echo "[INFO] Max retries: ${MAX_RETRIES} (0 = disabled)"
echo "[INFO] Retry delay: ${RETRY_DELAY}s"

if [[ ! -f "$SWEEP_CSV" ]]; then
  echo "ts,run_id,run_dir,lr,epochs,dataset_tag,best_macro_f1,exit_code,retry_count" > "$SWEEP_CSV"
fi

for lr in $LRS; do
  echo ""
  echo "[INFO] ===== sweep lr=${lr} =====" | tee -a "$SWEEP_LOG"

  retry_count=0
  success=false
  
  while [[ $retry_count -le $MAX_RETRIES ]]; do
    if [[ $retry_count -gt 0 ]]; then
      echo "[WARN] Retry attempt ${retry_count}/${MAX_RETRIES} for lr=${lr}" | tee -a "$SWEEP_LOG"
      echo "[INFO] Waiting ${RETRY_DELAY}s before retry..." | tee -a "$SWEEP_LOG"
      sleep "$RETRY_DELAY"
    fi

    # Force auto RUN_ID to keep naming consistent, but record params in CSV.
    set +e  # Temporarily disable exit on error to handle retries
    (
      export LR="$lr"
      export RUN_ID=""
      bash "$TRAIN_SH"
    ) 2>&1 | tee -a "$SWEEP_LOG"
    train_exit_code=$?
    set -e  # Re-enable exit on error

    LAST_RUN_DIR="$(cat "${LOG_DIR}/last_run_dir.txt" 2>/dev/null || true)"
    LAST_RUN_ID="$(cat "${LOG_DIR}/last_run_id.txt" 2>/dev/null || true)"

    if [[ -z "$LAST_RUN_DIR" || -z "$LAST_RUN_ID" ]]; then
      echo "[ERR] Could not read last_run_dir/last_run_id from ${LOG_DIR}. Did training script finish?" | tee -a "$SWEEP_LOG"
      if [[ $retry_count -lt $MAX_RETRIES ]]; then
        retry_count=$((retry_count + 1))
        continue
      else
        echo "[ERR] Max retries reached. Skipping lr=${lr}" | tee -a "$SWEEP_LOG"
        echo "${ts},ERROR,ERROR,${lr},${epochs},${dataset_tag},,${train_exit_code},${retry_count}" >> "$SWEEP_CSV"
        break
      fi
    fi

    SUMMARY_JSON="${LAST_RUN_DIR}/meta/summary.json"
    if [[ ! -f "$SUMMARY_JSON" ]]; then
      echo "[ERR] summary.json not found: ${SUMMARY_JSON}" | tee -a "$SWEEP_LOG"
      if [[ $retry_count -lt $MAX_RETRIES ]]; then
        retry_count=$((retry_count + 1))
        continue
      else
        echo "[ERR] Max retries reached. Skipping lr=${lr}" | tee -a "$SWEEP_LOG"
        echo "${ts},${LAST_RUN_ID},${LAST_RUN_DIR},${lr},${epochs},${dataset_tag},,${train_exit_code},${retry_count}" >> "$SWEEP_CSV"
        break
      fi
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

    # Check if training was successful (exit_code 0)
    if [[ "${exit_code}" == "0" ]]; then
      success=true
      break
    elif [[ $retry_count -lt $MAX_RETRIES ]]; then
      echo "[WARN] Training failed with exit_code=${exit_code}. Will retry." | tee -a "$SWEEP_LOG"
      retry_count=$((retry_count + 1))
      continue
    else
      echo "[WARN] Training failed with exit_code=${exit_code}. Max retries reached." | tee -a "$SWEEP_LOG"
      success=false
      break
    fi
  done

  echo "${ts},${LAST_RUN_ID},${LAST_RUN_DIR},${lr},${epochs},${dataset_tag},${best_f1},${exit_code},${retry_count}" >> "$SWEEP_CSV"
  if [[ "$success" == "true" ]]; then
    echo "[INFO] Recorded: run_id=${LAST_RUN_ID} best_macro_f1=${best_f1} (retries: ${retry_count})" | tee -a "$SWEEP_LOG"
  else
    echo "[WARN] Recorded failed run: run_id=${LAST_RUN_ID} (retries: ${retry_count})" | tee -a "$SWEEP_LOG"
  fi
done

echo ""
echo "[INFO] Sweep done."
echo "[INFO] CSV: ${SWEEP_CSV}"

# メール通知（オプション）
if [[ -n "${GMAIL_TO:-}" && -n "${GMAIL_FROM:-}" && -n "${GMAIL_PASSWORD:-}" ]]; then
  echo "[INFO] Sending email notification..."
  "$CONDA_PY" "${SCRIPT_DIR}/tools/email_notify.py" \
    --to "${GMAIL_TO}" \
    --from_email "${GMAIL_FROM}" \
    --password "${GMAIL_PASSWORD}" \
    --sweep_csv "${SWEEP_CSV}" \
    --html \
    || echo "[WARN] Failed to send email notification"
fi

# Slack通知（オプション）
if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
  echo "[INFO] Sending Slack notification..."
  "$CONDA_PY" "${SCRIPT_DIR}/tools/slack_notify.py" \
    --webhook_url "${SLACK_WEBHOOK_URL}" \
    --sweep_csv "${SWEEP_CSV}" \
    || echo "[WARN] Failed to send Slack notification"
fi

