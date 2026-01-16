#!/bin/sh
# This script uses bash features (arrays, [[ ]], etc).
# Make it robust even if invoked as: sh run_train_recommended.sh
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# Recommended training launcher for:
#   /home/nishioka/GNN/GNN_hole_2026/GNN_program/GNN_zscore_sub_noise_defect_free.py
#
# Key notes based on the last run (20260115_015448):
# - Your script default is --data_usage_ratio 0.5, so explicitly set 1.0 for full-data training.
# - Layer-constraint masking and per-file prediction mapping were fixed in the script; re-run to regenerate outputs.
#
# Usage:
#   bash run_train_recommended.sh

CONDA_PY="/home/nishioka/miniconda3/envs/gnn_final_env/bin/python"
TORCHRUN="/home/nishioka/miniconda3/envs/gnn_final_env/bin/torchrun"

SCRIPT="/home/nishioka/GNN/GNN_hole_2026/GNN_program/GNN_zscore_sub_noise_defect_free.py"
WORKDIR="/home/nishioka/GNN/GNN_hole_2026/GNN_program"
LAUNCHER="/home/nishioka/GNN/tools/launch_run.py"
PROFILE="${PROFILE:-ndf_recommended}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/nishioka/GNN/runs}"
DATASET_TAG="${DATASET_TAG:-NDF}"

# GPUs
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# Training hyperparams (baseline)
EPOCHS="${EPOCHS:-2000}"
PATIENCE="${PATIENCE:-300}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HIDDEN="${HIDDEN:-16}"
# NOTE: 0.01 can be unstable for this setup; use 0.002 as a safer default.
LR="${LR:-0.002}"
WD="${WD:-5e-4}"
DROPOUT="${DROPOUT:-0.10}"
EDGE_DROP="${EDGE_DROP:-0.01}"

# Imbalance handling
USE_CLASS_FREQ_SAMPLER="${USE_CLASS_FREQ_SAMPLER:-1}"  # 1=on, 0=off

# Data
DATA_USAGE_RATIO="${DATA_USAGE_RATIO:-1.0}" # IMPORTANT: override script default(0.5)

# Performance
USE_AMP="${USE_AMP:-1}"                     # 1=on, 0=off

# Logging
# - LOG_LEVEL: DEBUG|INFO|WARNING|ERROR (default: INFO)
# - VERBOSE_PRINT: 1 enables extra debug blocks in the python script
LOG_LEVEL="${LOG_LEVEL:-INFO}"
VERBOSE_PRINT="${VERBOSE_PRINT:-0}"

ts="$(date +'%Y%m%d_%H%M%S')"
RUN_ID="${RUN_ID:-}"
AUTO_RUN_ID=0
RESUME="${RESUME:-auto}"   # auto|off
DRY_RUN="${DRY_RUN:-0}"   # 1=dry run (no training)
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"  # 1=preflight checks only

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export LOG_LEVEL
export VERBOSE_PRINT

die() {
  echo "[ERR] $*" >&2
  exit 1
}

preflight() {
  [[ -x "$CONDA_PY" ]] || die "python not found: $CONDA_PY"
  [[ -x "$TORCHRUN" ]] || die "torchrun not found: $TORCHRUN"
  [[ -f "$SCRIPT" ]] || die "script not found: $SCRIPT"
  [[ -f "$LAUNCHER" ]] || die "launcher not found: $LAUNCHER"
  [[ -d "$WORKDIR" ]] || die "workdir not found: $WORKDIR"

  # GPU visibility / count check (fail fast if nproc_per_node is impossible).
  NPROC_PER_NODE="$NPROC_PER_NODE" "$CONDA_PY" - <<'PY'
import os, sys
import torch

req = int(os.environ.get("NPROC_PER_NODE", "1"))
cuda = torch.cuda.is_available()
ng = torch.cuda.device_count()
print("[INFO] torch:", getattr(torch, "__version__", "unknown"))
print("[INFO] cuda_available:", cuda, "num_gpus:", ng)

if req > 1 and not cuda:
    print(f"[ERR] NPROC_PER_NODE={req} but CUDA is not available")
    sys.exit(1)
if cuda and req > ng:
    print(f"[ERR] NPROC_PER_NODE={req} but only {ng} GPUs visible")
    sys.exit(1)
PY
}

slug_num() {
  # Make numeric-ish strings filename-friendly (e.g., 0.002 -> 0p002, 5e-4 -> 5em4)
  local s="${1}"
  s="${s//./p}"
  s="${s//-/m}"
  s="${s//+/p}"
  echo "${s}"
}

maybe_autoset_run_id() {
  # If RUN_ID is not provided, create a compact informative one:
  # dataset + epoch + lr (+ git sha)
  [[ -n "${RUN_ID}" ]] && return 0

  local git_sha="nogit"
  local git_dirty=""
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_sha="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
    if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
      git_dirty="_dirty"
    fi
  fi

  local lr_s; lr_s="$(slug_num "${LR}")"
  RUN_ID="${ts}_${git_sha}${git_dirty}_ds${DATASET_TAG}_ep${EPOCHS}_lr${lr_s}"
  AUTO_RUN_ID=1
}

cd "$WORKDIR"
preflight
maybe_autoset_run_id

echo "[INFO] Using python: $("$CONDA_PY" -c 'import sys; print(sys.executable)')"
echo "[INFO] GPUs (nproc_per_node): ${NPROC_PER_NODE}"
echo "[INFO] RUN_ID: ${RUN_ID:-'(auto)'}"
echo "[INFO] RESUME: ${RESUME}"
echo "[INFO] DRY_RUN: ${DRY_RUN}"
echo "[INFO] PREFLIGHT_ONLY: ${PREFLIGHT_ONLY}"
echo "[INFO] LOG_LEVEL: ${LOG_LEVEL}"
echo "[INFO] VERBOSE_PRINT: ${VERBOSE_PRINT}"

LOG_DIR="${LOG_DIR:-/home/nishioka/GNN/runs/_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_${ts}${RUN_ID:+_${RUN_ID}}.log}"
echo "[INFO] LOG_FILE: ${LOG_FILE}"

args=(
  "--hidden_channels" "${HIDDEN}"
  "--learning_rate" "${LR}"
  "--weight_decay" "${WD}"
  "--batch_size" "${BATCH_SIZE}"
  "--epochs" "${EPOCHS}"
  "--patience" "${PATIENCE}"
  "--dropout" "${DROPOUT}"
  "--edge_drop_prob" "${EDGE_DROP}"
  "--data_usage_ratio" "${DATA_USAGE_RATIO}"
  "--use_onecycle"
)

if [[ "${USE_AMP}" == "1" ]]; then
  args+=("--use_amp")
fi

if [[ "${USE_CLASS_FREQ_SAMPLER}" == "1" ]]; then
  args+=("--use_class_frequency_sampler")
fi

echo "[INFO] Command:"
echo "  ${CONDA_PY} ${LAUNCHER} --profile ${PROFILE} --torchrun ${TORCHRUN} --nproc_per_node ${NPROC_PER_NODE} --script ${SCRIPT} --workdir ${WORKDIR} --output_base ${OUTPUT_BASE} ${RUN_ID:+--run_id ${RUN_ID}} --resume ${RESUME} -- ${args[*]}"

launcher_extra=()
if [[ "${DRY_RUN}" == "1" ]]; then
  launcher_extra+=("--dry_run")
fi
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  launcher_extra+=("--preflight_only")
fi

"$CONDA_PY" "$LAUNCHER" \
  --profile "$PROFILE" \
  --torchrun "$TORCHRUN" \
  --nproc_per_node "$NPROC_PER_NODE" \
  --script "$SCRIPT" \
  --workdir "$WORKDIR" \
  --output_base "$OUTPUT_BASE" \
  ${RUN_ID:+--run_id "$RUN_ID"} \
  --resume "$RESUME" \
  "${launcher_extra[@]}" \
  -- "${args[@]}" 2>&1 | tee -a "$LOG_FILE"

FINAL_RUN_ID="$RUN_ID"
FINAL_RUN_DIR="${OUTPUT_BASE}/${RUN_ID}"

if [[ "${DRY_RUN}" != "1" && "${PREFLIGHT_ONLY}" != "1" && "${AUTO_RUN_ID}" == "1" ]]; then
  RUN_DIR="${OUTPUT_BASE}/${RUN_ID}"
  SUMMARY_JSON="${RUN_DIR}/meta/summary.json"
  if [[ -f "$SUMMARY_JSON" ]]; then
    F1_TAG="$(
      SUMMARY_JSON="$SUMMARY_JSON" "$CONDA_PY" - <<'PY'
import json, os
p = os.environ["SUMMARY_JSON"]
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)
f1 = d.get("best_macro_f1", None)
if f1 is None:
    print("")
else:
    print(("F1" + f"{float(f1):.3f}").replace(".", "p"))
PY
    )"
    if [[ -n "$F1_TAG" ]]; then
      NEW_RUN_ID="${RUN_ID}_${F1_TAG}"
      NEW_RUN_DIR="${OUTPUT_BASE}/${NEW_RUN_ID}"
      if [[ "$NEW_RUN_ID" != "$RUN_ID" && ! -e "$NEW_RUN_DIR" ]]; then
        mv "$RUN_DIR" "$NEW_RUN_DIR"
        ln -s "$NEW_RUN_DIR" "$RUN_DIR"
        FINAL_RUN_ID="$NEW_RUN_ID"
        FINAL_RUN_DIR="$NEW_RUN_DIR"
        echo "[INFO] Renamed run dir: ${NEW_RUN_DIR}"
      fi
    fi
  fi
fi

echo "$FINAL_RUN_DIR" > "${LOG_DIR}/last_run_dir.txt"
echo "$FINAL_RUN_ID" > "${LOG_DIR}/last_run_id.txt"
echo "[INFO] FINAL_RUN_ID: ${FINAL_RUN_ID}"
echo "[INFO] FINAL_RUN_DIR: ${FINAL_RUN_DIR}"
echo "[INFO] Done."

