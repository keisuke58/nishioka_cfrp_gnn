#!/bin/sh
# This script uses bash features (arrays, [[ ]], etc).
# Make it robust even if invoked as: sh run_train_integrated_crossedge.sh
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

# Training launcher for integrated script with Cross-edge enabled
# Usage:
#   bash run_train_integrated_crossedge.sh

CONDA_PY="/home/nishioka/miniconda3/envs/gnn_final_env/bin/python"
TORCHRUN="/home/nishioka/miniconda3/envs/gnn_final_env/bin/torchrun"

# 統合版スクリプトを使用
SCRIPT="/home/nishioka/GNN/GNN_hole_2026/GNN_program/GNN_zscore_sub_noise_defect_free_integrated.py"
WORKDIR="/home/nishioka/GNN/GNN_hole_2026/GNN_program"
LAUNCHER="/home/nishioka/GNN/tools/launch_run.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${PROFILE:-ndf_recommended}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/nishioka/GNN/runs}"
DATASET_TAG="${DATASET_TAG:-NDF}"

# GPUs
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# Training hyperparams (baseline)
EPOCHS="${EPOCHS:-2000}"
PATIENCE="${PATIENCE:-300}"
# Cross-edge使用時はメモリ使用量が増えるため、バッチサイズを減らす
BATCH_SIZE="${BATCH_SIZE:-32}"
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
  
  # config.yamlの存在確認
  [[ -f "${SCRIPT_DIR}/config.yaml" ]] || die "config.yaml not found: ${SCRIPT_DIR}/config.yaml"

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
  RUN_ID="${ts}_${git_sha}${git_dirty}_ds${DATASET_TAG}_ep${EPOCHS}_lr${lr_s}_crossedge"
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
echo "[INFO] Script: ${SCRIPT}"
echo "[INFO] Cross-edge: enabled (from config.yaml)"

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
  --profile "${PROFILE}" \
  --torchrun "${TORCHRUN}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --script "${SCRIPT}" \
  --workdir "${WORKDIR}" \
  --output_base "${OUTPUT_BASE}" \
  ${RUN_ID:+--run_id "${RUN_ID}"} \
  --resume "${RESUME}" \
  "${launcher_extra[@]}" \
  -- \
  "${args[@]}" \
  2>&1 | tee "${LOG_FILE}"

exit_code="${PIPESTATUS[0]}"
echo "[INFO] Training finished with exit code: ${exit_code}"
exit "${exit_code}"
