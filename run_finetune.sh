#!/bin/sh
# This script uses bash features (arrays, [[ ]], etc).
# Make it robust even if invoked as: sh run_finetune.sh
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

# Fine-tuning script for:
#   /home/nishioka/GNN/GNN_hole_2026/GNN_program/GNN_zscore_sub_noise_defect_free.py
#
# Usage:
#   bash run_finetune.sh
#
# Environment variables to customize:
#   FINE_TUNE_FROM: Path to pretrained model (default: specified checkpoint)
#   FREEZE_BACKBONE: 1 to freeze backbone layers (default: 0)
#   BACKBONE_LR: Learning rate for backbone (default: same as LR)
#   HEAD_LR: Learning rate for classification head (default: same as LR)
#   EPOCHS: Number of epochs (default: 500)
#   LR: Learning rate (default: 0.0001 for fine-tuning)

CONDA_PY="/home/nishioka/miniconda3/envs/gnn_final_env/bin/python"
TORCHRUN="/home/nishioka/miniconda3/envs/gnn_final_env/bin/torchrun"

SCRIPT="/home/nishioka/GNN/GNN_hole_2026/GNN_program/GNN_zscore_sub_noise_defect_free.py"
WORKDIR="/home/nishioka/GNN/GNN_hole_2026/GNN_program"
LAUNCHER="/home/nishioka/GNN/tools/launch_run.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${PROFILE:-ndf_recommended}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/nishioka/GNN/runs}"
DATASET_TAG="${DATASET_TAG:-NDF}"

# Pretrained model checkpoint
FINE_TUNE_FROM="${FINE_TUNE_FROM:-/home/nishioka/GNN/runs/20260116_104929_nogit_dsNDF_ep2000_lr0p001_F10p730/outputs/GNN_model/19classmodel_hole_zscore/GATModel_20260116_104950_Best_Final.pth}"

# GPUs
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# Fine-tuning hyperparams
EPOCHS="${EPOCHS:-500}"  # Typically fewer epochs for fine-tuning
PATIENCE="${PATIENCE:-100}"  # Reduced patience for fine-tuning
BATCH_SIZE="${BATCH_SIZE:-64}"
HIDDEN="${HIDDEN:-16}"
LR="${LR:-0.0001}"  # Lower learning rate for fine-tuning
WD="${WD:-5e-4}"
DROPOUT="${DROPOUT:-0.10}"
EDGE_DROP="${EDGE_DROP:-0.01}"

# Fine-tuning options
FREEZE_BACKBONE="${FREEZE_BACKBONE:-0}"  # 1=freeze backbone, 0=train all layers
BACKBONE_LR="${BACKBONE_LR:-}"  # If empty, uses LR
HEAD_LR="${HEAD_LR:-}"  # If empty, uses LR

# Imbalance handling
USE_CLASS_FREQ_SAMPLER="${USE_CLASS_FREQ_SAMPLER:-1}"  # 1=on, 0=off

# Data
DATA_USAGE_RATIO="${DATA_USAGE_RATIO:-1.0}"

# Performance
USE_AMP="${USE_AMP:-1}"                     # 1=on, 0=off

# Logging
LOG_LEVEL="${LOG_LEVEL:-INFO}"
VERBOSE_PRINT="${VERBOSE_PRINT:-0}"

ts="$(date +'%Y%m%d_%H%M%S')"
RUN_ID="${RUN_ID:-}"
AUTO_RUN_ID=0
RESUME="${RESUME:-off}"   # Fine-tuning starts from epoch 0, so no resume
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

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
  [[ -f "$FINE_TUNE_FROM" ]] || die "pretrained model not found: $FINE_TUNE_FROM"

  # GPU visibility / count check
  NPROC_PER_NODE="$NPROC_PER_NODE" "$CONDA_PY" - <<'PY'
import os, sys
import torch

req = int(os.environ.get("NPROC_PER_NODE", "1"))
cuda = torch.cuda.is_available()
if not cuda:
    print("[WARNING] CUDA not available", file=sys.stderr)
    sys.exit(0)  # Not fatal, but warn

count = torch.cuda.device_count()
if count < req:
    print(f"[ERR] Only {count} GPU(s) available, but NPROC_PER_NODE={req}", file=sys.stderr)
    sys.exit(1)
PY
}

slug_num() {
  # Convert 0.001 -> 0p001, 0.0001 -> 0p0001
  echo "$1" | sed 's/\./p/g' | sed 's/^-//'
}

maybe_autoset_run_id() {
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
  local ft_tag="ft"
  if [[ "${FREEZE_BACKBONE}" == "1" ]]; then
    ft_tag="ft_frozen"
  fi
  RUN_ID="${ts}_${git_sha}${git_dirty}_${ft_tag}_ds${DATASET_TAG}_ep${EPOCHS}_lr${lr_s}"
  AUTO_RUN_ID=1
}

cd "$WORKDIR"
preflight
maybe_autoset_run_id

echo "[INFO] Fine-tuning configuration:"
echo "[INFO]   Pretrained model: ${FINE_TUNE_FROM}"
echo "[INFO]   Freeze backbone: ${FREEZE_BACKBONE}"
echo "[INFO]   Backbone LR: ${BACKBONE_LR:-${LR} (default)}"
echo "[INFO]   Head LR: ${HEAD_LR:-${LR} (default)}"
echo "[INFO]   Learning rate: ${LR}"
echo "[INFO]   Epochs: ${EPOCHS}"
echo "[INFO]   Using python: $("$CONDA_PY" -c 'import sys; print(sys.executable)')"
echo "[INFO]   GPUs (nproc_per_node): ${NPROC_PER_NODE}"
echo "[INFO]   RUN_ID: ${RUN_ID:-'(auto)'}"
echo "[INFO]   RESUME: ${RESUME}"
echo "[INFO]   DRY_RUN: ${DRY_RUN}"
echo "[INFO]   PREFLIGHT_ONLY: ${PREFLIGHT_ONLY}"
echo "[INFO]   LOG_LEVEL: ${LOG_LEVEL}"
echo "[INFO]   VERBOSE_PRINT: ${VERBOSE_PRINT}"

LOG_DIR="${LOG_DIR:-/home/nishioka/GNN/runs/_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/finetune_${ts}${RUN_ID:+_${RUN_ID}}.log}"
echo "[INFO] LOG_FILE: ${LOG_FILE}"

args=(
  "--fine_tune_from" "${FINE_TUNE_FROM}"
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

if [[ "${FREEZE_BACKBONE}" == "1" ]]; then
  args+=("--freeze_backbone")
fi

if [[ -n "${BACKBONE_LR}" ]]; then
  args+=("--backbone_lr" "${BACKBONE_LR}")
fi

if [[ -n "${HEAD_LR}" ]]; then
  args+=("--head_lr" "${HEAD_LR}")
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
  -- \
  "${args[@]}" 2>&1 | tee -a "$LOG_FILE"

exit_code="${PIPESTATUS[0]}"
echo "[INFO] Fine-tuning finished with exit code: ${exit_code}"
exit "${exit_code}"
