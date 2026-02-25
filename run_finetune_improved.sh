#!/bin/sh
# Improved fine-tuning script with better hyperparameters
# This script uses bash features (arrays, [[ ]], etc).
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

# Improved fine-tuning script
# Usage:
#   # Strategy 1: Freeze backbone, train head only (RECOMMENDED)
#   STRATEGY=freeze_backbone bash run_finetune_improved.sh
#
#   # Strategy 2: Different learning rates for backbone and head
#   STRATEGY=differential_lr bash run_finetune_improved.sh
#
#   # Strategy 3: Higher learning rate for all layers
#   STRATEGY=higher_lr bash run_finetune_improved.sh
#
#   # Strategy 4: Two-stage fine-tuning
#   STRATEGY=two_stage bash run_finetune_improved.sh

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

# Strategy selection
STRATEGY="${STRATEGY:-freeze_backbone}"  # freeze_backbone, differential_lr, higher_lr, two_stage

# Strategy-specific hyperparameters
case "$STRATEGY" in
  freeze_backbone)
    # Strategy 1: Freeze backbone, train head only (RECOMMENDED)
    EPOCHS="${EPOCHS:-500}"
    PATIENCE="${PATIENCE:-150}"
    LR="${LR:-0.001}"  # Higher LR for head only
    FREEZE_BACKBONE=1
    BACKBONE_LR=""
    HEAD_LR="${HEAD_LR:-0.001}"
    USE_ONECYCLE="${USE_ONECYCLE:-1}"
    ;;
  differential_lr)
    # Strategy 2: Different learning rates
    EPOCHS="${EPOCHS:-1000}"
    PATIENCE="${PATIENCE:-200}"
    LR="${LR:-0.001}"
    FREEZE_BACKBONE=0
    BACKBONE_LR="${BACKBONE_LR:-0.0001}"  # Low LR for backbone
    HEAD_LR="${HEAD_LR:-0.001}"  # Higher LR for head
    USE_ONECYCLE="${USE_ONECYCLE:-1}"
    ;;
  higher_lr)
    # Strategy 3: Higher learning rate for all layers
    EPOCHS="${EPOCHS:-1000}"
    PATIENCE="${PATIENCE:-200}"
    LR="${LR:-0.0005}"  # 5x higher than default
    FREEZE_BACKBONE=0
    BACKBONE_LR=""
    HEAD_LR=""
    USE_ONECYCLE="${USE_ONECYCLE:-1}"
    ;;
  two_stage)
    # Strategy 4: Two-stage (will be handled separately)
    echo "[INFO] Two-stage fine-tuning requires running twice. Use freeze_backbone first, then differential_lr."
    exit 1
    ;;
  *)
    echo "[ERROR] Unknown strategy: $STRATEGY"
    echo "Available strategies: freeze_backbone, differential_lr, higher_lr, two_stage"
    exit 1
    ;;
esac

# Common hyperparams
BATCH_SIZE="${BATCH_SIZE:-64}"
HIDDEN="${HIDDEN:-16}"
WD="${WD:-5e-4}"
DROPOUT="${DROPOUT:-0.10}"
EDGE_DROP="${EDGE_DROP:-0.01}"

# Imbalance handling
USE_CLASS_FREQ_SAMPLER="${USE_CLASS_FREQ_SAMPLER:-1}"

# Data
DATA_USAGE_RATIO="${DATA_USAGE_RATIO:-1.0}"

# Performance
USE_AMP="${USE_AMP:-1}"

# Logging
LOG_LEVEL="${LOG_LEVEL:-INFO}"
VERBOSE_PRINT="${VERBOSE_PRINT:-0}"

ts="$(date +'%Y%m%d_%H%M%S')"
RUN_ID="${RUN_ID:-}"
AUTO_RUN_ID=0
RESUME="${RESUME:-off}"
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
}

slug_num() {
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
  local strategy_tag="${STRATEGY}"
  if [[ "${FREEZE_BACKBONE}" == "1" ]]; then
    strategy_tag="${strategy_tag}_frozen"
  fi
  if [[ -n "${BACKBONE_LR}" ]]; then
    local blr_s; blr_s="$(slug_num "${BACKBONE_LR}")"
    strategy_tag="${strategy_tag}_blr${blr_s}"
  fi
  if [[ -n "${HEAD_LR}" ]]; then
    local hlr_s; hlr_s="$(slug_num "${HEAD_LR}")"
    strategy_tag="${strategy_tag}_hlr${hlr_s}"
  fi
  RUN_ID="${ts}_${git_sha}${git_dirty}_ft_${strategy_tag}_ep${EPOCHS}_lr${lr_s}"
  AUTO_RUN_ID=1
}

cd "$WORKDIR"
preflight
maybe_autoset_run_id

echo "[INFO] ============================================================"
echo "[INFO] Improved Fine-tuning Configuration"
echo "[INFO] ============================================================"
echo "[INFO] Strategy: ${STRATEGY}"
echo "[INFO] Pretrained model: ${FINE_TUNE_FROM}"
echo "[INFO] Freeze backbone: ${FREEZE_BACKBONE}"
if [[ -n "${BACKBONE_LR}" ]]; then
  echo "[INFO] Backbone LR: ${BACKBONE_LR}"
fi
if [[ -n "${HEAD_LR}" ]]; then
  echo "[INFO] Head LR: ${HEAD_LR}"
fi
echo "[INFO] Learning rate: ${LR}"
echo "[INFO] Epochs: ${EPOCHS}"
echo "[INFO] Patience: ${PATIENCE}"
echo "[INFO] Using python: $("$CONDA_PY" -c 'import sys; print(sys.executable)')"
echo "[INFO] GPUs (nproc_per_node): ${NPROC_PER_NODE}"
echo "[INFO] RUN_ID: ${RUN_ID}"
echo "[INFO] ============================================================"

LOG_DIR="${LOG_DIR:-/home/nishioka/GNN/runs/_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/finetune_improved_${ts}${RUN_ID:+_${RUN_ID}}.log}"
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
)

if [[ "${USE_AMP}" == "1" ]]; then
  args+=("--use_amp")
fi

if [[ "${USE_CLASS_FREQ_SAMPLER}" == "1" ]]; then
  args+=("--use_class_frequency_sampler")
fi

if [[ "${USE_ONECYCLE:-1}" == "1" ]]; then
  args+=("--use_onecycle")
else
  args+=("--no_onecycle")
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
