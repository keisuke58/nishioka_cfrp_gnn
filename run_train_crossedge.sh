#!/bin/bash
# Cross-edge (Yehia方式) 有効での学習
# config.yaml の cross_edge.enabled: true を参照
# GNN_zscore_sub_noise_defect_free_integrated.py を使用

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${SCRIPT_DIR}/.env" ]] && set -a && source <(grep -v '^#' "${SCRIPT_DIR}/.env" | grep -v '^$' | sed 's/^export //') && set +a

CONDA_PY="/home/nishioka/miniconda3/envs/gnn_final_env/bin/python"
TORCHRUN="/home/nishioka/miniconda3/envs/gnn_final_env/bin/torchrun"
# Integrated script: OOD + Cross-edge 対応
SCRIPT="${SCRIPT_DIR}/GNN_hole_2026/GNN_program/GNN_zscore_sub_noise_defect_free_integrated.py"
WORKDIR="${SCRIPT_DIR}/GNN_hole_2026/GNN_program"
LAUNCHER="${SCRIPT_DIR}/tools/launch_run.py"
OUTPUT_BASE="${OUTPUT_BASE:-${SCRIPT_DIR}/runs}"
RUN_ID="${RUN_ID:-}"
ts="$(date +'%Y%m%d_%H%M%S')"
[[ -z "$RUN_ID" ]] && RUN_ID="${ts}_crossedge_ep${EPOCHS:-100}_lr${LR:-0p002}"

EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-25}"
LR="${LR:-0.002}"

cd "$WORKDIR"
echo "[INFO] Cross-edge training: $SCRIPT"
echo "[INFO] Config: config.yaml (cross_edge.enabled, split_type)"
echo "[INFO] RUN_ID: $RUN_ID"

"$CONDA_PY" "$LAUNCHER" \
  --profile ndf_recommended \
  --script "$SCRIPT" \
  --workdir "$WORKDIR" \
  --torchrun "$TORCHRUN" \
  --nproc_per_node "${NPROC_PER_NODE:-4}" \
  --output_base "$OUTPUT_BASE" \
  --run_id "$RUN_ID" \
  --resume auto \
  -- --hidden_channels 16 --learning_rate "$LR" --weight_decay 5e-4 \
  --batch_size 64 --epochs "$EPOCHS" --patience "$PATIENCE" \
  --dropout 0.10 --edge_drop_prob 0.01 --data_usage_ratio 1.0 \
  --use_onecycle --use_amp --use_class_frequency_sampler

echo "[INFO] Done. Run dir: $OUTPUT_BASE/$RUN_ID"
