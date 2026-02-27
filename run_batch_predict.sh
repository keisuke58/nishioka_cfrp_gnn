#!/bin/bash
# M4-2: バッチ推論スクリプト
# フォルダ指定で一括推論、CSV出力
#
# Usage:
#   ./run_batch_predict.sh <input_folder> [output_csv]
#   ./run_batch_predict.sh GNN_hole_2026/all_sub_hole_defect_zscore_noise/test reports/predictions.csv
#
# MODEL_PATH でモデルを上書き可能

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_FOLDER="${1:?Usage: $0 <input_folder> [output_csv]}"
OUTPUT_CSV="${2:-reports/predictions_$(date +%Y%m%d_%H%M%S).csv}"
MODEL_PATH="${MODEL_PATH:-$SCRIPT_DIR/runs/20260116_104929_nogit_dsNDF_ep2000_lr0p001_F10p730/outputs/GNN_model/19classmodel_hole_zscore/GATModel_20260116_104950_Best_Final.pth}"

if [[ ! -d "$INPUT_FOLDER" ]]; then
  echo "[ERROR] Input folder not found: $INPUT_FOLDER"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"
echo "[INFO] Input: $INPUT_FOLDER"
echo "[INFO] Output CSV: $OUTPUT_CSV"

python "$SCRIPT_DIR/tools/batch_predict.py" \
  --input "$INPUT_FOLDER" \
  --output "$OUTPUT_CSV" \
  --model "$MODEL_PATH" \
  --batch_size 64
