#!/bin/bash
# おすすめ設定: Sampler + OneCycleLR（logit_adjustはデフォルト無効）
# Macro F1向上とNDFの誤検出抑制を両立

# NCCL環境変数の設定（タイムアウト対策）
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800  # 30分に延長

# タイムスタンプを取得
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ログファイル名
LOG_FILE="/home/nishioka/GNN/GNN_hole_2026/GNN_program/training_log_recommended_${TIMESTAMP}.log"

# 実行コマンド（改善3点セット適用）
# 1. Early stopping: Macro F1基準
# 2. Sampler: 1/sqrt(freq)ベースのクラス頻度サンプラー
COMMAND="torchrun --nproc_per_node=4 /home/nishioka/GNN/GNN_hole_2026/GNN_program/GNN_zscore_sub.py \
  --epochs 2000 \
  --learning_rate 0.002 \
  --batch_size 64 \
  --hidden_channels 16 \
  --patience 300 \
  --use_onecycle \
  --dropout 0.1 \
  --edge_drop_prob 0.01 \
  --use_class_frequency_sampler"

echo "Starting recommended training at $(date)"
echo "Log file: ${LOG_FILE}"
echo "Command: ${COMMAND}"
echo "=========================================="
echo "Configuration (改善3点セット):"
echo "  1. Early stopping: Macro F1基準（val_lossは監視のみ）"
echo "  2. Class Frequency Sampler: 1/sqrt(freq)ベースの重み付け"
echo "  - Batch Size: 64, Hidden Channels: 16"
echo "  - Learning Rate: 0.002 (適切な値に調整)"
echo "  - OneCycleLR, Dropout: 0.1, Edge Drop: 0.01"
echo "  - Epochs: 2000, Patience: 300"
echo "=========================================="

# nohupで実行（標準出力と標準エラー出力をログファイルに保存）
nohup ${COMMAND} > ${LOG_FILE} 2>&1 &

# プロセスIDを保存
PID=$!
echo "Training started with PID: ${PID}"
echo "PID: ${PID}" > "/home/nishioka/GNN/GNN_hole_2026/GNN_program/training_pid_recommended_${TIMESTAMP}.txt"
echo "To check progress: tail -f ${LOG_FILE}"
echo "To stop training: kill ${PID}"
