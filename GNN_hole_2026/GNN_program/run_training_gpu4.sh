#!/bin/bash
# GPU4つで分散学習を実行するスクリプト（改善3点セット適用）
# Macro F1向上を目指す最強の組み合わせ

# NCCL環境変数の設定（タイムアウト対策）
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800  # 30分に延長

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/home/nishioka/GNN/GNN_hole_2026/GNN_program"
LOG_FILE="${LOG_DIR}/training_log_recommended_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/training_pid_recommended_${TIMESTAMP}.txt"

cd /home/nishioka/GNN/GNN_hole_2026/GNN_program

echo "=========================================="
echo "Training with 4 GPUs (改善3点セット)"
echo "Started at $(date)"
echo "=========================================="
echo "Training configuration (改善3点セット):"
echo "  1. Early stopping: Macro F1基準（val_lossは監視のみ）"
echo "  2. Class Frequency Sampler: 1/sqrt(freq)ベースの重み付け"
echo "  - Batch Size: 64, Hidden Channels: 16"
echo "  - Learning Rate: 0.002 (適切な値に調整)"
echo "  - OneCycleLR, Dropout: 0.1, Edge Drop: 0.01"
echo "  - Epochs: 2000, Patience: 300"
echo "=========================================="
echo "Log file: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
echo "=========================================="

nohup torchrun --nproc_per_node=4 GNN_zscore_sub.py \
  --epochs 2000 \
  --learning_rate 0.002 \
  --batch_size 64 \
  --hidden_channels 16 \
  --patience 300 \
  --use_onecycle \
  --use_class_frequency_sampler \
  --dropout 0.1 \
  --edge_drop_prob 0.01 \
  > "${LOG_FILE}" 2>&1 &

TRAINING_PID=$!
echo $TRAINING_PID > "${PID_FILE}"

echo "Training started with PID: ${TRAINING_PID}"
echo "To check progress: tail -f ${LOG_FILE}"
echo "To stop training: kill ${TRAINING_PID}"
