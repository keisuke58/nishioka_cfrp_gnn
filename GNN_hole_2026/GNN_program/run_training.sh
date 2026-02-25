#!/bin/bash
# Training script with nohup and logging

# NCCL環境変数の設定（タイムアウト対策）
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=0
export NCCL_DEBUG=WARN
# NCCLタイムアウトを延長（デフォルトは600秒=10分）
export NCCL_TIMEOUT=1800  # 30分に延長

# タイムスタンプを取得
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ログファイル名
LOG_FILE="/home/nishioka/GNN/GNN_hole_2026/GNN_program/training_log_${TIMESTAMP}.log"

# 実行コマンド
COMMAND="torchrun --standalone --nproc_per_node=4 /home/nishioka/GNN/GNN_hole_2026/GNN_program/GNN_zscore_sub.py --data_usage_ratio 0.5 --epochs 5000 --learning_rate 0.01 --patience 1000"

echo "Starting training at $(date)"
echo "Log file: ${LOG_FILE}"
echo "Command: ${COMMAND}"
echo "=========================================="

# nohupで実行（標準出力と標準エラー出力をログファイルに保存）
nohup ${COMMAND} > ${LOG_FILE} 2>&1 &

# プロセスIDを保存
PID=$!
echo "Training started with PID: ${PID}"
echo "PID: ${PID}" > "/home/nishioka/GNN/GNN_hole_2026/GNN_program/training_pid_${TIMESTAMP}.txt"
echo "To check progress: tail -f ${LOG_FILE}"
echo "To stop training: kill ${PID}"
