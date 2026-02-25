#!/bin/bash
# Baseline安定設定
# 現在のところ良い結果を出している設定

# NCCL環境変数の設定（タイムアウト対策）
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800  # 30分に延長

# タイムスタンプを取得
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ログファイル名
LOG_FILE="/home/nishioka/GNN/GNN_hole_2026/GNN_program/training_log_baseline_${TIMESTAMP}.log"

# 実行コマンド
# GPUメモリに余裕があるため、バッチサイズとhidden_channelsを増やしています
COMMAND="torchrun --nproc_per_node=4 /home/nishioka/GNN/GNN_hole_2026/GNN_program/GNN_zscore_sub.py \
  --use_onecycle \
  --dropout 0.1 \
  --edge_drop_prob 0.01 \
  --batch_size 128 \
  --hidden_channels 32"

echo "Starting baseline training at $(date)"
echo "Log file: ${LOG_FILE}"
echo "Command: ${COMMAND}"
echo "=========================================="

# nohupで実行（標準出力と標準エラー出力をログファイルに保存）
nohup ${COMMAND} > ${LOG_FILE} 2>&1 &

# プロセスIDを保存
PID=$!
echo "Training started with PID: ${PID}"
echo "PID: ${PID}" > "/home/nishioka/GNN/GNN_hole_2026/GNN_program/training_pid_baseline_${TIMESTAMP}.txt"
echo "To check progress: tail -f ${LOG_FILE}"
echo "To stop training: kill ${PID}"
