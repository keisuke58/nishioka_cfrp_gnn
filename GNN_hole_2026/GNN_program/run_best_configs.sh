#!/bin/bash
# Best performing configurations (2026-01-13)
# 現在のところ、この2つの設定が良い結果を出しています

# NCCL環境変数の設定（タイムアウト対策）
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800  # 30分に延長

# タイムスタンプを取得
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 設定1: baseline安定
echo "=========================================="
echo "Configuration 1: Baseline Stable"
echo "=========================================="
LOG_FILE1="/home/nishioka/GNN/GNN_hole_2026/GNN_program/training_log_baseline_${TIMESTAMP}.log"
# GPUメモリに余裕があるため、バッチサイズとhidden_channelsを増やしています
COMMAND1="torchrun --nproc_per_node=4 /home/nishioka/GNN/GNN_hole_2026/GNN_program/GNN_zscore_sub.py \
  --use_onecycle \
  --dropout 0.1 \
  --edge_drop_prob 0.01 \
  --batch_size 128 \
  --hidden_channels 32"

echo "Starting baseline training at $(date)"
echo "Log file: ${LOG_FILE1}"
echo "Command: ${COMMAND1}"
echo "=========================================="

# nohupで実行
nohup ${COMMAND1} > ${LOG_FILE1} 2>&1 &
PID1=$!
echo "Baseline training started with PID: ${PID1}"
echo "PID: ${PID1}" > "/home/nishioka/GNN/GNN_hole_2026/GNN_program/training_pid_baseline_${TIMESTAMP}.txt"
echo "To check progress: tail -f ${LOG_FILE1}"
echo "To stop training: kill ${PID1}"
echo ""

# 設定2: minority sampler追加
echo "=========================================="
echo "Configuration 2: With Minority Sampler"
echo "=========================================="
LOG_FILE2="/home/nishioka/GNN/GNN_hole_2026/GNN_program/training_log_minority_${TIMESTAMP}.log"
# GPUメモリに余裕があるため、バッチサイズとhidden_channelsを増やしています
COMMAND2="torchrun --nproc_per_node=4 /home/nishioka/GNN/GNN_hole_2026/GNN_program/GNN_zscore_sub.py \
  --use_onecycle \
  --dropout 0.1 \
  --edge_drop_prob 0.01 \
  --use_minority_sampler \
  --minority_weight_power 1.0 \
  --batch_size 128 \
  --hidden_channels 32"

echo "Starting minority sampler training at $(date)"
echo "Log file: ${LOG_FILE2}"
echo "Command: ${COMMAND2}"
echo "=========================================="

# nohupで実行
nohup ${COMMAND2} > ${LOG_FILE2} 2>&1 &
PID2=$!
echo "Minority sampler training started with PID: ${PID2}"
echo "PID: ${PID2}" > "/home/nishioka/GNN/GNN_hole_2026/GNN_program/training_pid_minority_${TIMESTAMP}.txt"
echo "To check progress: tail -f ${LOG_FILE2}"
echo "To stop training: kill ${PID2}"
echo ""

echo "=========================================="
echo "Both configurations started successfully!"
echo "=========================================="
echo "Baseline PID: ${PID1}"
echo "Minority Sampler PID: ${PID2}"
