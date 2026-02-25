#!/bin/bash
# おすすめ設定（強め）: Minority Sampler（logit_adjustはデフォルト無効）
# Macro F1向上を目指す（過度なlogit_adjustはNDF誤検出を増やすことがある）

# NCCL環境変数の設定（タイムアウト対策）
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800  # 30分に延長

# タイムスタンプを取得
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ログファイル名
LOG_FILE="/home/nishioka/GNN/GNN_hole_2026/GNN_program/training_log_recommended_strong_${TIMESTAMP}.log"

# 実行コマンド（Minority Sampler）
COMMAND="torchrun --nproc_per_node=4 /home/nishioka/GNN/GNN_hole_2026/GNN_program/GNN_zscore_sub.py \
  --use_onecycle \
  --dropout 0.1 \
  --edge_drop_prob 0.01 \
  --use_minority_sampler \
  --minority_weight_power 1.0 \
  --batch_size 128 \
  --hidden_channels 32"

echo "Starting recommended training (strong) at $(date)"
echo "Log file: ${LOG_FILE}"
echo "Command: ${COMMAND}"
echo "=========================================="
echo "Configuration:"
echo "  - Minority Sampler: マイノリティクラス重視"
echo "  - Batch Size: 128, Hidden Channels: 32"
echo "  - OneCycleLR, Dropout: 0.1, Edge Drop: 0.01"
echo "=========================================="

# nohupで実行（標準出力と標準エラー出力をログファイルに保存）
nohup ${COMMAND} > ${LOG_FILE} 2>&1 &

# プロセスIDを保存
PID=$!
echo "Training started with PID: ${PID}"
echo "PID: ${PID}" > "/home/nishioka/GNN/GNN_hole_2026/GNN_program/training_pid_recommended_strong_${TIMESTAMP}.txt"
echo "To check progress: tail -f ${LOG_FILE}"
echo "To stop training: kill ${PID}"
