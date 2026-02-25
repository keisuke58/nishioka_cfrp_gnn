#!/bin/bash
# ノイズあり欠陥なしデータセット用の学習スクリプト（おすすめ設定）
# データセット: ノイズあり欠陥なし5000個 + 欠陥あり5000個
# OneCycleLR + Sampler（logit_adjustはデフォルト無効）

# NCCL環境変数の設定（タイムアウト対策）
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800  # 30分に延長

# タイムスタンプを取得
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ログファイル名
LOG_DIR="/home/nishioka/GNN/GNN_hole_2026/GNN_program"
LOG_FILE="${LOG_DIR}/training_log_noise_defect_free_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/training_pid_noise_defect_free_${TIMESTAMP}.txt"

cd /home/nishioka/GNN/GNN_hole_2026/GNN_program

# 実行コマンド（改善3点セット適用）
# 1. Early stopping: Macro F1基準
# 2. Sampler: 1/sqrt(freq)ベースのクラス頻度サンプラー
COMMAND="torchrun --nproc_per_node=4 GNN_zscore_sub_noise_defect_free.py \
  --use_onecycle \
  --dropout 0.1 \
  --edge_drop_prob 0.01 \
  --use_class_frequency_sampler \
  --batch_size 64 \
  --hidden_channels 16 \
  --epochs 2000 \
  --patience 300"

echo "=========================================="
echo "Training with Noise Defect-Free Dataset"
echo "Started at $(date)"
echo "=========================================="
echo "Dataset configuration:"
echo "  - Noise defect-free samples: 5000"
echo "  - Defect samples: 5000"
echo "  - Total: 10000 samples"
echo "  - Split: Train 7000 / Val 1500 / Test 1500"
echo ""
echo "Training configuration (改善3点セット):"
echo "  1. Early stopping: Macro F1基準（val_lossは監視のみ）"
echo "  2. Class Frequency Sampler: 1/sqrt(freq)ベースの重み付け"
echo "  - Batch Size: 64, Hidden Channels: 16"
echo "  - OneCycleLR, Dropout: 0.1, Edge Drop: 0.01"
echo "  - Epochs: 2000, Patience: 300"
echo "=========================================="
echo "Log file: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
echo "Command: ${COMMAND}"
echo "=========================================="

# nohupで実行（標準出力と標準エラー出力をログファイルに保存）
nohup ${COMMAND} > ${LOG_FILE} 2>&1 &

# プロセスIDを保存
PID=$!
echo $PID > "${PID_FILE}"

echo "Training started with PID: ${PID}"
echo "To check progress: tail -f ${LOG_FILE}"
echo "To stop training: kill ${PID}"
