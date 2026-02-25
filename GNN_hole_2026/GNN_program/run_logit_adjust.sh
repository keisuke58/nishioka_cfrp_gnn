#!/bin/bash
# LogitAdjustLoss単体設定
# 優先度1の施策を試す

# 安全対策: 明示的に有効化しない限り実行しない（誤爆防止）
if [[ "${ENABLE_LOGIT_ADJUST:-0}" != "1" ]]; then
  echo "[SAFEGUARD] run_logit_adjust.sh is disabled by default."
  echo "            To run anyway: ENABLE_LOGIT_ADJUST=1 bash run_logit_adjust.sh"
  exit 2
fi

# NCCL環境変数の設定（タイムアウト対策）
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800  # 30分に延長

# タイムスタンプを取得
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ログファイル名
LOG_FILE="/home/nishioka/GNN/GNN_hole_2026/GNN_program/training_log_logitadjust_${TIMESTAMP}.log"

# 実行コマンド（LogitAdjustLoss単体）
COMMAND="torchrun --nproc_per_node=4 /home/nishioka/GNN/GNN_hole_2026/GNN_program/GNN_zscore_sub.py \
  --use_onecycle \
  --dropout 0.1 \
  --edge_drop_prob 0.01 \
  --use_logit_adjust \
  --logit_adjust_tau 1.5 \
  --batch_size 128 \
  --hidden_channels 32"

echo "Starting LogitAdjustLoss training at $(date)"
echo "Log file: ${LOG_FILE}"
echo "Command: ${COMMAND}"
echo "=========================================="

# nohupで実行（標準出力と標準エラー出力をログファイルに保存）
nohup ${COMMAND} > ${LOG_FILE} 2>&1 &

# プロセスIDを保存
PID=$!
echo "Training started with PID: ${PID}"
echo "PID: ${PID}" > "/home/nishioka/GNN/GNN_hole_2026/GNN_program/training_pid_logitadjust_${TIMESTAMP}.txt"
echo "To check progress: tail -f ${LOG_FILE}"
echo "To stop training: kill ${PID}"
