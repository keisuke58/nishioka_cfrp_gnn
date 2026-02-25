#!/bin/bash
# 評価専用スクリプトの実行例

# 最良モデルのパス
MODEL_PATH="/home/nishioka/GNN/runs/20260117_134228_nogit_dsNDF_ep1000_lr0p002_crossedge/outputs/GNN_model/19classmodel_hole_zscore/GATModel_20260117_134247_Best_Final.pth"

# Python環境のパス（必要に応じて変更）
PYTHON="/home/nishioka/miniconda3/envs/gnn_final_env/bin/python"

# 評価スクリプトの実行
$PYTHON /home/nishioka/GNN/GNN_hole_2026/GNN_program/evaluate_only.py \
    --model_path "$MODEL_PATH" \
    --hidden_channels 16 \
    --batch_size 32 \
    --dropout 0.1 \
    --edge_drop_prob 0.01 \
    --device cuda:0 \
    --use_layer_constraint \
    --layer_constraint_weight 1.0
