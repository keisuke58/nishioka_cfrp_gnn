#!/bin/bash
# Test run with 1000 epochs for GNN_zscore_sub_hard0_stage2.py

torchrun --nproc_per_node=4 GNN_zscore_sub_hard0_stage2.py \
  --epochs 1500 \
  --batch_size 64 \
  --hidden_channels 16 \
  --learning_rate 0.01 \
  --use_onecycle \
  --onecycle_max_lr 0.01 \
  --data_usage_ratio 0.5 \
  --use_minority_sampler \
  --stage1_minority_weight_power 1.5 \
  --stage2_minority_weight_power 1.0 \
  --stage1_hard0_keep_ratio 1.0 \
  --stage2_hard0_keep_ratio 1.0 \
  --stage2_epoch 401 \
  --use_class_mean_loss