#!/bin/bash
# アブレーション実験のスイープスクリプト

# Cross-edge ON/OFFの比較
echo "Running Cross-edge ablation..."
python tools/run_ablation_experiment.py \
    --experiment cross_edge_off \
    --split_type iid \
    --output_dir ./ablation_results

python tools/run_ablation_experiment.py \
    --experiment cross_edge_on \
    --split_type iid \
    --cross_edges \
    --cross_edge_k 1 \
    --output_dir ./ablation_results

# OOD分割の比較
echo "Running OOD split ablation..."
for split_type in iid defect_size defect_ratio layer; do
    echo "  Testing split_type: $split_type"
    python tools/run_ablation_experiment.py \
        --experiment "ood_split_${split_type}" \
        --split_type "$split_type" \
        --output_dir ./ablation_results
done

# Cross-edge k値の比較
echo "Running Cross-edge k ablation..."
for k in 1 3 5; do
    echo "  Testing k=$k"
    python tools/run_ablation_experiment.py \
        --experiment "cross_edge_k${k}" \
        --split_type iid \
        --cross_edges \
        --cross_edge_k "$k" \
        --output_dir ./ablation_results
done

echo "Ablation sweep completed!"
