#!/usr/bin/env python3
"""
アブレーション実験スクリプト

Cross-edge ON/OFFや異なるOOD分割方法を比較するためのスクリプト
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# プロジェクトルートをパスに追加
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from gnn_common.integration_utils import (
    apply_ood_split,
    prepare_data_with_cross_edges,
    evaluate_with_localization_metrics,
    create_model_with_cross_edges,
    load_config_for_splitting,
    load_config_for_cross_edges
)
from gnn_common.data_utils import create_data_label_pairs, visualize_split_statistics


def run_ablation_experiment(
    experiment_name: str,
    split_type: str = 'iid',
    use_cross_edges: bool = False,
    cross_edge_k: int = 1,
    output_dir: str = "./ablation_results",
    **kwargs
) -> Dict[str, Any]:
    """
    アブレーション実験を実行
    
    Args:
        experiment_name: 実験名
        split_type: 分割タイプ
        use_cross_edges: Cross-edgeを使用するか
        cross_edge_k: Cross-edgeのk値
        output_dir: 出力ディレクトリ
        **kwargs: その他のパラメータ
    
    Returns:
        実験結果の辞書
    """
    print(f"\n{'='*80}")
    print(f"Running Ablation Experiment: {experiment_name}")
    print(f"{'='*80}")
    print(f"  Split type: {split_type}")
    print(f"  Cross-edges: {'ON' if use_cross_edges else 'OFF'}")
    if use_cross_edges:
        print(f"  Cross-edge k: {cross_edge_k}")
    print(f"{'='*80}\n")
    
    # ここに実際の実験コードを追加
    # 現在はプレースホルダー
    
    results = {
        'experiment_name': experiment_name,
        'split_type': split_type,
        'use_cross_edges': use_cross_edges,
        'cross_edge_k': cross_edge_k if use_cross_edges else None,
        'timestamp': datetime.now().isoformat(),
        'status': 'completed'  # または 'failed'
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument(
        '--experiment',
        type=str,
        default='cross_edge_ablation',
        help='Experiment name'
    )
    parser.add_argument(
        '--split_type',
        type=str,
        default='iid',
        choices=['iid', 'defect_size', 'defect_ratio', 'layer'],
        help='Data split type'
    )
    parser.add_argument(
        '--cross_edges',
        action='store_true',
        help='Use cross-edges'
    )
    parser.add_argument(
        '--cross_edge_k',
        type=int,
        default=1,
        help='Cross-edge k value'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./ablation_results',
        help='Output directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Config file path'
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 実験を実行
    results = run_ablation_experiment(
        experiment_name=args.experiment,
        split_type=args.split_type,
        use_cross_edges=args.cross_edges,
        cross_edge_k=args.cross_edge_k,
        output_dir=str(output_dir),
        config_path=args.config
    )
    
    # 結果を保存
    output_file = output_dir / f"{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[INFO] Results saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
