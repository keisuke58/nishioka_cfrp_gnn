#!/usr/bin/env python3
"""
複数の実行結果を比較するツール

Usage:
    python tools/compare_runs.py --run_dirs runs/run1 runs/run2 runs/run3 --output_dir reports/comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_run_metrics(run_dir: Path) -> Dict[str, Any]:
    """実行ディレクトリからメトリクスを読み込む"""
    metrics_path = run_dir / 'metrics.json'
    summary_path = run_dir / 'meta' / 'summary.json'
    
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics['metrics'] = json.load(f)
    
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            metrics['summary'] = json.load(f)
    
    metrics['run_id'] = run_dir.name
    metrics['run_dir'] = str(run_dir)
    
    return metrics


def create_comparison_table(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    """比較テーブルを作成"""
    rows = []
    
    for run in runs:
        summary = run.get('summary', {})
        test_metrics = run.get('metrics', {}).get('test', {})
        best_metrics = run.get('metrics', {}).get('best', {})
        
        row = {
            'Run ID': run.get('run_id', 'Unknown'),
            'Best Macro F1': best_metrics.get('macro_f1'),
            'Best Epoch': best_metrics.get('epoch'),
            'Test Accuracy': test_metrics.get('accuracy'),
            'Test Macro F1': test_metrics.get('macro_f1'),
            'Test Weighted F1': test_metrics.get('weighted_f1'),
            'Test Balanced Acc': test_metrics.get('balanced_accuracy'),
            'Test MCC': test_metrics.get('mcc'),
            'Exit Code': summary.get('exit_code'),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_comparison(runs: List[Dict[str, Any]], output_dir: Path):
    """比較グラフを作成"""
    # メトリクスの準備
    run_ids = [r.get('run_id', 'Unknown') for r in runs]
    test_f1s = [r.get('metrics', {}).get('test', {}).get('macro_f1') for r in runs]
    test_accs = [r.get('metrics', {}).get('test', {}).get('accuracy') for r in runs]
    best_f1s = [r.get('metrics', {}).get('best', {}).get('macro_f1') for r in runs]
    
    # None値を除外
    valid_indices = [i for i, v in enumerate(test_f1s) if v is not None]
    if not valid_indices:
        print("[WARN] No valid test metrics found for comparison")
        return
    
    run_ids_valid = [run_ids[i] for i in valid_indices]
    test_f1s_valid = [test_f1s[i] for i in valid_indices]
    test_accs_valid = [test_accs[i] for i in valid_indices]
    best_f1s_valid = [best_f1s[i] for i in valid_indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # F1スコアの比較
    x_pos = range(len(run_ids_valid))
    width = 0.35
    axes[0].bar([x - width/2 for x in x_pos], test_f1s_valid, width, 
                label='Test Macro F1', alpha=0.7)
    axes[0].bar([x + width/2 for x in x_pos], best_f1s_valid, width,
                label='Best Macro F1', alpha=0.7)
    axes[0].set_xlabel('Run ID', fontsize=12)
    axes[0].set_ylabel('F1 Score', fontsize=12)
    axes[0].set_title('Macro F1 Score Comparison', fontsize=14)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(run_ids_valid, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 精度の比較
    axes[1].bar(x_pos, test_accs_valid, alpha=0.7, color='green')
    axes[1].set_xlabel('Run ID', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Test Accuracy Comparison', fontsize=14)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(run_ids_valid, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'run_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def plot_learning_curves_comparison(runs: List[Dict[str, Any]], output_dir: Path):
    """学習曲線の比較"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for run in runs:
        run_id = run.get('run_id', 'Unknown')
        epochs_data = run.get('metrics', {}).get('epochs', [])
        
        if not epochs_data:
            continue
        
        epoch_nums = [e['epoch'] for e in epochs_data]
        val_losses = [e.get('val_loss') for e in epochs_data]
        macro_f1s = [e.get('macro_f1') for e in epochs_data]
        
        axes[0].plot(epoch_nums, val_losses, label=f'{run_id} (Val Loss)', alpha=0.7)
        axes[1].plot(epoch_nums, macro_f1s, label=f'{run_id} (Macro F1)', alpha=0.7)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Validation Loss Comparison', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Macro F1', fontsize=12)
    axes[1].set_title('Macro F1 Score Comparison', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'learning_curves_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare multiple training runs")
    parser.add_argument('--run_dirs', type=str, nargs='+', required=True, 
                       help='List of run directories to compare')
    parser.add_argument('--output_dir', type=str, default='reports/comparison',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Load metrics from all runs
    runs = []
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str).expanduser().resolve()
        if not run_dir.exists():
            print(f"[WARN] Run directory not found: {run_dir}")
            continue
        
        metrics = load_run_metrics(run_dir)
        if metrics:
            runs.append(metrics)
            print(f"[INFO] Loaded metrics from: {run_dir}")
    
    if not runs:
        print("[ERR] No valid runs found")
        return 1
    
    print(f"[INFO] Comparing {len(runs)} runs")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison table
    df = create_comparison_table(runs)
    
    # Save comparison table
    csv_path = output_dir / 'comparison_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved: {csv_path}")
    
    # Save as JSON for programmatic access
    json_path = output_dir / 'comparison_table.json'
    df.to_json(json_path, orient='records', indent=2)
    print(f"[INFO] Saved: {json_path}")
    
    # Generate visualizations
    plot_comparison(runs, output_dir)
    plot_learning_curves_comparison(runs, output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    print(df.to_string(index=False))
    
    print(f"\n[INFO] Comparison complete. Output directory: {output_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
