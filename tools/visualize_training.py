#!/usr/bin/env python3
"""
学習曲線とメトリクスの可視化ツール

Usage:
    python tools/visualize_training.py --run_dir runs/20250115_abc123
    python tools/visualize_training.py --metrics_json runs/20250115_abc123/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """メトリクスJSONを読み込む"""
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_learning_curves(metrics: Dict[str, Any], output_dir: Path):
    """学習曲線をプロット"""
    epochs = metrics.get('epochs', [])
    if not epochs:
        print("[WARN] No epoch data found")
        return
    
    # データの準備
    epoch_nums = [e['epoch'] for e in epochs]
    train_losses = [e.get('train_loss') for e in epochs]
    val_losses = [e.get('val_loss') for e in epochs]
    val_accs = [e.get('val_acc') for e in epochs]
    macro_f1s = [e.get('macro_f1') for e in epochs]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(epoch_nums, train_losses, label='Train Loss', alpha=0.7, linewidth=2)
    axes[0, 0].plot(epoch_nums, val_losses, label='Val Loss', alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epoch_nums, val_accs, label='Val Accuracy', color='green', alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Validation Accuracy', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Macro F1
    axes[1, 0].plot(epoch_nums, macro_f1s, label='Macro F1', color='purple', alpha=0.7, linewidth=2)
    if metrics.get('best', {}).get('epoch'):
        best_epoch = metrics['best']['epoch']
        best_f1 = metrics['best'].get('macro_f1')
        if best_f1:
            axes[1, 0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best (Epoch {best_epoch})')
            axes[1, 0].plot(best_epoch, best_f1, 'ro', markersize=10)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Macro F1', fontsize=12)
    axes[1, 0].set_title('Macro F1 Score', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined view
    ax2 = axes[1, 0].twinx()
    ax2.plot(epoch_nums, val_losses, label='Val Loss', color='orange', alpha=0.5, linestyle='--')
    ax2.set_ylabel('Validation Loss', fontsize=12, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Metrics comparison
    axes[1, 1].plot(epoch_nums, val_accs, label='Val Accuracy', alpha=0.7)
    axes[1, 1].plot(epoch_nums, macro_f1s, label='Macro F1', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Metrics Comparison', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'learning_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def plot_test_metrics(metrics: Dict[str, Any], output_dir: Path):
    """テストメトリクスを可視化"""
    test = metrics.get('test', {})
    if not test or all(v is None for v in test.values()):
        print("[WARN] No test metrics found")
        return
    
    # メトリクスの準備
    metric_names = []
    metric_values = []
    
    for key, value in test.items():
        if value is not None and isinstance(value, (int, float)):
            metric_names.append(key.replace('_', ' ').title())
            metric_values.append(value)
    
    if not metric_names:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(metric_names, metric_values, alpha=0.7, color='steelblue')
    
    # 値をバーの横に表示
    for i, (name, value) in enumerate(zip(metric_names, metric_values)):
        ax.text(value, i, f' {value:.4f}', va='center', fontsize=10)
    
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Test Metrics Summary', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = output_dir / 'test_metrics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument('--run_dir', type=str, help='Run directory containing metrics.json')
    parser.add_argument('--metrics_json', type=str, help='Path to metrics.json file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: same as run_dir or metrics_json dir)')
    
    args = parser.parse_args()
    
    # Determine metrics path
    if args.metrics_json:
        metrics_path = Path(args.metrics_json).expanduser().resolve()
    elif args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        metrics_path = run_dir / 'metrics.json'
    else:
        parser.error("Either --run_dir or --metrics_json must be provided")
    
    if not metrics_path.exists():
        print(f"[ERR] Metrics file not found: {metrics_path}")
        return 1
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = metrics_path.parent / 'visualizations'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    print(f"[INFO] Loading metrics from: {metrics_path}")
    metrics = load_metrics(metrics_path)
    
    # Generate visualizations
    plot_learning_curves(metrics, output_dir)
    plot_test_metrics(metrics, output_dir)
    
    print(f"[INFO] Visualizations saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
