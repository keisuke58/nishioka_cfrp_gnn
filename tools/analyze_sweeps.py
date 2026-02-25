#!/usr/bin/env python3
"""
ハイパーパラメータスイープ結果の分析と可視化ツール

Usage:
    python tools/analyze_sweeps.py --sweep_csv runs/_sweeps/sweep_lr_*.csv
    python tools/analyze_sweeps.py --sweep_dir runs/_sweeps --output_dir reports
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional
from typing import Any, Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def load_sweep_csv(csv_path: Path) -> pd.DataFrame:
    """スイープCSVを読み込む"""
    df = pd.read_csv(csv_path)
    # 数値列の変換
    numeric_cols = ['lr', 'epochs', 'best_macro_f1', 'exit_code']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def find_best_model(df: pd.DataFrame, metric: str = 'best_macro_f1') -> Dict[str, Any]:
    """最良モデルを特定"""
    if metric not in df.columns:
        return {}
    
    # 成功した実行のみを考慮（exit_code == 0）
    successful = df[df['exit_code'] == 0].copy() if 'exit_code' in df.columns else df.copy()
    
    if successful.empty:
        return {}
    
    best_idx = successful[metric].idxmax()
    best = successful.loc[best_idx].to_dict()
    
    return {
        'run_id': best.get('run_id', ''),
        'run_dir': best.get('run_dir', ''),
        'lr': best.get('lr', None),
        'epochs': best.get('epochs', None),
        'best_macro_f1': best.get('best_macro_f1', None),
        'dataset_tag': best.get('dataset_tag', ''),
    }


def plot_learning_rate_comparison(df: pd.DataFrame, output_dir: Path):
    """学習率の比較グラフを作成"""
    successful = df[df['exit_code'] == 0].copy() if 'exit_code' in df.columns else df.copy()
    
    if successful.empty or 'lr' not in successful.columns or 'best_macro_f1' not in successful.columns:
        print("[WARN] Not enough data for LR comparison plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot: LR vs F1
    axes[0].scatter(successful['lr'], successful['best_macro_f1'], alpha=0.6, s=100)
    axes[0].set_xlabel('Learning Rate', fontsize=12)
    axes[0].set_ylabel('Best Macro F1', fontsize=12)
    axes[0].set_title('Learning Rate vs Best Macro F1', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    # Box plot: LR groups
    if len(successful['lr'].unique()) > 1:
        successful.boxplot(column='best_macro_f1', by='lr', ax=axes[1])
        axes[1].set_xlabel('Learning Rate', fontsize=12)
        axes[1].set_ylabel('Best Macro F1', fontsize=12)
        axes[1].set_title('Best Macro F1 Distribution by Learning Rate', fontsize=14)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'lr_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def plot_hyperparameter_heatmap(df: pd.DataFrame, output_dir: Path):
    """ハイパーパラメータのヒートマップを作成"""
    successful = df[df['exit_code'] == 0].copy() if 'exit_code' in df.columns else df.copy()
    
    if successful.empty or 'best_macro_f1' not in successful.columns:
        print("[WARN] Not enough data for heatmap")
        return
    
    # 数値列のみを選択
    numeric_cols = ['lr', 'epochs', 'best_macro_f1']
    available_cols = [col for col in numeric_cols if col in successful.columns]
    
    if len(available_cols) < 2:
        print("[WARN] Not enough numeric columns for heatmap")
        return
    
    corr = successful[available_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Hyperparameter Correlation Matrix', fontsize=14, pad=20)
    plt.tight_layout()
    
    output_path = output_dir / 'hyperparameter_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def generate_summary_report(df: pd.DataFrame, best_model: Dict[str, Any], output_dir: Path):
    """サマリーレポートを生成"""
    successful = df[df['exit_code'] == 0].copy() if 'exit_code' in df.columns else df.copy()
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'total_runs': len(df),
        'successful_runs': len(successful),
        'failed_runs': len(df) - len(successful),
        'best_model': best_model,
    }
    
    if 'best_macro_f1' in successful.columns and not successful.empty:
        report['statistics'] = {
            'mean_f1': float(successful['best_macro_f1'].mean()),
            'std_f1': float(successful['best_macro_f1'].std()),
            'min_f1': float(successful['best_macro_f1'].min()),
            'max_f1': float(successful['best_macro_f1'].max()),
            'median_f1': float(successful['best_macro_f1'].median()),
        }
    
    if 'lr' in successful.columns and not successful.empty:
        report['learning_rate_stats'] = {
            'unique_lrs': sorted(successful['lr'].unique().tolist()),
            'best_lr': float(best_model.get('lr', 0)) if best_model.get('lr') else None,
        }
    
    output_path = output_dir / 'sweep_summary.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved: {output_path}")
    
    # テキストサマリーも出力
    txt_path = output_dir / 'sweep_summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Hyperparameter Sweep Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated at: {report['generated_at']}\n\n")
        f.write(f"Total runs: {report['total_runs']}\n")
        f.write(f"Successful runs: {report['successful_runs']}\n")
        f.write(f"Failed runs: {report['failed_runs']}\n\n")
        
        if 'statistics' in report:
            stats = report['statistics']
            f.write("F1 Score Statistics:\n")
            f.write(f"  Mean:   {stats['mean_f1']:.6f}\n")
            f.write(f"  Std:    {stats['std_f1']:.6f}\n")
            f.write(f"  Min:    {stats['min_f1']:.6f}\n")
            f.write(f"  Max:    {stats['max_f1']:.6f}\n")
            f.write(f"  Median: {stats['median_f1']:.6f}\n\n")
        
        if best_model:
            f.write("Best Model:\n")
            for key, value in best_model.items():
                f.write(f"  {key}: {value}\n")
    
    print(f"[INFO] Saved: {txt_path}")


def load_optuna_study(study_db_path: str, study_name: Optional[str] = None) -> pd.DataFrame:
    """Load trials from an Optuna SQLite study database as a DataFrame.

    Args:
        study_db_path: Path to the .db file (SQLite)
        study_name: Study name (if None, uses the first study found)

    Returns:
        DataFrame with columns similar to sweep CSVs for unified analysis
    """
    try:
        import optuna
    except ImportError:
        print("[ERR] optuna is required for --optuna_db. pip install optuna")
        return pd.DataFrame()

    storage = f"sqlite:///{Path(study_db_path).resolve()}"
    studies = optuna.study.get_all_study_summaries(storage)
    if not studies:
        print(f"[WARN] No studies found in {study_db_path}")
        return pd.DataFrame()

    if study_name is None:
        study_name = studies[0].study_name
        print(f"[INFO] Using study: {study_name}")

    study = optuna.load_study(study_name=study_name, storage=storage)
    df = study.trials_dataframe()

    if df.empty:
        return df

    # Rename columns to match sweep CSV format for unified analysis
    rename_map = {}
    for col in df.columns:
        if col.startswith("params_"):
            rename_map[col] = col.replace("params_", "")
    df = df.rename(columns=rename_map)

    if "value" in df.columns:
        df["best_macro_f1"] = df["value"]
    if "number" in df.columns:
        df["run_id"] = df["number"].apply(lambda n: f"trial_{n:03d}")

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results")
    parser.add_argument('--sweep_csv', type=str, nargs='+', help='Path(s) to sweep CSV file(s)')
    parser.add_argument('--sweep_dir', type=str, help='Directory containing sweep CSV files')
    parser.add_argument('--optuna_db', type=str, help='Path to Optuna SQLite study database (.db)')
    parser.add_argument('--study_name', type=str, default=None, help='Optuna study name (default: first)')
    parser.add_argument('--output_dir', type=str, default='reports', help='Output directory for reports')
    parser.add_argument('--pattern', type=str, default='sweep_*.csv', help='File pattern for sweep CSVs')

    args = parser.parse_args()
    
    # Optuna DB mode
    if args.optuna_db:
        db_path = Path(args.optuna_db).expanduser().resolve()
        if not db_path.exists():
            print(f"[ERR] Optuna DB not found: {db_path}")
            return 1
        combined_df = load_optuna_study(str(db_path), args.study_name)
        if combined_df.empty:
            print("[ERR] No trials found in Optuna study")
            return 1
        print(f"[INFO] Loaded {len(combined_df)} trials from Optuna study")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_model = find_best_model(combined_df)
        if best_model:
            print(f"[INFO] Best model: {best_model['run_id']} (F1: {best_model.get('best_macro_f1', 'N/A')})")
        plot_learning_rate_comparison(combined_df, output_dir)
        plot_hyperparameter_heatmap(combined_df, output_dir)
        generate_summary_report(combined_df, best_model, output_dir)
        print(f"[INFO] Analysis complete. Output directory: {output_dir}")
        return 0

    # Collect CSV files
    csv_files: List[Path] = []
    if args.sweep_csv:
        for path_str in args.sweep_csv:
            path = Path(path_str).expanduser().resolve()
            if path.exists():
                csv_files.append(path)
            else:
                # Try glob pattern
                csv_files.extend(Path().glob(path_str))
    elif args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
        csv_files = list(sweep_dir.glob(args.pattern))
    else:
        parser.error("Either --sweep_csv, --sweep_dir, or --optuna_db must be provided")
    
    if not csv_files:
        print("[ERR] No CSV files found")
        return 1
    
    print(f"[INFO] Found {len(csv_files)} CSV file(s)")
    
    # Load and combine dataframes
    dfs = []
    for csv_file in csv_files:
        try:
            df = load_sweep_csv(csv_file)
            df['source_file'] = str(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed to load {csv_file}: {e}")
    
    if not dfs:
        print("[ERR] No valid CSV files loaded")
        return 1
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Loaded {len(combined_df)} total runs")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find best model
    best_model = find_best_model(combined_df)
    if best_model:
        print(f"[INFO] Best model: {best_model['run_id']} (F1: {best_model.get('best_macro_f1', 'N/A')})")
    
    # Generate visualizations
    plot_learning_rate_comparison(combined_df, output_dir)
    plot_hyperparameter_heatmap(combined_df, output_dir)
    
    # Generate summary report
    generate_summary_report(combined_df, best_model, output_dir)
    
    print(f"[INFO] Analysis complete. Output directory: {output_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
