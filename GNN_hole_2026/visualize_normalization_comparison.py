"""
正規化結果を可視化して比較するスクリプト
Min-Max正規化版とZ-score標準化版を比較
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import random


def visualize_comparison(original_dir, minmax_dir, zscore_dir, output_dir, num_samples=50):
    """
    正規化結果を可視化して比較
    
    Args:
        original_dir: 元のデータディレクトリ（引き算後）
        minmax_dir: Min-Max正規化済みディレクトリ
        zscore_dir: Z-score標準化済みディレクトリ
        output_dir: 可視化結果の出力ディレクトリ
        num_samples: 可視化するファイル数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # すべてのファイルを取得
    all_files = sorted([f for f in os.listdir(original_dir) if f.endswith('.npy')])
    
    # サンプリング（ランダムに選択）
    if len(all_files) > num_samples:
        random.seed(42)
        sample_files = random.sample(all_files, num_samples)
    else:
        sample_files = all_files
    
    print(f"可視化対象ファイル数: {len(sample_files)}")
    
    # 各ファイルを可視化
    for filename in tqdm(sample_files, desc="可視化中"):
        try:
            # データを読み込む
            original_path = os.path.join(original_dir, filename)
            minmax_path = os.path.join(minmax_dir, filename)
            zscore_path = os.path.join(zscore_dir, filename)
            
            if not all([os.path.exists(p) for p in [original_path, minmax_path, zscore_path]]):
                continue
            
            original = np.load(original_path)
            minmax = np.load(minmax_path)
            zscore = np.load(zscore_path)
            
            # 可視化
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{filename}', fontsize=12, fontweight='bold')
            
            # 元のデータ（ヒストグラム）
            axes[0, 0].hist(original.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].set_title('Original (Subtracted)', fontsize=10)
            axes[0, 0].set_xlabel('Value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axvline(np.mean(original), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(original):.3f}')
            axes[0, 0].legend()
            
            # Min-Max正規化（ヒストグラム）
            axes[0, 1].hist(minmax.flatten(), bins=100, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('Min-Max Normalized [0, 1]', fontsize=10)
            axes[0, 1].set_xlabel('Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axvline(np.mean(minmax), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(minmax):.3f}')
            axes[0, 1].legend()
            
            # Z-score標準化（ヒストグラム）
            axes[1, 0].hist(zscore.flatten(), bins=100, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_title('Z-score Standardized (mean=0, std=1)', fontsize=10)
            axes[1, 0].set_xlabel('Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axvline(np.mean(zscore), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(zscore):.3f}')
            axes[1, 0].axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            axes[1, 0].legend()
            
            # 統計情報の比較
            axes[1, 1].axis('off')
            stats_text = f"""
Statistics Comparison:

Original (Subtracted):
  Min: {np.min(original):.6f}
  Max: {np.max(original):.6f}
  Mean: {np.mean(original):.6f}
  Std: {np.std(original):.6f}
  Median: {np.median(original):.6f}

Min-Max Normalized:
  Min: {np.min(minmax):.6f}
  Max: {np.max(minmax):.6f}
  Mean: {np.mean(minmax):.6f}
  Std: {np.std(minmax):.6f}
  Median: {np.median(minmax):.6f}

Z-score Standardized:
  Min: {np.min(zscore):.6f}
  Max: {np.max(zscore):.6f}
  Mean: {np.mean(zscore):.6f}
  Std: {np.std(zscore):.6f}
  Median: {np.median(zscore):.6f}
            """
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
                           verticalalignment='center', transform=axes[1, 1].transAxes)
            
            plt.tight_layout()
            
            # 保存
            output_filename = filename.replace('.npy', '_comparison.png')
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        except Exception as e:
            print(f"\nエラー: {filename} の可視化に失敗: {e}")
            continue
    
    print(f"\n可視化完了!")
    print(f"  出力ディレクトリ: {output_dir}")
    print(f"  可視化ファイル数: {len(sample_files)}")


def main():
    # ディレクトリパス
    original_dir = '/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect'
    minmax_dir = '/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_normalized'
    zscore_dir = '/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore'
    output_dir = '/home/nishioka/GNN/GNN_hole_2026/normalization_comparison_visualization'
    
    print("=" * 80)
    print("正規化結果の可視化スクリプト")
    print("=" * 80)
    print(f"\n元のデータ: {original_dir}")
    print(f"Min-Max正規化: {minmax_dir}")
    print(f"Z-score標準化: {zscore_dir}")
    print(f"出力ディレクトリ: {output_dir}\n")
    
    # 可視化を実行
    visualize_comparison(original_dir, minmax_dir, zscore_dir, output_dir, num_samples=50)


if __name__ == "__main__":
    main()
