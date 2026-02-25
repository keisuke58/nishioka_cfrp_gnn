#!/usr/bin/env python3
"""
ノイズデータを可視化するスクリプト

元のデータとノイズデータを比較して可視化します
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import matplotlib.colors as mcolors

def load_coordinates():
    """座標データを読み込む"""
    x_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_x_2layer.npy")
    y_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_y_2layer.npy")
    z_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_z_2layer.npy")
    return x_coords[:13942], y_coords[:13942], z_coords[:13942]

def visualize_comparison(original_data, noise_data, x_coords, y_coords, z_coords, 
                        output_path, title_suffix=""):
    """
    元のデータとノイズデータを比較して可視化
    
    Args:
        original_data: 元のデータ（1次元配列）
        noise_data: ノイズデータ（1次元配列）
        x_coords, y_coords, z_coords: 座標データ
        output_path: 出力ファイルのパス
        title_suffix: タイトルのサフィックス
    """
    # データの統計情報を計算
    diff = noise_data - original_data
    
    # 図を作成（3行2列）
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    fig.suptitle(f'Noise Data Comparison{title_suffix}', fontsize=16, fontweight='bold')
    
    # 1行目: 元のデータ
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(x_coords, y_coords, c=original_data, cmap='viridis', 
                          s=1, alpha=0.6, vmin=np.percentile(original_data, 1), 
                          vmax=np.percentile(original_data, 99))
    ax1.set_title('Original Data (XY projection)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, label='Value')
    
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(x_coords, z_coords, c=original_data, cmap='viridis', 
                          s=1, alpha=0.6, vmin=np.percentile(original_data, 1), 
                          vmax=np.percentile(original_data, 99))
    ax2.set_title('Original Data (XZ projection)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Z coordinate')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, label='Value')
    
    # 2行目: ノイズデータ
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(x_coords, y_coords, c=noise_data, cmap='viridis', 
                          s=1, alpha=0.6, vmin=np.percentile(noise_data, 1), 
                          vmax=np.percentile(noise_data, 99))
    ax3.set_title('Noise Data (XY projection)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X coordinate')
    ax3.set_ylabel('Y coordinate')
    ax3.set_aspect('equal')
    plt.colorbar(scatter3, ax=ax3, label='Value')
    
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(x_coords, z_coords, c=noise_data, cmap='viridis', 
                          s=1, alpha=0.6, vmin=np.percentile(noise_data, 1), 
                          vmax=np.percentile(noise_data, 99))
    ax4.set_title('Noise Data (XZ projection)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('X coordinate')
    ax4.set_ylabel('Z coordinate')
    ax4.set_aspect('equal')
    plt.colorbar(scatter4, ax=ax4, label='Value')
    
    # 3行目: 差分
    ax5 = axes[2, 0]
    scatter5 = ax5.scatter(x_coords, y_coords, c=diff, cmap='RdBu_r', 
                          s=1, alpha=0.6, vmin=np.percentile(diff, 1), 
                          vmax=np.percentile(diff, 99))
    ax5.set_title('Difference (Noise - Original) (XY projection)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('X coordinate')
    ax5.set_ylabel('Y coordinate')
    ax5.set_aspect('equal')
    plt.colorbar(scatter5, ax=ax5, label='Difference')
    
    ax6 = axes[2, 1]
    scatter6 = ax6.scatter(x_coords, z_coords, c=diff, cmap='RdBu_r', 
                          s=1, alpha=0.6, vmin=np.percentile(diff, 1), 
                          vmax=np.percentile(diff, 99))
    ax6.set_title('Difference (Noise - Original) (XZ projection)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('X coordinate')
    ax6.set_ylabel('Z coordinate')
    ax6.set_aspect('equal')
    plt.colorbar(scatter6, ax=ax6, label='Difference')
    
    # 統計情報をテキストで追加
    stats_text = (
        f"Original: mean={original_data.mean():.6f}, std={original_data.std():.6f}\n"
        f"Noise: mean={noise_data.mean():.6f}, std={noise_data.std():.6f}\n"
        f"Difference: mean={diff.mean():.6f}, std={diff.std():.6f}\n"
        f"Max diff: {np.abs(diff).max():.6f}"
    )
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")

def visualize_histogram_comparison(original_data, noise_data, diff, output_path):
    """
    ヒストグラムでデータの分布を比較
    
    Args:
        original_data: 元のデータ
        noise_data: ノイズデータ
        diff: 差分データ
        output_path: 出力ファイルのパス
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Data Distribution Comparison', fontsize=16, fontweight='bold')
    
    # 元のデータのヒストグラム
    axes[0].hist(original_data, bins=100, alpha=0.7, color='blue', label='Original')
    axes[0].set_title('Original Data Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ノイズデータのヒストグラム
    axes[1].hist(noise_data, bins=100, alpha=0.7, color='green', label='Noise')
    axes[1].set_title('Noise Data Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 差分のヒストグラム
    axes[2].hist(diff, bins=100, alpha=0.7, color='red', label='Difference')
    axes[2].set_title('Difference Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Difference')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize noise data")
    parser.add_argument("--original_data_dir", type=str,
                       default="/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore",
                       help="Directory containing original data")
    parser.add_argument("--noise_data_dir", type=str,
                       default="/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore_noise",
                       help="Directory containing noise data")
    parser.add_argument("--output_dir", type=str,
                       default="/home/nishioka/GNN/GNN_hole_2026/noise_data_visualization",
                       help="Output directory for visualizations")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"],
                       default="train", help="Which split to visualize")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to visualize (default: 10)")
    parser.add_argument("--max_nodes", type=int, default=13942,
                       help="Maximum number of nodes (default: 13942)")
    
    args = parser.parse_args()
    
    # ディレクトリの存在確認
    original_dir = os.path.join(args.original_data_dir, args.split)
    noise_dir = os.path.join(args.noise_data_dir, args.split)
    
    if not os.path.exists(original_dir):
        print(f"Error: Original data directory does not exist: {original_dir}")
        return
    
    if not os.path.exists(noise_dir):
        print(f"Error: Noise data directory does not exist: {noise_dir}")
        return
    
    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 座標データを読み込む
    print("Loading coordinates...")
    x_coords, y_coords, z_coords = load_coordinates()
    
    # データファイルを取得
    original_files = [f for f in os.listdir(original_dir) 
                     if f.startswith("Defect_L") and f.endswith(".npy")]
    noise_files = [f for f in os.listdir(noise_dir) 
                  if f.startswith("Defect_L") and f.endswith(".npy")]
    
    # 共通のファイルを取得
    common_files = sorted(list(set(original_files) & set(noise_files)))
    
    print(f"\n=== Dataset Summary ===")
    print(f"Original files: {len(original_files)}")
    print(f"Noise files: {len(noise_files)}")
    print(f"Common files: {len(common_files)}")
    
    if len(common_files) == 0:
        print("Error: No common files found")
        return
    
    # サンプル数を制限
    sample_files = common_files[:min(args.num_samples, len(common_files))]
    
    print(f"\n=== Visualizing {len(sample_files)} samples ===")
    
    # 各ファイルを可視化
    for data_file in tqdm(sample_files, desc="Visualizing"):
        original_path = os.path.join(original_dir, data_file)
        noise_path = os.path.join(noise_dir, data_file)
        
        try:
            # データを読み込む
            original_data = np.load(original_path)[:args.max_nodes]
            noise_data = np.load(noise_path)[:args.max_nodes]
            
            # データの形状を確認
            if len(original_data.shape) > 1:
                original_data = original_data.flatten()[:args.max_nodes]
            if len(noise_data.shape) > 1:
                noise_data = noise_data.flatten()[:args.max_nodes]
            
            # ファイル名からベース名を取得
            base_name = os.path.splitext(data_file)[0]
            
            # 空間可視化
            output_path = os.path.join(args.output_dir, f"{base_name}_noise_comparison.png")
            visualize_comparison(original_data, noise_data, x_coords, y_coords, z_coords,
                               output_path, title_suffix=f" - {base_name}")
            
            # ヒストグラム可視化
            diff = noise_data - original_data
            hist_output_path = os.path.join(args.output_dir, f"{base_name}_noise_histogram.png")
            visualize_histogram_comparison(original_data, noise_data, diff, hist_output_path)
            
        except Exception as e:
            print(f"Error processing {data_file}: {e}")
            continue
    
    print(f"\n=== Summary ===")
    print(f"Visualizations saved to: {args.output_dir}")
    print(f"Total visualizations: {len(sample_files) * 2}")

if __name__ == "__main__":
    main()
