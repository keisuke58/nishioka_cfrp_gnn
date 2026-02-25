#!/usr/bin/env python3
"""
欠陥なしデータとラベルを可視化するスクリプト
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

def visualize_data_spatial(data, x_coords, y_coords, z_coords, output_path, title="Defect-Free Data"):
    """
    データの空間分布を可視化
    
    Args:
        data: データ（1次元配列）
        x_coords, y_coords, z_coords: 座標データ
        output_path: 出力ファイルのパス
        title: タイトル
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # XY投影
    ax1 = axes[0]
    scatter1 = ax1.scatter(x_coords, y_coords, c=data, cmap='viridis', 
                          s=1, alpha=0.6, vmin=np.percentile(data, 1), 
                          vmax=np.percentile(data, 99))
    ax1.set_title('XY Projection', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, label='Value')
    
    # XZ投影
    ax2 = axes[1]
    scatter2 = ax2.scatter(x_coords, z_coords, c=data, cmap='viridis', 
                          s=1, alpha=0.6, vmin=np.percentile(data, 1), 
                          vmax=np.percentile(data, 99))
    ax2.set_title('XZ Projection', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Z coordinate')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, label='Value')
    
    # 統計情報をテキストで追加
    stats_text = (
        f"Mean: {data.mean():.6f}\n"
        f"Std: {data.std():.6f}\n"
        f"Min: {data.min():.6f}\n"
        f"Max: {data.max():.6f}"
    )
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")

def visualize_data_histogram(data, output_path, title="Defect-Free Data Distribution"):
    """
    データのヒストグラムを可視化
    
    Args:
        data: データ（1次元配列）
        output_path: 出力ファイルのパス
        title: タイトル
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    ax.hist(data, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title('Data Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # 統計情報を追加
    stats_text = (
        f"Mean: {data.mean():.6f}\n"
        f"Std: {data.std():.6f}\n"
        f"Min: {data.min():.6f}\n"
        f"Max: {data.max():.6f}"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")

def visualize_label(label, x_coords, y_coords, z_coords, output_path, title="Defect-Free Label"):
    """
    ラベルの空間分布を可視化
    
    Args:
        label: ラベル（1次元配列または2次元配列）
        x_coords, y_coords, z_coords: 座標データ
        output_path: 出力ファイルのパス
        title: タイトル
    """
    # ラベルの形状を確認
    if len(label.shape) == 2:
        # One-hot形式の場合、クラスIDに変換
        label_class = np.argmax(label, axis=1)
    else:
        label_class = label
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # クラス分布を確認
    unique_classes, counts = np.unique(label_class, return_counts=True)
    
    # XY投影
    ax1 = axes[0]
    scatter1 = ax1.scatter(x_coords, y_coords, c=label_class, cmap='tab10', 
                          s=1, alpha=0.6, vmin=0, vmax=18)
    ax1.set_title('XY Projection - Class Labels', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, label='Class ID')
    
    # XZ投影
    ax2 = axes[1]
    scatter2 = ax2.scatter(x_coords, z_coords, c=label_class, cmap='tab10', 
                          s=1, alpha=0.6, vmin=0, vmax=18)
    ax2.set_title('XZ Projection - Class Labels', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Z coordinate')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, label='Class ID')
    
    # クラス分布情報をテキストで追加
    class_info = "Class distribution:\n"
    for cls, count in zip(unique_classes, counts):
        percentage = (count / len(label_class)) * 100
        class_info += f"  Class {cls}: {count} ({percentage:.2f}%)\n"
    
    fig.text(0.5, 0.02, class_info, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"  Class distribution: {dict(zip(unique_classes, counts))}")

def visualize_label_histogram(label, output_path, title="Defect-Free Label Distribution"):
    """
    ラベルのヒストグラムを可視化
    
    Args:
        label: ラベル（1次元配列または2次元配列）
        output_path: 出力ファイルのパス
        title: タイトル
    """
    # ラベルの形状を確認
    if len(label.shape) == 2:
        # One-hot形式の場合、クラスIDに変換
        label_class = np.argmax(label, axis=1)
    else:
        label_class = label
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    unique_classes, counts = np.unique(label_class, return_counts=True)
    
    ax.bar(unique_classes, counts, alpha=0.7, color='green', edgecolor='black')
    ax.set_title('Class Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Count')
    ax.set_xticks(unique_classes)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 各バーの上に値を表示
    for cls, count in zip(unique_classes, counts):
        percentage = (count / len(label_class)) * 100
        ax.text(cls, count, f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize defect-free data and labels")
    parser.add_argument("--data_dir", type=str,
                       default="/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore_noise/train",
                       help="Directory containing defect-free data files")
    parser.add_argument("--label_dir", type=str,
                       default="/home/nishioka/GNN/GNN_hole_2026/all_19class_label",
                       help="Directory containing label files")
    parser.add_argument("--output_dir", type=str,
                       default="/home/nishioka/GNN/GNN_hole_2026/defect_free_visualization",
                       help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to visualize (default: 10)")
    parser.add_argument("--max_nodes", type=int, default=13942,
                       help="Maximum number of nodes (default: 13942)")
    
    args = parser.parse_args()
    
    # ディレクトリの存在確認
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
    
    if not os.path.exists(args.label_dir):
        print(f"Error: Label directory does not exist: {args.label_dir}")
        return
    
    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "labels"), exist_ok=True)
    
    # 座標データを読み込む
    print("Loading coordinates...")
    x_coords, y_coords, z_coords = load_coordinates()
    
    # データファイルを取得
    data_files = sorted([f for f in os.listdir(args.data_dir) 
                        if f.startswith("NoiseDefectFree_") and f.endswith(".npy")])
    
    print(f"\n=== Dataset Summary ===")
    print(f"Found {len(data_files)} data files")
    
    if len(data_files) == 0:
        print("Error: No data files found")
        return
    
    # サンプル数を制限
    sample_files = data_files[:min(args.num_samples, len(data_files))]
    
    print(f"\n=== Visualizing {len(sample_files)} samples ===")
    
    # 各ファイルを可視化
    for data_file in tqdm(sample_files, desc="Visualizing"):
        base_name = os.path.splitext(data_file)[0]
        data_path = os.path.join(args.data_dir, data_file)
        label_path = os.path.join(args.label_dir, f"{base_name}_19label.npy")
        
        try:
            # データを読み込む
            data = np.load(data_path)[:args.max_nodes]
            
            # データの形状を確認
            if len(data.shape) > 1:
                data = data.flatten()[:args.max_nodes]
            
            # データの可視化
            data_spatial_path = os.path.join(args.output_dir, "data", f"{base_name}_spatial.png")
            visualize_data_spatial(data, x_coords, y_coords, z_coords, 
                                  data_spatial_path, title=f"Defect-Free Data - {base_name}")
            
            data_hist_path = os.path.join(args.output_dir, "data", f"{base_name}_histogram.png")
            visualize_data_histogram(data, data_hist_path, 
                                    title=f"Defect-Free Data Distribution - {base_name}")
            
            # ラベルの可視化（ラベルファイルが存在する場合）
            if os.path.exists(label_path):
                label = np.load(label_path)[:args.max_nodes]
                
                label_spatial_path = os.path.join(args.output_dir, "labels", f"{base_name}_label_spatial.png")
                visualize_label(label, x_coords, y_coords, z_coords, 
                              label_spatial_path, title=f"Defect-Free Label - {base_name}")
                
                label_hist_path = os.path.join(args.output_dir, "labels", f"{base_name}_label_histogram.png")
                visualize_label_histogram(label, label_hist_path, 
                                         title=f"Defect-Free Label Distribution - {base_name}")
            else:
                print(f"  Warning: Label file not found: {label_path}")
            
        except Exception as e:
            print(f"Error processing {data_file}: {e}")
            continue
    
    # 全体の統計を可視化（最初の10サンプルを使用）
    print(f"\n=== Creating overall statistics visualization ===")
    all_data = []
    all_labels = []
    
    for data_file in sample_files[:min(10, len(sample_files))]:
        base_name = os.path.splitext(data_file)[0]
        data_path = os.path.join(args.data_dir, data_file)
        label_path = os.path.join(args.label_dir, f"{base_name}_19label.npy")
        
        try:
            data = np.load(data_path)[:args.max_nodes]
            if len(data.shape) > 1:
                data = data.flatten()[:args.max_nodes]
            all_data.append(data)
            
            if os.path.exists(label_path):
                label = np.load(label_path)[:args.max_nodes]
                if len(label.shape) == 2:
                    label_class = np.argmax(label, axis=1)
                else:
                    label_class = label
                all_labels.append(label_class)
        except Exception as e:
            continue
    
    if len(all_data) > 0:
        all_data = np.concatenate(all_data)
        
        # 全体のデータ分布
        overall_data_path = os.path.join(args.output_dir, "overall_data_distribution.png")
        visualize_data_histogram(all_data, overall_data_path, 
                                title="Overall Defect-Free Data Distribution")
        
        if len(all_labels) > 0:
            all_labels = np.concatenate(all_labels)
            
            # 全体のラベル分布
            overall_label_path = os.path.join(args.output_dir, "overall_label_distribution.png")
            visualize_label_histogram(all_labels, overall_label_path, 
                                     title="Overall Defect-Free Label Distribution")
    
    print(f"\n=== Summary ===")
    print(f"Visualizations saved to: {args.output_dir}")
    print(f"  - Data visualizations: {args.output_dir}/data/")
    print(f"  - Label visualizations: {args.output_dir}/labels/")
    print(f"  - Overall statistics: {args.output_dir}/")

if __name__ == "__main__":
    main()
