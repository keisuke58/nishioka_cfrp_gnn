"""
すべての正規化フォルダからランダムにデータを選んで比較可視化するスクリプト
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re
import random
import argparse
from pathlib import Path
from tqdm import tqdm

# フォントの設定
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 10

# グリッド設定
num_cols = 57
num_rows = 125
vertices_per_layer = 6971
total_elements = num_cols * num_rows

# 穴領域の設定
hole_col_start = 26
hole_row_start1 = 51
hole_row_start2 = 77
hole_size_cols = 7
hole_size_rows1 = 7
hole_size_rows2 = 15

# 穴の位置計算
hole_elements = []
for r in range(hole_row_start1, hole_row_start1 + hole_size_rows1):
    for c in range(hole_col_start, hole_col_start + hole_size_cols):
        elem = r * num_cols + c
        hole_elements.append(elem)

for r in range(hole_row_start2, hole_row_start2 + hole_size_rows2):
    for c in range(hole_col_start, hole_col_start + hole_size_cols):
        elem = r * num_cols + c
        hole_elements.append(elem)

hole_elements = [node - 1 for node in hole_elements]


def process_file(file_path):
    """ファイルを読み込んで2層に分けて処理"""
    data = np.load(file_path)
    if len(data.shape) == 2:
        decoded_data = np.argmax(data, axis=1)
    elif len(data.shape) == 1:
        decoded_data = data
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    # ノード数を確認・調整
    if len(decoded_data) < vertices_per_layer * 2:
        decoded_data = np.pad(decoded_data, (0, vertices_per_layer * 2 - len(decoded_data)), mode='constant')
    elif len(decoded_data) > vertices_per_layer * 2:
        decoded_data = decoded_data[:vertices_per_layer * 2]

    layer1_data = decoded_data[:vertices_per_layer]
    layer2_data = decoded_data[vertices_per_layer:]

    layer1_full = np.empty(total_elements)
    layer1_full[:] = np.nan
    layer2_full = np.empty(total_elements)
    layer2_full[:] = np.nan

    data_index = 0
    for i in range(total_elements):
        if i not in hole_elements and data_index < vertices_per_layer:
            layer1_full[i] = layer1_data[data_index]
            data_index += 1

    data_index = 0
    for i in range(total_elements):
        if i not in hole_elements and data_index < vertices_per_layer:
            layer2_full[i] = layer2_data[data_index]
            data_index += 1

    layer1_reshaped = layer1_full.reshape((num_rows, num_cols))
    layer2_reshaped = layer2_full.reshape((num_rows, num_cols))

    return np.flipud(np.rot90(layer1_reshaped, k=1)), np.flipud(np.rot90(layer2_reshaped, k=1))


def extract_l_b_numbers(file_name):
    """ファイル名からL番号とB番号を抽出"""
    match = re.search(r'L(\d+)_B(\d+)', file_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def find_common_files(folders):
    """すべてのフォルダに共通するファイルを見つける"""
    # 最初のフォルダのファイルリストを取得
    if not folders or not os.path.exists(folders[0]):
        return []
    
    first_folder_files = set([f for f in os.listdir(folders[0]) if f.endswith('.npy')])
    
    # 他のフォルダと共通するファイルを探す
    common_files = first_folder_files.copy()
    for folder in folders[1:]:
        if not os.path.exists(folder):
            continue
        folder_files = set([f for f in os.listdir(folder) if f.endswith('.npy')])
        common_files = common_files.intersection(folder_files)
    
    return sorted(list(common_files))


def visualize_comparison(selected_files, folders, folder_labels, output_dir, base_name):
    """複数のフォルダのデータを比較可視化"""
    
    # レイアウト: 行=フォルダ数、列=2（Layer1, Layer2）
    num_folders = len(folders)
    fig, axes = plt.subplots(num_folders, 2, figsize=(14, 4 * num_folders))
    
    if num_folders == 1:
        axes = axes.reshape(1, -1)
    
    for folder_idx, (folder, label) in enumerate(zip(folders, folder_labels)):
        filename = selected_files[folder_idx] if isinstance(selected_files, list) else selected_files
        file_path = os.path.join(folder, filename)
        
        if not os.path.exists(file_path):
            axes[folder_idx, 0].text(0.5, 0.5, f"File not found\n{filename}", 
                                     ha='center', va='center', fontsize=12)
            axes[folder_idx, 0].axis('off')
            axes[folder_idx, 1].axis('off')
            continue
        
        try:
            layer1, layer2 = process_file(file_path)
            
            # Layer 1
            im1 = axes[folder_idx, 0].imshow(layer1, cmap='jet', aspect='equal', interpolation='none')
            axes[folder_idx, 0].set_title(f"{label}\nBottom Layer\n{os.path.basename(filename)}", 
                                          fontsize=11, pad=10)
            axes[folder_idx, 0].set_xlabel("Column")
            axes[folder_idx, 0].set_ylabel("Row")
            plt.colorbar(im1, ax=axes[folder_idx, 0], label="Value", fraction=0.025, pad=0.04)
            
            # Layer 2
            im2 = axes[folder_idx, 1].imshow(layer2, cmap='jet', aspect='equal', interpolation='none')
            axes[folder_idx, 1].set_title(f"{label}\nUpper Layer\n{os.path.basename(filename)}", 
                                          fontsize=11, pad=10)
            axes[folder_idx, 1].set_xlabel("Column")
            axes[folder_idx, 1].set_ylabel("Row")
            plt.colorbar(im2, ax=axes[folder_idx, 1], label="Value", fraction=0.025, pad=0.04)
        
        except Exception as e:
            axes[folder_idx, 0].text(0.5, 0.5, f"Error: {str(e)}", 
                                     ha='center', va='center', fontsize=12)
            axes[folder_idx, 0].axis('off')
            axes[folder_idx, 1].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{base_name}_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='すべての正規化フォルダからランダムにデータを選んで比較可視化')
    parser.add_argument('--base_dir', type=str, 
                       default='/home/nishioka/GNN/GNN_hole_2026',
                       help='ベースディレクトリ')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='出力ディレクトリ（指定しない場合は自動生成）')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='ランダムに選ぶサンプル数（デフォルト: 5）')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='ランダムシード（デフォルト: 42）')
    
    args = parser.parse_args()
    
    # すべての正規化フォルダを取得
    base_dir = args.base_dir
    sizes = ['2x2', '4x4', '8x8']
    normalizations = ['minmax', 'zscore', 'robust_zscore']
    
    all_folders = {}
    for size in sizes:
        all_folders[size] = {}
        for norm in normalizations:
            folder_path = os.path.join(base_dir, 
                f"Defect_hole_{size}_Region1_21_npy_subtracted_outlier_removed_percentile_{norm}")
            if os.path.exists(folder_path):
                all_folders[size][norm] = folder_path
    
    # 出力ディレクトリを設定
    if args.output_dir is None:
        args.output_dir = os.path.join(base_dir, "normalization_comparison_visualization")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"出力ディレクトリ: {args.output_dir}\n")
    
    # 各サイズごとに比較可視化
    random.seed(args.random_seed)
    
    for size in sizes:
        if size not in all_folders or len(all_folders[size]) == 0:
            print(f"警告: {size}版のフォルダが見つかりません")
            continue
        
        print(f"{size}版の比較可視化を実行中...")
        folders = list(all_folders[size].values())
        folder_labels = [f"{size}_{norm}" for norm in all_folders[size].keys()]
        
        # 共通ファイルを見つける
        common_files = find_common_files(folders)
        if len(common_files) == 0:
            print(f"  警告: {size}版に共通ファイルが見つかりません")
            continue
        
        print(f"  共通ファイル数: {len(common_files)}")
        
        # ランダムにサンプルを選ぶ
        num_samples = min(args.num_samples, len(common_files))
        selected_files = random.sample(common_files, num_samples)
        
        print(f"  可視化するファイル数: {num_samples}\n")
        
        # 各ファイルを可視化
        for filename in tqdm(selected_files, desc=f"  {size}版処理"):
            base_name = os.path.splitext(filename)[0]
            output_path = visualize_comparison(filename, folders, folder_labels, args.output_dir, base_name)
            print(f"    保存: {output_path}")
    
    print(f"\n可視化完了!")
    print(f"  保存先: {args.output_dir}")


if __name__ == "__main__":
    main()
