"""
normalized と zscale のデータを空間的に可視化するスクリプト
pred_compare1.ipynb のスタイルを参考に
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re
import os
from pathlib import Path
from tqdm import tqdm
import random

# フォントの設定
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10

# カラーマップの設定
normalized_cmap = 'jet'      # Min-Max正規化用
zscore_cmap = 'coolwarm'     # Z-score標準化用

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


def reshape_data_to_grid(data):
    """
    1次元データを2層のグリッドに再形成
    
    Args:
        data: 1次元配列（13942要素）
    
    Returns:
        layer1_grid, layer2_grid: 各レイヤーのグリッドデータ (num_rows, num_cols)
    """
    # データを2層に分割
    layer1_data = data[:vertices_per_layer]
    layer2_data = data[vertices_per_layer:]
    
    # グリッドを初期化（NaNで埋める）
    layer1_full = np.empty(total_elements)
    layer1_full[:] = np.nan
    layer2_full = np.empty(total_elements)
    layer2_full[:] = np.nan
    
    # Layer 1のデータを配置
    data_index = 0
    for i in range(total_elements):
        if i not in hole_elements and data_index < vertices_per_layer:
            layer1_full[i] = layer1_data[data_index]
            data_index += 1
    
    # Layer 2のデータを配置
    data_index = 0
    for i in range(total_elements):
        if i not in hole_elements and data_index < vertices_per_layer:
            layer2_full[i] = layer2_data[data_index]
            data_index += 1
    
    # グリッドに再形成
    layer1_reshaped = layer1_full.reshape((num_rows, num_cols))
    layer2_reshaped = layer2_full.reshape((num_rows, num_cols))
    
    # 回転と反転（ノートブックと同じ処理）
    layer1_grid = np.flipud(np.rot90(layer1_reshaped, k=1))
    layer2_grid = np.flipud(np.rot90(layer2_reshaped, k=1))
    
    return layer1_grid, layer2_grid


def extract_layer_number(file_name):
    """ファイル名からL番号を抽出"""
    match = re.search(r'L(\d+)', file_name)
    return int(match.group(1)) if match else None


def visualize_file(normalized_file, zscore_file, output_dir):
    """
    1つのファイルのnormalized と zscore を可視化
    
    Args:
        normalized_file: Min-Max正規化ファイルのパス
        zscore_file: Z-score標準化ファイルのパス
        output_dir: 出力ディレクトリ
    """
    try:
        # データを読み込む
        normalized_data = np.load(normalized_file)
        zscore_data = np.load(zscore_file)
        
        # グリッドに再形成
        norm_layer1, norm_layer2 = reshape_data_to_grid(normalized_data)
        zscore_layer1, zscore_layer2 = reshape_data_to_grid(zscore_data)
        
        # ファイル名から情報を取得
        filename = Path(normalized_file).stem
        layer_number = extract_layer_number(filename)
        
        if layer_number is None:
            print(f"警告: レイヤー番号を抽出できませんでした: {filename}")
            return
        
        # レイヤーを決定（L2-L10: Layer 1, L11-L20: Layer 2）
        if 2 <= layer_number <= 10:
            norm_layer = norm_layer1
            zscore_layer = zscore_layer1
            layer_name = "Bottom Layer"
        elif 11 <= layer_number <= 20:
            norm_layer = norm_layer2
            zscore_layer = zscore_layer2
            layer_name = "Upper Layer"
        else:
            # L2-L20以外はLayer 1を使用
            norm_layer = norm_layer1
            zscore_layer = zscore_layer1
            layer_name = "Layer 1"
        
        # マスク処理: NaN 部分を透明にする
        norm_masked = np.ma.masked_where(np.isnan(norm_layer), norm_layer)
        zscore_masked = np.ma.masked_where(np.isnan(zscore_layer), zscore_layer)
        
        # 可視化
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{filename} - {layer_name}', fontsize=14, fontweight='bold')
        
        # Min-Max正規化
        im1 = axes[0].imshow(norm_masked, cmap=normalized_cmap, aspect="equal", interpolation='none')
        axes[0].set_title('Min-Max Normalized [0, 1]', fontsize=12)
        cbar1 = plt.colorbar(im1, ax=axes[0], label="Normalized Value", fraction=0.025, pad=0.04)
        
        # Z-score標準化
        im2 = axes[1].imshow(zscore_masked, cmap=zscore_cmap, aspect="equal", interpolation='none')
        axes[1].set_title('Z-score Standardized (mean=0, std=1)', fontsize=12)
        cbar2 = plt.colorbar(im2, ax=axes[1], label="Z-score Value", fraction=0.025, pad=0.04)
        
        plt.tight_layout()
        
        # 保存
        output_filename = filename + '_spatial_visualization.png'
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
    
    except Exception as e:
        print(f"エラー: {normalized_file} の可視化に失敗: {e}")
        return False


def main():
    # ディレクトリパス
    normalized_dir = '/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_normalized'
    zscore_dir = '/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore'
    output_dir = '/home/nishioka/GNN/GNN_hole_2026/normalized_zscore_spatial_visualization'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Normalized と Z-score の空間的可視化")
    print("=" * 80)
    print(f"\nMin-Max正規化ディレクトリ: {normalized_dir}")
    print(f"Z-score標準化ディレクトリ: {zscore_dir}")
    print(f"出力ディレクトリ: {output_dir}\n")
    
    # すべてのファイルを取得
    normalized_files = sorted([f for f in os.listdir(normalized_dir) if f.endswith('.npy')])
    
    # サンプリング（50個）
    if len(normalized_files) > 50:
        random.seed(42)
        sample_files = random.sample(normalized_files, 50)
    else:
        sample_files = normalized_files
    
    print(f"可視化対象ファイル数: {len(sample_files)}")
    
    # 各ファイルを可視化
    success_count = 0
    for filename in tqdm(sample_files, desc="可視化中"):
        normalized_path = os.path.join(normalized_dir, filename)
        zscore_path = os.path.join(zscore_dir, filename)
        
        if not os.path.exists(zscore_path):
            print(f"警告: Z-scoreファイルが見つかりません: {filename}")
            continue
        
        if visualize_file(normalized_path, zscore_path, output_dir):
            success_count += 1
    
    print(f"\n可視化完了!")
    print(f"  成功: {success_count} ファイル")
    print(f"  出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    main()
