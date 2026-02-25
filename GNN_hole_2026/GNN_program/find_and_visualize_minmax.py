"""
指定ディレクトリ内の全npyファイルから最大値・最小値を持つデータを見つけて可視化するスクリプト
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
from pathlib import Path

# フォントの設定
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12

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
        # 2次元配列の場合はargmaxを取る
        decoded_data = np.argmax(data, axis=1)
    elif len(data.shape) == 1:
        decoded_data = data
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    # ノード数を確認・調整
    if len(decoded_data) < vertices_per_layer * 2:
        # データが不足している場合はゼロパディング
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


def find_minmax_files(data_dir):
    """全ファイルをスキャンして最大値・最小値を持つファイルを見つける"""
    print(f"ディレクトリをスキャン中: {data_dir}")
    
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    print(f"見つかったファイル数: {len(all_files)}")
    
    global_min = float('inf')
    global_max = float('-inf')
    min_file = None
    max_file = None
    
    for i, filename in enumerate(all_files):
        file_path = os.path.join(data_dir, filename)
        
        try:
            data = np.load(file_path)
            
            # データの形状に応じて処理
            if len(data.shape) == 2:
                # 2次元配列の場合はargmaxを取る
                values = np.argmax(data, axis=1)
            elif len(data.shape) == 1:
                values = data
            else:
                print(f"警告: {filename} の形状が予期しない形式です: {data.shape}")
                continue
            
            # 有効な値のみを考慮（NaNや無限大を除外）
            valid_values = values[np.isfinite(values)]
            
            if len(valid_values) == 0:
                continue
            
            file_min = np.min(valid_values)
            file_max = np.max(valid_values)
            
            # グローバル最小値・最大値を更新
            if file_min < global_min:
                global_min = file_min
                min_file = filename
            
            if file_max > global_max:
                global_max = file_max
                max_file = filename
            
            if (i + 1) % 1000 == 0:
                print(f"  処理中: {i + 1}/{len(all_files)} ファイル")
                print(f"    現在の最小値: {global_min} ({min_file})")
                print(f"    現在の最大値: {global_max} ({max_file})")
        
        except Exception as e:
            print(f"警告: {filename} の読み込みに失敗しました: {e}")
            continue
    
    print(f"\nスキャン完了!")
    print(f"  グローバル最小値: {global_min} (ファイル: {min_file})")
    print(f"  グローバル最大値: {global_max} (ファイル: {max_file})")
    
    return min_file, max_file, global_min, global_max


def visualize_file(file_path, output_path, title_suffix=""):
    """ファイルを可視化"""
    try:
        layer1, layer2 = process_file(file_path)
        
        filename = os.path.basename(file_path)
        
        # 2行1列のレイアウト: Layer 1 と Layer 2
        fig, axes = plt.subplots(2, 1, figsize=(12, 16))
        
        # Layer 1 (Bottom Layer)
        im1 = axes[0].imshow(layer1, cmap='jet', aspect='equal', interpolation='none')
        axes[0].set_title(f"Bottom Layer{title_suffix}\n{filename}", fontsize=14, pad=20)
        axes[0].set_xlabel("Column")
        axes[0].set_ylabel("Row")
        plt.colorbar(im1, ax=axes[0], label="Value", fraction=0.025, pad=0.04)
        
        # Layer 2 (Upper Layer)
        im2 = axes[1].imshow(layer2, cmap='jet', aspect='equal', interpolation='none')
        axes[1].set_title(f"Upper Layer{title_suffix}\n{filename}", fontsize=14, pad=20)
        axes[1].set_xlabel("Column")
        axes[1].set_ylabel("Row")
        plt.colorbar(im2, ax=axes[1], label="Value", fraction=0.025, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"可視化を保存: {output_path}")
        
    except Exception as e:
        print(f"エラー: {file_path} の可視化に失敗しました: {e}")


def main():
    parser = argparse.ArgumentParser(description='最大値・最小値を持つデータを見つけて可視化')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='データディレクトリのパス')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='可視化結果の保存先ディレクトリ（指定しない場合は自動生成）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"エラー: ディレクトリが存在しません: {args.data_dir}")
        return
    
    # 出力ディレクトリを設定
    if args.output_dir is None:
        data_dir_name = Path(args.data_dir).name
        args.output_dir = os.path.join(Path(args.data_dir).parent, f"{data_dir_name}_minmax_visualization")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"出力ディレクトリ: {args.output_dir}\n")
    
    # 最大値・最小値を持つファイルを見つける
    min_file, max_file, global_min, global_max = find_minmax_files(args.data_dir)
    
    if min_file is None or max_file is None:
        print("エラー: 有効なデータファイルが見つかりませんでした")
        return
    
    # 可視化
    print("\n可視化を実行中...")
    
    # 最小値を持つファイルを可視化
    min_file_path = os.path.join(args.data_dir, min_file)
    min_output_path = os.path.join(args.output_dir, f"min_value_{global_min}_{os.path.splitext(min_file)[0]}_visualization.png")
    visualize_file(min_file_path, min_output_path, f" (最小値: {global_min})")
    
    # 最大値を持つファイルを可視化
    max_file_path = os.path.join(args.data_dir, max_file)
    max_output_path = os.path.join(args.output_dir, f"max_value_{global_max}_{os.path.splitext(max_file)[0]}_visualization.png")
    visualize_file(max_file_path, max_output_path, f" (最大値: {global_max})")
    
    # 結果をテキストファイルに保存
    result_file = os.path.join(args.output_dir, "minmax_results.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"データディレクトリ: {args.data_dir}\n")
        f.write(f"グローバル最小値: {global_min}\n")
        f.write(f"最小値を持つファイル: {min_file}\n")
        f.write(f"グローバル最大値: {global_max}\n")
        f.write(f"最大値を持つファイル: {max_file}\n")
    
    print(f"\n結果を保存: {result_file}")
    print(f"\n可視化完了!")
    print(f"  最小値: {global_min} ({min_file})")
    print(f"  最大値: {global_max} ({max_file})")
    print(f"  保存先: {args.output_dir}")


if __name__ == "__main__":
    main()
