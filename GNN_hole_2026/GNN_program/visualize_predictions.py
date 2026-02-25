"""
予測結果を可視化するスクリプト
pred_compare1.ipynbを参考に実装
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import re
import os
import argparse
from pathlib import Path

# フォントの設定
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10

# カラーマップの設定
dpsss_cmap = 'jet'         # DSPSSデータ用：カラフルな 'jet'
defect_cmap = 'coolwarm'   # Defect Label & Predict Data用：赤青の 'coolwarm'

# クラス0を黒で表示するカスタムカラーマップ
def get_defect_cmap_with_white_zero():
    """クラス0を黒で表示するカスタムカラーマップ"""
    # coolwarmカラーマップを取得
    coolwarm = cm.get_cmap('coolwarm', 19)
    colors = coolwarm(np.linspace(0, 1, 19))
    # クラス0を黒に設定
    colors[0] = [0.0, 0.0, 0.0, 1.0]  # 黒
    return ListedColormap(colors)

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


def extract_layer_number(file_name):
    """ファイル名からL番号を抽出"""
    match = re.search(r'L(\d+)', file_name)
    return int(match.group(1)) if match else None


def calculate_19label_max(label_files):
    """19labelファイルの最大値を計算"""
    max_values = []
    for file_path in label_files:
        if "_19label.npy" in file_path:
            data = np.load(file_path)
            if len(data.shape) == 2:
                decoded_data = np.argmax(data, axis=1)
            else:
                decoded_data = data
            max_value = int(np.max(decoded_data))
            max_values.append(max_value)
    return max_values


def extract_l_b_numbers(file_name):
    """ファイル名からL番号とB番号を抽出（後方互換性のため保持）"""
    match = re.search(r'L(\d+)_B(\d+)', file_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None

def get_base_name(file_name):
    """ファイル名からベース名を取得（拡張子と_19label、_predを除く）"""
    base = file_name.replace('_19label.npy', '').replace('_pred.npy', '').replace('.npy', '')
    return base

def find_matching_files(predict_dir, label_dir, dpsss_dir, input_data_dir=None):
    """予測、ラベル、DSPSSファイルを対応付ける（完全なファイル名でマッチング）"""
    # 予測ファイルを取得（predictionsフォルダも確認）
    pred_files = []
    # 直接predict_dir内を確認
    if os.path.exists(predict_dir):
        pred_files.extend([f for f in os.listdir(predict_dir) if f.endswith('_pred.npy')])
    # predictionsフォルダ内も確認
    predictions_folder = os.path.join(predict_dir, 'predictions')
    if os.path.exists(predictions_folder):
        pred_files.extend([f for f in os.listdir(predictions_folder) if f.endswith('_pred.npy')])
    pred_files = sorted(pred_files)
    
    # 入力データディレクトリを優先的に使用（予測に使ったデータ）
    data_dir = input_data_dir if input_data_dir and os.path.exists(input_data_dir) else dpsss_dir
    
    # DSPSSファイルをベース名でインデックス化（完全一致マッチング）
    dpsss_index = {}
    if os.path.exists(data_dir):
        data_files = os.listdir(data_dir)
        for data_file_name in data_files:
            if data_file_name.endswith('.npy') and not data_file_name.endswith('_pred.npy') and not data_file_name.endswith('_19label.npy'):
                base_name = get_base_name(data_file_name)
                dpsss_index[base_name] = os.path.join(data_dir, data_file_name)
    
    # ラベルファイルをベース名でインデックス化（完全一致マッチング）
    label_index = {}
    if os.path.exists(label_dir):
        label_files = os.listdir(label_dir)
        for label_file_name in label_files:
            if label_file_name.endswith('_19label.npy'):
                base_name = get_base_name(label_file_name)
                label_index[base_name] = os.path.join(label_dir, label_file_name)
    
    file_triplets = []
    for pred_file in pred_files:
        # base_nameから_predを削除
        base_name = get_base_name(pred_file)
        
        # 対応するラベルファイルを探す（完全一致）
        label_file = label_index.get(base_name)
        
        # 対応するDSPSSファイルを探す（完全一致）
        dpsss_file_path = dpsss_index.get(base_name)
        
        # 予測ファイルのパスを決定（predictionsフォルダ内かどうか確認）
        pred_path = os.path.join(predict_dir, pred_file)
        if not os.path.exists(pred_path):
            pred_path = os.path.join(predictions_folder, pred_file)
        
        # DSPSSファイルが見つかった場合のみ追加
        if dpsss_file_path and os.path.exists(dpsss_file_path) and os.path.exists(pred_path):
            # ベース名が完全に一致していることを確認
            dpsss_base = get_base_name(os.path.basename(dpsss_file_path))
            if base_name != dpsss_base:
                print(f"警告: ベース名が一致しません - 予測: {base_name}, データ: {dpsss_base}")
                continue
            
            file_triplets.append({
                'dpsss': dpsss_file_path,
                'label': label_file,
                'pred': pred_path,
                'base_name': base_name
            })
    
    return file_triplets


def visualize_comparison(file_triplets, output_dir=None, max_files=None, input_data_dir=None):
    """予測結果を可視化"""
    if max_files:
        file_triplets = file_triplets[:max_files]
    
    print(f"可視化するファイル数: {len(file_triplets)}")
    
    for i, files in enumerate(file_triplets):
        dpsss_file = files['dpsss']
        label_file = files['label']
        pred_file = files['pred']
        base_name = files['base_name']
        
        # ファイル存在確認
        if not os.path.exists(dpsss_file):
            print(f"Warning: DSPSS file not found: {dpsss_file}")
            continue
        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file not found: {pred_file}")
            continue
        
        layer_number = extract_layer_number(dpsss_file)
        if layer_number is None:
            print(f"Warning: Could not extract layer number from {dpsss_file}")
            continue
        
        # DSPSSデータの処理
        dpsss_layer1, dpsss_layer2 = process_file(dpsss_file)
        
        # Defect Labelは常に0~18で固定
        k_max = 18  # Defect Labelは常に0~18で固定
        
        # カスタムカラーマップ（クラス0を白色）
        defect_cmap_custom = get_defect_cmap_with_white_zero()
        
        # Layer 1とLayer 2の両方を常に可視化
        # 2行3列のレイアウト: Layer 1 (上段) と Layer 2 (下段)
        # 各段: DSPSS, Defect Label, Prediction
        
        num_cols = 3  # DSPSS, Defect Label, Prediction
        num_rows = 2  # Layer 1, Layer 2
        
        plt.figure(figsize=(18, 12))
        
        # Layer 1 (上段)
        # DSPSS Layer 1
        plt.subplot(num_rows, num_cols, 1)
        plt.imshow(dpsss_layer1, cmap=dpsss_cmap, aspect="equal", interpolation='none')
        plt.title(f"Bottom Layer - DSPSS\n{os.path.basename(dpsss_file)}", fontsize=12, pad=50)
        plt.colorbar(label="Normalized DSPSS", fraction=0.025, pad=0.04)
        
        # Defect Label Layer 1（存在する場合）
        if label_file and os.path.exists(label_file):
            label_layer1, label_layer2 = process_file(label_file)
            label_layer1_masked = np.ma.masked_where(np.isnan(label_layer1), label_layer1)
            
            plt.subplot(num_rows, num_cols, 2)
            label_layer1_int = np.where(np.isnan(label_layer1), 0, label_layer1).astype(int)
            label_layer1_int_masked = np.ma.masked_where(np.isnan(label_layer1), label_layer1_int)
            im = plt.imshow(label_layer1_int_masked, cmap=defect_cmap_custom, aspect="equal", 
                          interpolation='none', vmin=0, vmax=18)  # 常に0~18で固定
            plt.title(f"Bottom Layer - Defect Label\n{os.path.basename(label_file)}", fontsize=12, pad=50)
            cbar = plt.colorbar(im, label="Defect Label", fraction=0.025, pad=0.04)
            cbar.set_ticks(range(0, 19))  # 0~18で固定
        else:
            plt.subplot(num_rows, num_cols, 2)
            plt.text(0.5, 0.5, "No Label", ha='center', va='center', fontsize=14)
            plt.axis('off')
        
        # Prediction Layer 1
        pred_layer1, pred_layer2 = process_file(pred_file)
        pred_layer1_masked = np.ma.masked_where(np.isnan(pred_layer1), pred_layer1)
        
        plt.subplot(num_rows, num_cols, 3)
        pred_layer1_int = np.where(np.isnan(pred_layer1), 0, pred_layer1).astype(int)
        pred_layer1_int_masked = np.ma.masked_where(np.isnan(pred_layer1), pred_layer1_int)
        # Predictionの最大値を計算
        pred_max = int(np.max(pred_layer1_int[~np.isnan(pred_layer1)])) if np.any(~np.isnan(pred_layer1)) else 18
        im = plt.imshow(pred_layer1_int_masked, cmap=defect_cmap_custom, aspect="equal", 
                      interpolation='none', vmin=0, vmax=max(pred_max, 18))
        plt.title(f"Bottom Layer - Prediction\n{os.path.basename(pred_file)}", fontsize=12, pad=50)
        cbar = plt.colorbar(im, label="Prediction", fraction=0.025, pad=0.04)
        cbar.set_ticks(range(0, max(pred_max, 18) + 1))
        
        # Layer 2 (下段)
        # DSPSS Layer 2
        plt.subplot(num_rows, num_cols, 4)
        plt.imshow(dpsss_layer2, cmap=dpsss_cmap, aspect="equal", interpolation='none')
        plt.title(f"Upper Layer - DSPSS\n{os.path.basename(dpsss_file)}", fontsize=12, pad=50)
        plt.colorbar(label="Normalized DSPSS", fraction=0.025, pad=0.04)
        
        # Defect Label Layer 2（存在する場合）
        # 常に後半（6971以降）をUpper Layerとして表示
        if label_file and os.path.exists(label_file):
            plt.subplot(num_rows, num_cols, 5)
            label_layer2_int = np.where(np.isnan(label_layer2), 0, label_layer2).astype(int)
            label_layer2_int_masked = np.ma.masked_where(np.isnan(label_layer2), label_layer2_int)
            im = plt.imshow(label_layer2_int_masked, cmap=defect_cmap_custom, aspect="equal", 
                          interpolation='none', vmin=0, vmax=18)  # 常に0~18で固定
            plt.title(f"Upper Layer - Defect Label\n{os.path.basename(label_file)}", fontsize=12, pad=50)
            cbar = plt.colorbar(im, label="Defect Label", fraction=0.025, pad=0.04)
            cbar.set_ticks(range(0, 19))  # 0~18で固定
        else:
            plt.subplot(num_rows, num_cols, 5)
            plt.text(0.5, 0.5, "No Label", ha='center', va='center', fontsize=14)
            plt.axis('off')
        
        # Prediction Layer 2
        pred_layer2_masked = np.ma.masked_where(np.isnan(pred_layer2), pred_layer2)
        
        plt.subplot(num_rows, num_cols, 6)
        pred_layer2_int = np.where(np.isnan(pred_layer2), 0, pred_layer2).astype(int)
        pred_layer2_int_masked = np.ma.masked_where(np.isnan(pred_layer2), pred_layer2_int)
        # Predictionの最大値を計算
        pred_max = int(np.max(pred_layer2_int[~np.isnan(pred_layer2)])) if np.any(~np.isnan(pred_layer2)) else 18
        im = plt.imshow(pred_layer2_int_masked, cmap=defect_cmap_custom, aspect="equal", 
                      interpolation='none', vmin=0, vmax=max(pred_max, 18))
        plt.title(f"Upper Layer - Prediction\n{os.path.basename(pred_file)}", fontsize=12, pad=50)
        cbar = plt.colorbar(im, label="Prediction", fraction=0.025, pad=0.04)
        cbar.set_ticks(range(0, max(pred_max, 18) + 1))
        
        plt.tight_layout()
        
        # 保存（PNGファイルを直接フォルダに保存）
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base_name}_visualization.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"保存: {output_path}")
            plt.close()  # メモリリークを防ぐ
        else:
            plt.show()
        
        if (i + 1) % 10 == 0:
            print(f"処理中: {i + 1}/{len(file_triplets)} ファイル")


def main():
    parser = argparse.ArgumentParser(description='予測結果を可視化')
    parser.add_argument('--predict_dir', type=str, required=True,
                       help='予測結果のディレクトリパス')
    parser.add_argument('--label_dir', type=str, 
                       default='/home/nishioka/GNN/GNN_hole_2026/Def4x4_19class_label',
                       help='ラベルファイルのディレクトリパス')
    parser.add_argument('--dpsss_dir', type=str,
                       default='/home/nishioka/GNN/GNN_hole_2026/DSPSS_8x8/8x8_norm_clip_final',
                       help='DSPSSデータのディレクトリパス')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='可視化結果の保存先ディレクトリ（指定しない場合は表示のみ）')
    parser.add_argument('--max_files', type=int, default=10,
                       help='可視化する最大ファイル数（デフォルト: 10）')
    parser.add_argument('--input_data_dir', type=str, default=None,
                       help='予測に使った入力データディレクトリ（指定しない場合はメタデータから読み込み）')
    
    args = parser.parse_args()
    
    # メタデータから入力データフォルダを読み込む
    input_data_dir = args.input_data_dir
    metadata_path = os.path.join(args.predict_dir, 'metadata.json')
    if not input_data_dir and os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if 'input_data_folder' in metadata:
                input_data_dir = metadata['input_data_folder']
                print(f"メタデータから入力データフォルダを読み込み: {input_data_dir}")
    
    # ファイルの対応付け
    print("ファイルを検索中...")
    file_triplets = find_matching_files(args.predict_dir, args.label_dir, args.dpsss_dir, input_data_dir)
    
    if len(file_triplets) == 0:
        print("エラー: 対応するファイルが見つかりませんでした")
        return
    
    print(f"見つかったファイルペア: {len(file_triplets)}")
    
    # 可視化
    visualize_comparison(file_triplets, args.output_dir, args.max_files, input_data_dir)
    
    print("\n可視化完了!")


if __name__ == "__main__":
    main()
