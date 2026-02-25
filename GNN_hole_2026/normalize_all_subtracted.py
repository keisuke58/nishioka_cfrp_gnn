"""
all_subtracted_hole_no_defectフォルダ内のすべてのファイルを0-1の範囲で正規化するスクリプト
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


def calculate_global_minmax(data_dir):
    """
    全データから最小値と最大値を計算
    
    Args:
        data_dir: データディレクトリ
    
    Returns:
        tuple: (min_value, max_value)
    """
    print(f"全データから最小値・最大値を計算中: {data_dir}")
    
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    print(f"  ファイル数: {len(all_files)}")
    
    global_min = None
    global_max = None
    
    for filename in tqdm(all_files, desc="  ファイル読み込み"):
        file_path = os.path.join(data_dir, filename)
        try:
            data = np.load(file_path)
            
            # 有効な値のみを取得（NaN、無限大を除外）
            valid_data = data[np.isfinite(data)]
            
            if len(valid_data) == 0:
                continue
            
            file_min = np.min(valid_data)
            file_max = np.max(valid_data)
            
            if global_min is None:
                global_min = file_min
                global_max = file_max
            else:
                global_min = min(global_min, file_min)
                global_max = max(global_max, file_max)
        
        except Exception as e:
            print(f"  警告: {filename} の読み込みに失敗: {e}")
            continue
    
    if global_min is None or global_max is None:
        raise ValueError("有効なデータが見つかりませんでした")
    
    print(f"\n全データの統計:")
    print(f"  最小値: {global_min:.6f}")
    print(f"  最大値: {global_max:.6f}")
    print(f"  範囲: {global_max - global_min:.6f}")
    
    return global_min, global_max


def normalize_to_01(data, global_min, global_max):
    """
    データを0-1の範囲に正規化（Min-Max正規化）
    
    Args:
        data: 入力データ（numpy配列）
        global_min: 全データの最小値
        global_max: 全データの最大値
    
    Returns:
        正規化されたデータ
    """
    data = data.copy()
    
    # 範囲が0の場合はすべて0を返す
    if global_max == global_min:
        return np.zeros_like(data)
    
    # Min-Max正規化: (x - min) / (max - min)
    normalized = (data - global_min) / (global_max - global_min)
    
    return normalized


def process_directory(input_dir, output_dir, global_min, global_max):
    """
    ディレクトリ内の全ファイルを正規化
    
    Args:
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ
        global_min: 全データの最小値
        global_max: 全データの最大値
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    print(f"\n処理中: {len(all_files)} ファイル")
    
    processed_count = 0
    error_count = 0
    error_files = []
    
    for filename in tqdm(all_files, desc="  正規化処理"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            data = np.load(input_path)
            
            # 元の形状を保持
            original_shape = data.shape
            
            # 正規化
            normalized_data = normalize_to_01(data, global_min, global_max)
            
            # 形状を確認（元の形状に戻す）
            normalized_data = normalized_data.reshape(original_shape)
            
            # 保存
            np.save(output_path, normalized_data)
            processed_count += 1
        
        except Exception as e:
            print(f"  エラー: {filename} の処理に失敗しました: {e}")
            error_count += 1
            error_files.append((filename, str(e)))
            continue
    
    print(f"\n処理完了!")
    print(f"  成功: {processed_count} ファイル")
    print(f"  エラー: {error_count} ファイル")
    print(f"  出力ディレクトリ: {output_dir}")
    
    return processed_count, error_count, error_files


def main():
    # 入力ディレクトリ
    input_dir = '/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect'
    
    # 出力ディレクトリ
    output_dir = '/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_normalized'
    
    print("=" * 80)
    print("0-1正規化スクリプト")
    print("=" * 80)
    print(f"\n入力ディレクトリ: {input_dir}")
    print(f"出力ディレクトリ: {output_dir}\n")
    
    if not os.path.exists(input_dir):
        print(f"エラー: ディレクトリが存在しません: {input_dir}")
        return
    
    # 全データから最小値・最大値を計算
    try:
        global_min, global_max = calculate_global_minmax(input_dir)
    except Exception as e:
        print(f"エラー: 統計量の計算に失敗しました: {e}")
        return
    
    # ディレクトリを処理
    processed_count, error_count, error_files = process_directory(
        input_dir, output_dir, global_min, global_max
    )
    
    # 処理情報をファイルに保存
    info_path = os.path.join(output_dir, 'normalization_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("0-1正規化処理情報\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【処理日時】\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("【入力情報】\n")
        f.write(f"入力ディレクトリ: {input_dir}\n\n")
        
        f.write("【出力情報】\n")
        f.write(f"出力ディレクトリ: {output_dir}\n\n")
        
        f.write("【正規化方法】\n")
        f.write("Min-Max正規化: (x - min) / (max - min)\n")
        f.write("範囲: [0, 1]\n\n")
        
        f.write("【全データ統計量】\n")
        f.write(f"最小値: {global_min:.6f}\n")
        f.write(f"最大値: {global_max:.6f}\n")
        f.write(f"範囲: {global_max - global_min:.6f}\n\n")
        
        f.write("【処理結果】\n")
        f.write(f"処理ファイル数: {processed_count + error_count}\n")
        f.write(f"成功: {processed_count} ファイル\n")
        f.write(f"エラー: {error_count} ファイル\n\n")
        
        if error_files:
            f.write("【エラーファイル一覧】\n")
            for filename, error_msg in error_files:
                f.write(f"  - {filename}: {error_msg}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"処理情報を保存: {info_path}")


if __name__ == "__main__":
    main()
