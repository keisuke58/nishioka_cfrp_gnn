"""
all_sub_hole_defectフォルダ内のすべてのファイルをZ-score標準化するスクリプト
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


def calculate_global_mean_std(data_dir):
    """
    全データから平均値と標準偏差を計算（メモリ効率的な方法）
    
    Args:
        data_dir: データディレクトリ
    
    Returns:
        tuple: (mean, std)
    """
    print(f"全データから平均値・標準偏差を計算中: {data_dir}")
    
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    print(f"  ファイル数: {len(all_files)}")
    
    # オンラインアルゴリズムで平均と分散を計算
    n = 0
    mean = 0.0
    M2 = 0.0  # 分散計算用の累積値
    min_val = None
    max_val = None
    
    for filename in tqdm(all_files, desc="  ファイル読み込み"):
        file_path = os.path.join(data_dir, filename)
        try:
            data = np.load(file_path)
            
            # 有効な値のみを取得（NaN、無限大を除外）
            valid_data = data[np.isfinite(data)]
            
            if len(valid_data) == 0:
                continue
            
            # 最小値・最大値を更新
            file_min = np.min(valid_data)
            file_max = np.max(valid_data)
            if min_val is None:
                min_val = file_min
                max_val = file_max
            else:
                min_val = min(min_val, file_min)
                max_val = max(max_val, file_max)
            
            # Welford's online algorithm で平均と分散を計算
            for x in valid_data:
                n += 1
                delta = x - mean
                mean += delta / n
                delta2 = x - mean
                M2 += delta * delta2
        
        except Exception as e:
            print(f"  警告: {filename} の読み込みに失敗: {e}")
            continue
    
    if n == 0:
        raise ValueError("有効なデータが見つかりませんでした")
    
    # 標準偏差を計算
    variance = M2 / n if n > 1 else 0.0
    global_std = np.sqrt(variance)
    global_mean = mean
    
    print(f"\n全データの統計:")
    print(f"  サンプル数: {n}")
    print(f"  平均値: {global_mean:.6f}")
    print(f"  標準偏差: {global_std:.6f}")
    print(f"  最小値: {min_val:.6f}")
    print(f"  最大値: {max_val:.6f}")
    
    return global_mean, global_std


def standardize_zscore(data, global_mean, global_std):
    """
    データをZ-score標準化（平均0、標準偏差1）
    
    Args:
        data: 入力データ（numpy配列）
        global_mean: 全データの平均値
        global_std: 全データの標準偏差
    
    Returns:
        標準化されたデータ
    """
    data = data.copy()
    
    # 標準偏差が0の場合はすべて0を返す
    if global_std == 0:
        return np.zeros_like(data)
    
    # Z-score標準化: (x - mean) / std
    standardized = (data - global_mean) / global_std
    
    return standardized


def process_directory(input_dir, output_dir, global_mean, global_std):
    """
    ディレクトリ内の全ファイルを標準化
    
    Args:
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ
        global_mean: 全データの平均値
        global_std: 全データの標準偏差
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    print(f"\n処理中: {len(all_files)} ファイル")
    
    processed_count = 0
    error_count = 0
    error_files = []
    
    for filename in tqdm(all_files, desc="  標準化処理"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            data = np.load(input_path)
            
            # 元の形状を保持
            original_shape = data.shape
            
            # 標準化
            standardized_data = standardize_zscore(data, global_mean, global_std)
            
            # 形状を確認（元の形状に戻す）
            standardized_data = standardized_data.reshape(original_shape)
            
            # 保存
            np.save(output_path, standardized_data)
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
    output_dir = '/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore'
    
    print("=" * 80)
    print("Z-score標準化スクリプト")
    print("=" * 80)
    print(f"\n入力ディレクトリ: {input_dir}")
    print(f"出力ディレクトリ: {output_dir}\n")
    
    if not os.path.exists(input_dir):
        print(f"エラー: ディレクトリが存在しません: {input_dir}")
        return
    
    # 全データから平均値・標準偏差を計算
    try:
        global_mean, global_std = calculate_global_mean_std(input_dir)
    except Exception as e:
        print(f"エラー: 統計量の計算に失敗しました: {e}")
        return
    
    # ディレクトリを処理
    processed_count, error_count, error_files = process_directory(
        input_dir, output_dir, global_mean, global_std
    )
    
    # 処理情報をファイルに保存
    info_path = os.path.join(output_dir, 'standardization_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Z-score標準化処理情報\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【処理日時】\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("【入力情報】\n")
        f.write(f"入力ディレクトリ: {input_dir}\n\n")
        
        f.write("【出力情報】\n")
        f.write(f"出力ディレクトリ: {output_dir}\n\n")
        
        f.write("【標準化方法】\n")
        f.write("Z-score標準化: (x - mean) / std\n")
        f.write("平均: 0, 標準偏差: 1\n\n")
        
        f.write("【全データ統計量】\n")
        f.write(f"平均値: {global_mean:.6f}\n")
        f.write(f"標準偏差: {global_std:.6f}\n\n")
        
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
