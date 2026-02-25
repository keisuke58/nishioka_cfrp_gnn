"""
外れ値を除去して正規化/標準化を行うスクリプト
"""

import os
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime


def calculate_global_statistics(data_dir, sample_size=None):
    """
    全データから統計量を計算（外れ値除去用）
    
    Args:
        data_dir: データディレクトリ
        sample_size: サンプルサイズ（Noneの場合は全ファイル、ただし最大1000ファイル）
    
    Returns:
        dict: 統計量（mean, std, q1, q3, iqr, percentile_1, percentile_99）
    """
    print(f"統計量を計算中: {data_dir}")
    
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    total_files = len(all_files)
    
    # デフォルトで最大1000ファイルをサンプリング（高速化）
    if sample_size is None:
        sample_size = min(1000, total_files)
    
    if sample_size < total_files:
        import random
        random.seed(42)
        all_files = random.sample(all_files, sample_size)
        print(f"  サンプルサイズ: {sample_size} / {total_files} ファイル")
    else:
        print(f"  全ファイル数: {len(all_files)}")
    
    all_values = []
    
    for filename in tqdm(all_files, desc="  ファイル読み込み"):
        file_path = os.path.join(data_dir, filename)
        try:
            data = np.load(file_path)
            
            # 1次元配列に変換
            if len(data.shape) == 2:
                values = data.flatten()
            elif len(data.shape) == 1:
                values = data
            else:
                continue
            
            # 有効な値のみを追加（NaN、無限大を除外）
            valid_values = values[np.isfinite(values)]
            # メモリ効率化: リストではなく配列に直接追加
            if len(all_values) == 0:
                all_values = valid_values
            else:
                all_values = np.concatenate([all_values, valid_values])
        
        except Exception as e:
            print(f"  警告: {filename} の読み込みに失敗: {e}")
            continue
    
    if len(all_values) == 0 or (isinstance(all_values, list) and len(all_values) == 0):
        raise ValueError("有効なデータが見つかりませんでした")
    
    if isinstance(all_values, list):
        all_values = np.array(all_values)
    
    # 統計量を計算
    mean = np.mean(all_values)
    std = np.std(all_values)
    median = np.median(all_values)
    # MAD (Median Absolute Deviation)
    mad = np.median(np.abs(all_values - median))
    # Robust Z-score用のMAD（ゼロ除算を避けるため、1e-8を追加）
    mad_robust = mad if mad > 1e-8 else 1e-8
    q1 = np.percentile(all_values, 25)
    q3 = np.percentile(all_values, 75)
    iqr = q3 - q1
    percentile_1 = np.percentile(all_values, 1)
    percentile_99 = np.percentile(all_values, 99)
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    
    stats = {
        'mean': float(mean),
        'std': float(std),
        'median': float(median),
        'mad': float(mad),
        'mad_robust': float(mad_robust),
        'q1': float(q1),
        'q3': float(q3),
        'iqr': float(iqr),
        'percentile_1': float(percentile_1),
        'percentile_99': float(percentile_99),
        'min': float(min_val),
        'max': float(max_val),
        'num_samples': len(all_values)
    }
    
    print(f"\n統計量:")
    print(f"  平均: {mean:.6f}")
    print(f"  標準偏差: {std:.6f}")
    print(f"  中央値: {median:.6f}")
    print(f"  MAD: {mad:.6f}")
    print(f"  最小値: {min_val:.6f}")
    print(f"  最大値: {max_val:.6f}")
    print(f"  1パーセンタイル: {percentile_1:.6f}")
    print(f"  99パーセンタイル: {percentile_99:.6f}")
    print(f"  Q1: {q1:.6f}, Q3: {q3:.6f}, IQR: {iqr:.6f}")
    
    return stats


def remove_outliers_and_normalize(data, stats, method='percentile', normalization='minmax'):
    """
    外れ値を除去して正規化/標準化
    
    Args:
        data: 入力データ（numpy配列）
        stats: 統計量の辞書
        method: 外れ値除去方法 ('percentile', 'iqr', 'zscore')
        normalization: 正規化方法 ('minmax', 'standardize')
    
    Returns:
        処理済みデータ
    """
    data = data.copy()
    
    # 外れ値の除去
    if method == 'percentile':
        # パーセンタイル法: 1%と99%の範囲外をクリップ
        lower_bound = stats['percentile_1']
        upper_bound = stats['percentile_99']
        data = np.clip(data, lower_bound, upper_bound)
    
    elif method == 'iqr':
        # IQR法: Q1 - 1.5*IQR と Q3 + 1.5*IQR の範囲外をクリップ
        lower_bound = stats['q1'] - 1.5 * stats['iqr']
        upper_bound = stats['q3'] + 1.5 * stats['iqr']
        data = np.clip(data, lower_bound, upper_bound)
    
    elif method == 'zscore':
        # Z-score法: |z| > 3 をクリップ
        z_scores = (data - stats['mean']) / stats['std']
        lower_bound = stats['mean'] - 3 * stats['std']
        upper_bound = stats['mean'] + 3 * stats['std']
        data = np.clip(data, lower_bound, upper_bound)
    
    # 正規化/標準化
    if normalization == 'minmax':
        # Min-Max正規化: [0, 1]にスケール
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)
    
    elif normalization == 'standardize' or normalization == 'zscore':
        # Z-score標準化: 平均0、標準偏差1
        data_mean = np.mean(data)
        data_std = np.std(data)
        if data_std > 0:
            data = (data - data_mean) / data_std
        else:
            data = np.zeros_like(data)
    
    elif normalization == 'robust_zscore':
        # Robust Z-score標準化: 中央値0、MAD基準
        data_median = np.median(data)
        data_mad = np.median(np.abs(data - data_median))
        if data_mad > 1e-8:
            data = (data - data_median) / data_mad
        else:
            data = np.zeros_like(data)
    
    return data


def process_directory(input_dir, output_dir, stats, method='percentile', normalization='minmax'):
    """
    ディレクトリ内の全ファイルを処理
    
    Args:
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ
        stats: 統計量の辞書
        method: 外れ値除去方法
        normalization: 正規化方法
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    print(f"\n処理中: {len(all_files)} ファイル")
    
    for filename in tqdm(all_files, desc="  処理"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            data = np.load(input_path)
            
            # 元の形状を保持
            original_shape = data.shape
            
            # 1次元に変換して処理
            if len(data.shape) == 2:
                data_flat = data.flatten()
            elif len(data.shape) == 1:
                data_flat = data
            else:
                print(f"  警告: {filename} の形状が予期しない形式です: {data.shape}")
                continue
            
            # 外れ値除去と正規化
            processed_data = remove_outliers_and_normalize(
                data_flat, stats, method=method, normalization=normalization
            )
            
            # 元の形状に戻す
            processed_data = processed_data.reshape(original_shape)
            
            # 保存
            np.save(output_path, processed_data)
        
        except Exception as e:
            print(f"  エラー: {filename} の処理に失敗しました: {e}")
            continue
    
    print(f"\n処理完了: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='外れ値を除去して正規化/標準化')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='入力データディレクトリ')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='出力ディレクトリ（指定しない場合は自動生成）')
    parser.add_argument('--method', type=str, default='percentile',
                       choices=['percentile', 'iqr', 'zscore'],
                       help='外れ値除去方法 (default: percentile)')
    parser.add_argument('--normalization', type=str, default='minmax',
                       choices=['minmax', 'standardize', 'zscore', 'robust_zscore'],
                       help='正規化方法 (default: minmax)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='統計量計算用のサンプルサイズ（Noneの場合は全ファイル）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"エラー: ディレクトリが存在しません: {args.input_dir}")
        return
    
    # 出力ディレクトリを設定
    if args.output_dir is None:
        input_dir_name = Path(args.input_dir).name
        method_suffix = f"_{args.method}_{args.normalization}"
        args.output_dir = os.path.join(Path(args.input_dir).parent, f"{input_dir_name}_outlier_removed{method_suffix}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"入力ディレクトリ: {args.input_dir}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print(f"外れ値除去方法: {args.method}")
    print(f"正規化方法: {args.normalization}\n")
    
    # 統計量を計算
    try:
        stats = calculate_global_statistics(args.input_dir, sample_size=args.sample_size)
    except Exception as e:
        print(f"エラー: 統計量の計算に失敗しました: {e}")
        return
    
    # 統計量を保存
    stats_path = os.path.join(args.output_dir, 'statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n統計量を保存: {stats_path}")
    
    # 処理内容をテキストファイルに保存
    process_info_path = os.path.join(args.output_dir, 'processing_info.txt')
    with open(process_info_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("データ処理情報\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【処理日時】\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("【入力情報】\n")
        f.write(f"入力ディレクトリ: {args.input_dir}\n")
        f.write(f"出力ディレクトリ: {args.output_dir}\n\n")
        
        f.write("【処理方法】\n")
        f.write(f"外れ値除去方法: {args.method}\n")
        if args.method == 'percentile':
            f.write("  - パーセンタイル法: 1パーセンタイルと99パーセンタイルの範囲外をクリップ\n")
        elif args.method == 'iqr':
            f.write("  - IQR法: Q1 - 1.5*IQR と Q3 + 1.5*IQR の範囲外をクリップ\n")
        elif args.method == 'zscore':
            f.write("  - Z-score法: |z| > 3 の範囲外をクリップ\n")
        
        f.write(f"正規化方法: {args.normalization}\n")
        if args.normalization == 'minmax':
            f.write("  - Min-Max正規化: データを[0, 1]の範囲にスケール\n")
        elif args.normalization == 'standardize' or args.normalization == 'zscore':
            f.write("  - Z-score標準化: 平均0、標準偏差1に変換\n")
        elif args.normalization == 'robust_zscore':
            f.write("  - Robust Z-score標準化: 中央値0、MAD基準で変換\n")
        
        f.write(f"統計量計算用サンプルサイズ: {args.sample_size if args.sample_size else '全ファイル'}\n\n")
        
        f.write("【統計量】\n")
        f.write(f"平均値: {stats['mean']:.6f}\n")
        f.write(f"標準偏差: {stats['std']:.6f}\n")
        f.write(f"最小値: {stats['min']:.6f}\n")
        f.write(f"最大値: {stats['max']:.6f}\n")
        f.write(f"1パーセンタイル: {stats['percentile_1']:.6f}\n")
        f.write(f"99パーセンタイル: {stats['percentile_99']:.6f}\n")
        f.write(f"第1四分位数 (Q1): {stats['q1']:.6f}\n")
        f.write(f"第3四分位数 (Q3): {stats['q3']:.6f}\n")
        f.write(f"四分位範囲 (IQR): {stats['iqr']:.6f}\n")
        f.write(f"統計量計算に使用したサンプル数: {stats['num_samples']}\n\n")
        
        f.write("【処理の詳細】\n")
        f.write("1. 全データファイルから統計量を計算（サンプリングを使用する場合あり）\n")
        f.write("2. 各データファイルに対して以下を実行:\n")
        f.write("   a. 外れ値を除去（指定された方法でクリップ）\n")
        f.write("   b. 正規化/標準化を適用\n")
        f.write("   c. 処理済みデータを保存\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    print(f"処理情報を保存: {process_info_path}")
    
    # ディレクトリを処理
    process_directory(args.input_dir, args.output_dir, stats, 
                     method=args.method, normalization=args.normalization)
    
    # 処理完了情報を追記
    all_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.npy')])
    processed_files = sorted([f for f in os.listdir(args.output_dir) if f.endswith('.npy')])
    
    with open(process_info_path, 'a', encoding='utf-8') as f:
        f.write(f"\n【処理結果】\n")
        f.write(f"入力ファイル数: {len(all_files)}\n")
        f.write(f"出力ファイル数: {len(processed_files)}\n")
        f.write(f"処理完了日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n完了!")
    print(f"  処理済みデータ: {args.output_dir}")
    print(f"  処理情報ファイル: {process_info_path}")


if __name__ == "__main__":
    main()
