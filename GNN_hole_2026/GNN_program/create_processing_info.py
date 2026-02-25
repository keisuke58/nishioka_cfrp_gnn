"""
既存の出力ディレクトリに処理情報ファイルを作成するスクリプト
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path


def create_processing_info(output_dir, input_dir=None, method='percentile', normalization='minmax', sample_size=None):
    """
    処理情報ファイルを作成
    
    Args:
        output_dir: 出力ディレクトリ
        input_dir: 入力ディレクトリ（オプション）
        method: 外れ値除去方法
        normalization: 正規化方法
        sample_size: サンプルサイズ
    """
    if not os.path.exists(output_dir):
        print(f"エラー: ディレクトリが存在しません: {output_dir}")
        return
    
    # 統計量ファイルを読み込み
    stats_path = os.path.join(output_dir, 'statistics.json')
    stats = None
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    
    # 処理情報ファイルを作成
    process_info_path = os.path.join(output_dir, 'processing_info.txt')
    
    with open(process_info_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("データ処理情報\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【処理日時】\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if input_dir:
            f.write("【入力情報】\n")
            f.write(f"入力ディレクトリ: {input_dir}\n")
        f.write(f"出力ディレクトリ: {output_dir}\n\n")
        
        f.write("【処理方法】\n")
        f.write(f"外れ値除去方法: {method}\n")
        if method == 'percentile':
            f.write("  - パーセンタイル法: 1パーセンタイルと99パーセンタイルの範囲外をクリップ\n")
        elif method == 'iqr':
            f.write("  - IQR法: Q1 - 1.5*IQR と Q3 + 1.5*IQR の範囲外をクリップ\n")
        elif method == 'zscore':
            f.write("  - Z-score法: |z| > 3 の範囲外をクリップ\n")
        
        f.write(f"正規化方法: {normalization}\n")
        if normalization == 'minmax':
            f.write("  - Min-Max正規化: データを[0, 1]の範囲にスケール\n")
        elif normalization == 'standardize' or normalization == 'zscore':
            f.write("  - Z-score標準化: 平均0、標準偏差1に変換\n")
        elif normalization == 'robust_zscore':
            f.write("  - Robust Z-score標準化: 中央値0、MAD基準で変換\n")
        
        if sample_size:
            f.write(f"統計量計算用サンプルサイズ: {sample_size}\n")
        f.write("\n")
        
        if stats:
            f.write("【統計量】\n")
            f.write(f"平均値: {stats.get('mean', 'N/A'):.6f}\n")
            f.write(f"標準偏差: {stats.get('std', 'N/A'):.6f}\n")
            if 'median' in stats:
                f.write(f"中央値: {stats['median']:.6f}\n")
            if 'mad' in stats:
                f.write(f"MAD (Median Absolute Deviation): {stats['mad']:.6f}\n")
            f.write(f"最小値: {stats.get('min', 'N/A'):.6f}\n")
            f.write(f"最大値: {stats.get('max', 'N/A'):.6f}\n")
            if 'percentile_1' in stats:
                f.write(f"1パーセンタイル: {stats['percentile_1']:.6f}\n")
            if 'percentile_99' in stats:
                f.write(f"99パーセンタイル: {stats['percentile_99']:.6f}\n")
            if 'q1' in stats:
                f.write(f"第1四分位数 (Q1): {stats['q1']:.6f}\n")
            if 'q3' in stats:
                f.write(f"第3四分位数 (Q3): {stats['q3']:.6f}\n")
            if 'iqr' in stats:
                f.write(f"四分位範囲 (IQR): {stats['iqr']:.6f}\n")
            if 'num_samples' in stats:
                f.write(f"統計量計算に使用したサンプル数: {stats['num_samples']}\n")
            f.write("\n")
        
        f.write("【処理の詳細】\n")
        f.write("1. 全データファイルから統計量を計算（サンプリングを使用する場合あり）\n")
        f.write("2. 各データファイルに対して以下を実行:\n")
        f.write("   a. 外れ値を除去（指定された方法でクリップ）\n")
        f.write("   b. 正規化/標準化を適用\n")
        f.write("   c. 処理済みデータを保存\n")
        f.write("\n")
        
        # ファイル数を確認
        processed_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.npy')])
        f.write("【処理結果】\n")
        f.write(f"出力ファイル数: {len(processed_files)}\n")
        f.write(f"情報作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
    
    print(f"処理情報ファイルを作成しました: {process_info_path}")


def main():
    parser = argparse.ArgumentParser(description='既存の出力ディレクトリに処理情報ファイルを作成')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='出力ディレクトリ')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='入力ディレクトリ（オプション）')
    parser.add_argument('--method', type=str, default='percentile',
                       choices=['percentile', 'iqr', 'zscore'],
                       help='外れ値除去方法 (default: percentile)')
    parser.add_argument('--normalization', type=str, default='minmax',
                       choices=['minmax', 'standardize'],
                       help='正規化方法 (default: minmax)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='統計量計算用のサンプルサイズ')
    
    args = parser.parse_args()
    
    create_processing_info(
        args.output_dir,
        input_dir=args.input_dir,
        method=args.method,
        normalization=args.normalization,
        sample_size=args.sample_size
    )


if __name__ == "__main__":
    main()
