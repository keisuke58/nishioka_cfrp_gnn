"""
8x8版のデータを2x2/4x4と同じ形式に変換するスクリプト
L1とL2を結合して1ファイルにし、ファイル名も変更
"""

import os
import numpy as np
import argparse
import re
from pathlib import Path
from tqdm import tqdm


def extract_file_info(filename):
    """
    ファイル名から情報を抽出
    例: delta__Defect_L10_B1_el115_H8_W8__L1.npy
    """
    # delta__Defect_L10_B1_el115_H8_W8__L1.npy の形式
    match = re.search(r'Defect_L(\d+)_B(\d+)_el(\d+)_H(\d+)_W(\d+)__L(\d+)', filename)
    if match:
        return {
            'L': int(match.group(1)),
            'B': int(match.group(2)),
            'el': int(match.group(3)),
            'H': int(match.group(4)),
            'W': int(match.group(5)),
            'layer': int(match.group(6))
        }
    return None


def convert_8x8_to_standard_format(input_dir, output_dir):
    """
    8x8版のデータを標準形式に変換
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 全ファイルを取得
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    
    # L1とL2のペアを見つける
    file_pairs = {}
    for filename in all_files:
        info = extract_file_info(filename)
        if info is None:
            continue
        
        # キーを作成（L, B, el, H, W）
        key = (info['L'], info['B'], info['el'], info['H'], info['W'])
        
        if key not in file_pairs:
            file_pairs[key] = {}
        
        file_pairs[key][info['layer']] = filename
    
    print(f"見つかったファイルペア数: {len(file_pairs)}")
    
    # 各ペアを処理
    processed_count = 0
    skipped_count = 0
    
    for key, files in tqdm(file_pairs.items(), desc="変換中"):
        L, B, el, H, W = key
        
        # L1とL2の両方が存在するか確認
        if 1 not in files or 2 not in files:
            print(f"警告: L1またはL2が見つかりません: L{L}_B{B}_el{el}_H{H}_W{W}")
            skipped_count += 1
            continue
        
        # L1とL2を読み込み
        l1_path = os.path.join(input_dir, files[1])
        l2_path = os.path.join(input_dir, files[2])
        
        try:
            l1_data = np.load(l1_path)
            l2_data = np.load(l2_path)
            
            # 形状を確認
            if len(l1_data.shape) == 2:
                # 2次元配列の場合は1次元に変換
                l1_flat = l1_data.flatten()
            else:
                l1_flat = l1_data
            
            if len(l2_data.shape) == 2:
                l2_flat = l2_data.flatten()
            else:
                l2_flat = l2_data
            
            # L1とL2を結合（2x2/4x4と同じ形式）
            combined_data = np.concatenate([l1_flat, l2_flat])
            
            # ファイル名を変更: Defect_L10_B1_el115_H8_W8.npy
            output_filename = f"Defect_L{L}_B{B}_el{el}_H{H}_W{W}.npy"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存
            np.save(output_path, combined_data)
            processed_count += 1
        
        except Exception as e:
            print(f"エラー: L{L}_B{B}_el{el}_H{H}_W{W} の処理に失敗: {e}")
            skipped_count += 1
            continue
    
    print(f"\n変換完了!")
    print(f"  処理済み: {processed_count} ファイル")
    print(f"  スキップ: {skipped_count} ファイル")
    print(f"  出力先: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='8x8版のデータを標準形式に変換')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='入力ディレクトリ（8x8版）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='出力ディレクトリ（指定しない場合は自動生成）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"エラー: ディレクトリが存在しません: {args.input_dir}")
        return
    
    # 出力ディレクトリを設定
    if args.output_dir is None:
        input_dir_name = Path(args.input_dir).name
        args.output_dir = os.path.join(Path(args.input_dir).parent.parent, f"Defect_hole_8x8_Region1_21_npy_subtracted")
    
    print(f"入力ディレクトリ: {args.input_dir}")
    print(f"出力ディレクトリ: {args.output_dir}\n")
    
    convert_8x8_to_standard_format(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
