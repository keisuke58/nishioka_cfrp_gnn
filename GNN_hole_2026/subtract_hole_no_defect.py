"""
複数のフォルダからすべてのファイルに対してhole_no_defect_original.npyを引き算するスクリプト
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


def subtract_from_all_files(input_dirs, subtract_file, output_dir):
    """
    複数のフォルダからすべてのファイルに対して引き算を実行
    
    Args:
        input_dirs: 入力フォルダのリスト
        subtract_file: 引き算するファイルのパス
        output_dir: 出力ディレクトリ
    """
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 引き算するファイルを読み込む
    print(f"引き算用ファイルを読み込み中: {subtract_file}")
    try:
        subtract_data = np.load(subtract_file)
        print(f"  形状: {subtract_data.shape}, データ型: {subtract_data.dtype}")
    except Exception as e:
        print(f"エラー: 引き算用ファイルの読み込みに失敗しました: {e}")
        return
    
    # 全ファイルのリストを作成
    all_files_info = []
    
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"警告: ディレクトリが存在しません: {input_dir}")
            continue
        
        print(f"\nディレクトリをスキャン中: {input_dir}")
        npy_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
        print(f"  見つかったファイル数: {len(npy_files)}")
        
        for filename in npy_files:
            file_path = os.path.join(input_dir, filename)
            all_files_info.append((file_path, filename, input_dir))
    
    print(f"\n合計処理ファイル数: {len(all_files_info)}")
    
    # 処理情報を記録
    processed_count = 0
    error_count = 0
    error_files = []
    
    # 各ファイルを処理
    for file_path, filename, source_dir in tqdm(all_files_info, desc="処理中"):
        try:
            # ファイルを読み込む
            data = np.load(file_path)
            
            # 形状チェック
            if data.shape != subtract_data.shape:
                print(f"\n警告: {filename} の形状が一致しません: {data.shape} vs {subtract_data.shape}")
                error_count += 1
                error_files.append((filename, f"形状不一致: {data.shape} vs {subtract_data.shape}"))
                continue
            
            # 引き算を実行
            result = data - subtract_data
            
            # 出力ファイル名を決定（重複を避けるため、元のフォルダ名をプレフィックスに追加）
            source_folder_name = Path(source_dir).name
            # 既に同じ名前のファイルがある場合は、元のフォルダ名を追加
            output_filename = filename
            output_path = os.path.join(output_dir, output_filename)
            
            # ファイル名の重複チェック
            if os.path.exists(output_path):
                # 元のフォルダ名をプレフィックスに追加
                output_filename = f"{source_folder_name}_{filename}"
                output_path = os.path.join(output_dir, output_filename)
            
            # 結果を保存
            np.save(output_path, result)
            processed_count += 1
        
        except Exception as e:
            print(f"\nエラー: {filename} の処理に失敗しました: {e}")
            error_count += 1
            error_files.append((filename, str(e)))
            continue
    
    # 処理結果を表示
    print(f"\n処理完了!")
    print(f"  成功: {processed_count} ファイル")
    print(f"  エラー: {error_count} ファイル")
    print(f"  出力ディレクトリ: {output_dir}")
    
    # 処理情報をファイルに保存
    info_path = os.path.join(output_dir, 'processing_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("引き算処理情報\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【処理日時】\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("【入力情報】\n")
        f.write(f"引き算用ファイル: {subtract_file}\n")
        f.write(f"引き算用ファイルの形状: {subtract_data.shape}\n\n")
        
        f.write("【入力ディレクトリ】\n")
        for i, input_dir in enumerate(input_dirs, 1):
            f.write(f"{i}. {input_dir}\n")
        f.write("\n")
        
        f.write("【出力情報】\n")
        f.write(f"出力ディレクトリ: {output_dir}\n\n")
        
        f.write("【処理結果】\n")
        f.write(f"処理ファイル数: {len(all_files_info)}\n")
        f.write(f"成功: {processed_count} ファイル\n")
        f.write(f"エラー: {error_count} ファイル\n\n")
        
        if error_files:
            f.write("【エラーファイル一覧】\n")
            for filename, error_msg in error_files:
                f.write(f"  - {filename}: {error_msg}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"処理情報を保存: {info_path}")


def main():
    # 入力ディレクトリのリスト
    input_dirs = [
        '/home/nishioka/GNN/GNN_hole_2026/Defect_hole_4x8_Region1_21_npy',
        '/home/nishioka/GNN/GNN_hole_2026/Defect_hole_2x2_Region1_21_npy',
        '/home/nishioka/GNN/GNN_hole_2026/Defect_hole_4x4_Region1_21_npy',
        '/home/nishioka/GNN/GNN_hole_2026/DSPSS_8x8/Defect_hole_8x8_original'
    ]
    
    # 引き算するファイル
    subtract_file = '/home/nishioka/GNN/GNN_hole_2026/DSPSS_8x8/hole_no_defect_original.npy'
    
    # 出力ディレクトリ
    output_dir = '/home/nishioka/GNN/GNN_hole_2026/all_subtracted_hole_no_defect'
    
    print("=" * 80)
    print("引き算処理スクリプト")
    print("=" * 80)
    print(f"\n入力ディレクトリ数: {len(input_dirs)}")
    print(f"引き算用ファイル: {subtract_file}")
    print(f"出力ディレクトリ: {output_dir}\n")
    
    # 処理を実行
    subtract_from_all_files(input_dirs, subtract_file, output_dir)


if __name__ == "__main__":
    main()
