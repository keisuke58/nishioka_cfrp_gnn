"""
2x2と4x4の両方のディレクトリを外れ値除去して正規化するバッチスクリプト
"""

import subprocess
import sys
import os

def main():
    base_dir = "/home/nishioka/GNN/GNN_hole_2026"
    
    # 処理するディレクトリ
    dirs_to_process = [
        {
            'input': os.path.join(base_dir, "Defect_hole_2x2_Region1_21_npy_subtracted"),
            'name': '2x2'
        },
        {
            'input': os.path.join(base_dir, "Defect_hole_4x4_Region1_21_npy_subtracted"),
            'name': '4x4'
        }
    ]
    
    script_path = os.path.join(os.path.dirname(__file__), "normalize_with_outlier_removal.py")
    
    for dir_info in dirs_to_process:
        input_dir = dir_info['input']
        name = dir_info['name']
        
        if not os.path.exists(input_dir):
            print(f"警告: ディレクトリが存在しません: {input_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"{name} ディレクトリを処理中...")
        print(f"{'='*60}\n")
        
        # パーセンタイル法 + Min-Max正規化
        cmd = [
            sys.executable,
            script_path,
            '--input_dir', input_dir,
            '--method', 'percentile',
            '--normalization', 'minmax',
            '--sample_size', '1000'  # 統計量計算用のサンプルサイズ
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"\n{name} の処理が完了しました")
        except subprocess.CalledProcessError as e:
            print(f"\nエラー: {name} の処理に失敗しました: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("全ての処理が完了しました!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
